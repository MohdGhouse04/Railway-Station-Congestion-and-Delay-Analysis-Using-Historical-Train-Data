"""
Railway Station Congestion & Delay Prediction — Flask REST API
==============================================================
Run:
    python app.py

Endpoints:
    GET  /              → health check
    POST /predict       → full prediction (delay class + minutes + cluster + RL action)
    POST /predict/delay → delay minutes only (regression)
    POST /predict/class → delay class only   (Low / Medium / High)
    GET  /stations      → list of known stations with avg stats
    GET  /docs          → API documentation page
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from datetime import datetime

app = Flask(__name__)

# ── Load Models ──────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

reg_model     = load_model('delay_prediction_model.pkl')
cls_model     = load_model('delay_class_model.pkl')
kmeans        = load_model('congestion_cluster_model.pkl')
Q_table       = load_model('rl_q_table.pkl')
le            = load_model('label_encoder.pkl')
scaler_k      = load_model('cluster_scaler.pkl')

MODELS_LOADED = all([m is not None for m in [reg_model, cls_model, kmeans, Q_table, le, scaler_k]])

# ── Constants ────────────────────────────────────────────────────────
ACTIONS = ['Maintain Schedule', 'Add Platform', 'Prioritize Train']

# Training dataset stats (fallback for station-level features if station unknown)
TRAIN_STATS = {
    'Station_Avg_Delay':   38.5,
    'Station_Max_Delay':  180.0,
    'Station_Std_Delay':   42.0,
    'Train_Count_Station': 12.0,
    'Train_Avg_Delay':     38.5,
    'Congestion_Index_Raw_Max': 685.52
}


# ── Helper Functions ─────────────────────────────────────────────────
def build_feature_vector(data: dict) -> np.ndarray:
    """
    Build the 14-feature vector from raw input.
    Required keys: pct_right_time, pct_slight_delay,
                   pct_significant_delay, pct_cancelled_unknown
    Optional keys: station_avg_delay, station_max_delay,
                   station_std_delay, train_count, train_avg_delay
    """
    rt  = float(data['pct_right_time'])
    sl  = float(data['pct_slight_delay'])
    sig = float(data['pct_significant_delay'])
    can = float(data['pct_cancelled_unknown'])

    avg_d   = 100 - rt
    ci_raw  = avg_d + sig + can
    ci_norm = min(ci_raw / TRAIN_STATS['Congestion_Index_Raw_Max'], 1.0)
    dr      = sig + can
    rti     = 100 - rt
    dss     = 0.5 * avg_d + 0.3 * sig + 0.2 * can
    dss_n   = min(dss / 100, 1.0) * 100
    ssr     = sl / (sig + 1e-5)

    s_avg = float(data.get('station_avg_delay',   TRAIN_STATS['Station_Avg_Delay']))
    s_max = float(data.get('station_max_delay',   TRAIN_STATS['Station_Max_Delay']))
    s_std = float(data.get('station_std_delay',   TRAIN_STATS['Station_Std_Delay']))
    t_cnt = float(data.get('train_count',         TRAIN_STATS['Train_Count_Station']))
    t_avg = float(data.get('train_avg_delay',     TRAIN_STATS['Train_Avg_Delay']))

    return np.array([[rt, sl, sig, can,
                      ci_norm, dr, rti, dss_n,
                      s_avg, s_max, s_std, t_cnt, t_avg, ssr]]), ci_norm, dss_n, avg_d, sig, can


def get_rl_action(ci_norm: float, dss_n: float) -> str:
    if Q_table is None:
        return 'Maintain Schedule'
    c = 0 if ci_norm < 0.33 else (1 if ci_norm < 0.66 else 2)
    s = 0 if dss_n   < 33   else (1 if dss_n   < 66   else 2)
    return ACTIONS[int(np.argmax(Q_table[c][s]))]


def validate_input(data: dict) -> str | None:
    """Returns error message string or None if valid."""
    required = ['pct_right_time', 'pct_slight_delay',
                'pct_significant_delay', 'pct_cancelled_unknown']
    for field in required:
        if field not in data:
            return f"Missing required field: '{field}'"
        try:
            val = float(data[field])
        except (ValueError, TypeError):
            return f"Field '{field}' must be a number"
        if not (0 <= val <= 100):
            return f"Field '{field}' must be between 0 and 100"

    total = (float(data['pct_right_time']) +
             float(data['pct_slight_delay']) +
             float(data['pct_significant_delay']) +
             float(data['pct_cancelled_unknown']))
    if total > 101:
        return f"Percentages sum to {total:.1f}% — should be ~100%"
    return None


# ── Routes ───────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status'       : 'ok',
        'service'      : 'Railway Delay Prediction API',
        'version'      : '1.0.0',
        'models_loaded': MODELS_LOADED,
        'timestamp'    : datetime.utcnow().isoformat() + 'Z',
        'endpoints'    : [
            'GET  /',
            'POST /predict',
            'POST /predict/delay',
            'POST /predict/class',
            'GET  /stations',
            'GET  /docs'
        ]
    })


@app.route('/predict', methods=['POST'])
def predict_full():
    """
    Full prediction — returns all 4 outputs.

    Request body (JSON):
    {
        "pct_right_time"        : 55.0,
        "pct_slight_delay"      : 20.0,
        "pct_significant_delay" : 18.0,
        "pct_cancelled_unknown" : 7.0,

        // optional — improves accuracy if you know the station
        "station_avg_delay"     : 42.0,
        "station_max_delay"     : 150.0,
        "station_std_delay"     : 35.0,
        "train_count"           : 15,
        "train_avg_delay"       : 40.0
    }

    Response:
    {
        "predicted_delay_minutes": 28.4,
        "delay_class"            : "High",
        "congestion_cluster"     : "Medium Congestion",
        "rl_recommendation"      : "Add Platform",
        "congestion_index"       : 0.72,
        "severity_score"         : 61.3
    }
    """
    if not MODELS_LOADED:
        return jsonify({'error': 'Models not loaded. Run the notebook first to generate .pkl files.'}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Request body must be JSON'}), 400

    err = validate_input(data)
    if err:
        return jsonify({'error': err}), 422

    try:
        vec, ci_norm, dss_n, avg_d, sig, can = build_feature_vector(data)

        # Regression
        delay_pred = float(reg_model.predict(vec)[0])

        # Classification
        class_enc   = int(cls_model.predict(vec)[0])
        class_label = le.inverse_transform([class_enc])[0]

        # Clustering
        clust_in  = scaler_k.transform([[avg_d, sig, can]])
        clust_raw = int(kmeans.predict(clust_in)[0])
        cluster_means_order = {0: 'Low Congestion', 1: 'Medium Congestion', 2: 'High Congestion'}
        clust_label = cluster_means_order.get(clust_raw, f'Cluster {clust_raw}')

        # RL
        rl_action = get_rl_action(ci_norm, dss_n)

        return jsonify({
            'predicted_delay_minutes': round(delay_pred, 1),
            'delay_class'            : class_label,
            'congestion_cluster'     : clust_label,
            'rl_recommendation'      : rl_action,
            'congestion_index'       : round(ci_norm, 4),
            'severity_score'         : round(dss_n, 2)
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/predict/delay', methods=['POST'])
def predict_delay_only():
    """Returns only the predicted delay in minutes (regression)."""
    if not MODELS_LOADED:
        return jsonify({'error': 'Models not loaded.'}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Request body must be JSON'}), 400

    err = validate_input(data)
    if err:
        return jsonify({'error': err}), 422

    try:
        vec, *_ = build_feature_vector(data)
        delay_pred = float(reg_model.predict(vec)[0])
        return jsonify({'predicted_delay_minutes': round(delay_pred, 1)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/class', methods=['POST'])
def predict_class_only():
    """Returns only the delay class (Low / Medium / High)."""
    if not MODELS_LOADED:
        return jsonify({'error': 'Models not loaded.'}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Request body must be JSON'}), 400

    err = validate_input(data)
    if err:
        return jsonify({'error': err}), 422

    try:
        vec, ci_norm, dss_n, *_ = build_feature_vector(data)
        class_enc   = int(cls_model.predict(vec)[0])
        class_label = le.inverse_transform([class_enc])[0]
        rl_action   = get_rl_action(ci_norm, dss_n)
        return jsonify({
            'delay_class'      : class_label,
            'rl_recommendation': rl_action
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stations', methods=['GET'])
def list_stations():
    """
    Returns list of known stations from training data with their stats.
    Optional query param: ?top=10  (returns top N most congested)
    """
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'etrain_delays.csv')
    if not os.path.exists(csv_path):
        return jsonify({'error': 'Dataset not found alongside app.py', 'stations': []}), 404

    try:
        df = pd.read_csv(csv_path).fillna(0)
        top_n = int(request.args.get('top', 50))

        stations = (
            df.groupby('station_name')
              .agg(
                  avg_delay=('average_delay_minutes', 'mean'),
                  max_delay=('average_delay_minutes', 'max'),
                  pct_right_time=('pct_right_time', 'mean'),
                  train_count=('train_number', 'count')
              )
              .sort_values('avg_delay', ascending=False)
              .head(top_n)
              .reset_index()
              .round(2)
        )

        return jsonify({
            'count'   : len(stations),
            'stations': stations.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/docs', methods=['GET'])
def docs():
    """Interactive API documentation page."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Railway API Docs</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; }
  .header { background: #1a1d2e; border-bottom: 1px solid #2d3748; padding: 20px 40px; display: flex; align-items: center; gap: 12px; }
  .badge { background: #22c55e; color: #052e16; font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 4px; }
  .header h1 { font-size: 20px; font-weight: 600; color: #f1f5f9; }
  .header p  { font-size: 13px; color: #94a3b8; margin-top: 2px; }
  .container { max-width: 860px; margin: 0 auto; padding: 32px 20px; }
  .endpoint   { background: #1a1d2e; border: 1px solid #2d3748; border-radius: 10px; margin-bottom: 20px; overflow: hidden; }
  .ep-header  { padding: 14px 20px; display: flex; align-items: center; gap: 10px; cursor: pointer; user-select: none; }
  .ep-header:hover { background: #212440; }
  .method     { font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 4px; min-width: 46px; text-align: center; }
  .get  { background: #0c4a6e; color: #7dd3fc; }
  .post { background: #14532d; color: #86efac; }
  .ep-path    { font-family: 'Consolas', monospace; font-size: 14px; color: #e2e8f0; }
  .ep-desc    { font-size: 13px; color: #94a3b8; margin-left: auto; }
  .ep-body    { padding: 16px 20px; border-top: 1px solid #2d3748; display: none; }
  .ep-body.open { display: block; }
  .section-label { font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 8px; }
  pre  { background: #0d0f1a; border: 1px solid #2d3748; border-radius: 6px; padding: 14px; font-size: 12px; overflow-x: auto; color: #a5f3fc; line-height: 1.6; }
  .try-btn { background: #4f46e5; color: #fff; border: none; padding: 8px 16px; border-radius: 6px; font-size: 13px; cursor: pointer; margin-top: 12px; }
  .try-btn:hover { background: #4338ca; }
  .result  { margin-top: 12px; background: #0d0f1a; border: 1px solid #2d3748; border-radius: 6px; padding: 14px; font-size: 12px; color: #86efac; min-height: 40px; display: none; white-space: pre-wrap; }
  .result.show { display: block; }
  textarea { width: 100%; background: #0d0f1a; border: 1px solid #2d3748; border-radius: 6px; padding: 10px; color: #e2e8f0; font-family: 'Consolas', monospace; font-size: 12px; resize: vertical; }
</style>
</head>
<body>
<div class="header">
  <div>
    <div style="display:flex;align-items:center;gap:10px">
      <h1>Railway Delay Prediction API</h1>
      <span class="badge">v1.0.0</span>
    </div>
    <p>Real-time delay prediction • Congestion clustering • RL operational recommendations</p>
  </div>
</div>

<div class="container">

  <!-- GET / -->
  <div class="endpoint">
    <div class="ep-header" onclick="toggle(this)">
      <span class="method get">GET</span>
      <span class="ep-path">/</span>
      <span class="ep-desc">Health check</span>
    </div>
    <div class="ep-body">
      <div class="section-label">Response</div>
      <pre>{"status":"ok","service":"Railway Delay Prediction API","models_loaded":true}</pre>
      <button class="try-btn" onclick="tryIt(this, '/', 'GET', null)">▶ Try it</button>
      <div class="result"></div>
    </div>
  </div>

  <!-- POST /predict -->
  <div class="endpoint">
    <div class="ep-header" onclick="toggle(this)">
      <span class="method post">POST</span>
      <span class="ep-path">/predict</span>
      <span class="ep-desc">Full prediction (all 4 outputs)</span>
    </div>
    <div class="ep-body">
      <div class="section-label">Request Body (JSON)</div>
      <textarea id="body-predict" rows="10">{
  "pct_right_time": 55.0,
  "pct_slight_delay": 20.0,
  "pct_significant_delay": 18.0,
  "pct_cancelled_unknown": 7.0
}</textarea>
      <button class="try-btn" onclick="tryIt(this, '/predict', 'POST', 'body-predict')">▶ Try it</button>
      <div class="result"></div>
    </div>
  </div>

  <!-- POST /predict/delay -->
  <div class="endpoint">
    <div class="ep-header" onclick="toggle(this)">
      <span class="method post">POST</span>
      <span class="ep-path">/predict/delay</span>
      <span class="ep-desc">Predicted delay minutes only</span>
    </div>
    <div class="ep-body">
      <div class="section-label">Request Body (JSON)</div>
      <textarea id="body-delay" rows="7">{
  "pct_right_time": 80.0,
  "pct_slight_delay": 12.0,
  "pct_significant_delay": 5.0,
  "pct_cancelled_unknown": 3.0
}</textarea>
      <button class="try-btn" onclick="tryIt(this, '/predict/delay', 'POST', 'body-delay')">▶ Try it</button>
      <div class="result"></div>
    </div>
  </div>

  <!-- POST /predict/class -->
  <div class="endpoint">
    <div class="ep-header" onclick="toggle(this)">
      <span class="method post">POST</span>
      <span class="ep-path">/predict/class</span>
      <span class="ep-desc">Delay class + RL recommendation</span>
    </div>
    <div class="ep-body">
      <div class="section-label">Request Body (JSON)</div>
      <textarea id="body-class" rows="7">{
  "pct_right_time": 40.0,
  "pct_slight_delay": 25.0,
  "pct_significant_delay": 25.0,
  "pct_cancelled_unknown": 10.0
}</textarea>
      <button class="try-btn" onclick="tryIt(this, '/predict/class', 'POST', 'body-class')">▶ Try it</button>
      <div class="result"></div>
    </div>
  </div>

  <!-- GET /stations -->
  <div class="endpoint">
    <div class="ep-header" onclick="toggle(this)">
      <span class="method get">GET</span>
      <span class="ep-path">/stations?top=10</span>
      <span class="ep-desc">Top congested stations</span>
    </div>
    <div class="ep-body">
      <p style="font-size:13px;color:#94a3b8;margin-bottom:10px">Returns station stats from training data. Use <code>?top=N</code> to limit results.</p>
      <button class="try-btn" onclick="tryIt(this, '/stations?top=5', 'GET', null)">▶ Try it (top 5)</button>
      <div class="result"></div>
    </div>
  </div>

</div>

<script>
function toggle(header) {
  const body = header.nextElementSibling;
  body.classList.toggle('open');
}
async function tryIt(btn, url, method, bodyId) {
  const result = btn.nextElementSibling;
  result.classList.add('show');
  result.style.color = '#94a3b8';
  result.textContent = 'Sending request...';
  try {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (bodyId) opts.body = document.getElementById(bodyId).value;
    const res  = await fetch(url, opts);
    const data = await res.json();
    result.style.color = res.ok ? '#86efac' : '#f87171';
    result.textContent = JSON.stringify(data, null, 2);
  } catch(e) {
    result.style.color = '#f87171';
    result.textContent = 'Error: ' + e.message;
  }
}
// Auto-open first endpoint
document.querySelector('.ep-body').classList.add('open');
</script>
</body>
</html>"""
    return html


# ── Error Handlers ───────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found', 'available': [
        'GET /', 'POST /predict', 'POST /predict/delay',
        'POST /predict/class', 'GET /stations', 'GET /docs'
    ]}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'error': 'Method not allowed'}), 405


if __name__ == '__main__':
    print('\n' + '='*55)
    print('  Railway Delay Prediction API')
    print('='*55)
    if MODELS_LOADED:
        print('  ✅ All 6 models loaded successfully')
    else:
        print('  ⚠️  Models not found in ./models/')
        print('  Run the notebook first to generate .pkl files')
        print('  then copy them into railway_api/models/')
    print('\n  Docs  →  http://127.0.0.1:5000/docs')
    print('  API   →  http://127.0.0.1:5000/predict')
    print('='*55 + '\n')
    app.run(debug=True, port=5000)
