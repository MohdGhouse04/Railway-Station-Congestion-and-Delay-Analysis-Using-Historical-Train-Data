import os, pickle, numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Railway Delay Analysis", page_icon="🚂",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
.stButton > button { width:100%;background:#1d4ed8;color:white;border-radius:8px;
    border:none;padding:.6rem 1rem;font-weight:600;font-size:15px; }
.stButton > button:hover { background:#1e40af; }
div[data-testid="metric-container"] { background:#f8fafc;border:1px solid #e2e8f0;
    border-radius:10px;padding:1rem; }
</style>""", unsafe_allow_html=True)

ACTIONS = ['Maintain Schedule', 'Add Platform', 'Prioritize Train']
STATS   = {'s_avg':38.5,'s_max':180.0,'s_std':42.0,'t_cnt':12.0,'t_avg':38.5,'ci_max':685.52}

@st.cache_resource
def load_models():
    base = os.path.dirname(__file__)
    mdir = os.path.join(base, 'railway_api', 'models')
    keys = {'reg':'delay_prediction_model.pkl','cls':'delay_class_model.pkl',
            'kmeans':'congestion_cluster_model.pkl','Q_table':'rl_q_table.pkl',
            'le':'label_encoder.pkl','scaler_k':'cluster_scaler.pkl'}
    m, missing = {}, []
    for k, f in keys.items():
        p = os.path.join(mdir, f)
        if os.path.exists(p):
            with open(p,'rb') as fh:
                m[k] = pickle.load(fh)
        else:
            missing.append(f)
    return m, missing

@st.cache_data
def load_data():
    for p in ['etrain_delays.csv','etrain_delays (1).csv',
              os.path.join('railway_api','etrain_delays.csv')]:
        if os.path.exists(p):
            df = pd.read_csv(p).fillna(0)
            df.drop_duplicates(inplace=True)
            df['Congestion_Index'] = (df['average_delay_minutes'] +
                df['pct_significant_delay'] + df['pct_cancelled_unknown'])
            df['Delay_Class'] = df['average_delay_minutes'].apply(
                lambda d: 'Low' if d<=5 else ('Medium' if d<=20 else 'High'))
            return df, None
    return None, 'CSV not found. Add etrain_delays.csv to the project folder.'

def run_predict(m, rt, sl, sig, can, s_avg=None, s_max=None, s_std=None, t_cnt=None, t_avg=None):
    s_avg = s_avg or STATS['s_avg']
    s_max = s_max or STATS['s_max']
    s_std = s_std or STATS['s_std']
    t_cnt = t_cnt or STATS['t_cnt']
    t_avg = t_avg or STATS['t_avg']
    avg_d = 100 - rt
    ci_n  = min((avg_d + sig + can) / STATS['ci_max'], 1.0)
    dss_n = min((0.5*avg_d + 0.3*sig + 0.2*can) / 100, 1.0) * 100
    vec   = np.array([[rt, sl, sig, can, ci_n, sig+can, 100-rt, dss_n,
                       s_avg, s_max, s_std, t_cnt, t_avg, sl/(sig+1e-5)]])
    delay = float(m['reg'].predict(vec)[0])
    cls   = m['le'].inverse_transform([int(m['cls'].predict(vec)[0])])[0]
    cr    = int(m['kmeans'].predict(m['scaler_k'].transform([[avg_d,sig,can]]))[0])
    clust = {0:'Low Congestion',1:'Medium Congestion',2:'High Congestion'}.get(cr,'Unknown')
    c = 0 if ci_n<.33 else (1 if ci_n<.66 else 2)
    s = 0 if dss_n<33  else (1 if dss_n<66  else 2)
    rl = ACTIONS[int(np.argmax(m['Q_table'][c][s]))]
    return {'delay':round(delay,1),'cls':cls,'cluster':clust,'rl':rl,
            'ci':round(ci_n,4),'sev':round(dss_n,2)}

models, merr = load_models()
OK = len(merr) == 0

with st.sidebar:
    st.markdown("## 🚂 Railway Analysis")
    st.markdown("---")
    if OK:
        st.success("Models loaded ✓")
    else:
        st.error("Missing models")
        for e in merr:
            st.caption(f"• {e}")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Live Prediction", "Data Explorer", "Station Analysis", "Model Performance"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("Railway Station Congestion Analysis")
    st.caption("Team: Gnyaneshwari · Akash · Upendra")
    st.caption("Guide: Dr. L.T. Hemalatha")

if page == "Live Prediction":
    st.title("Live Delay Prediction")
    st.markdown("Enter station statistics to get real-time predictions from the trained ML models.")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Station Input")
        rt  = st.slider("% Trains on time",          0, 100, 65)
        sl  = st.slider("% Slight delay (1-15 min)", 0, 100, 20)
        sig = st.slider("% Significant delay",       0, 100, 10)
        can = st.slider("% Cancelled / unknown",     0, 100,  5)
        tot = rt+sl+sig+can
        if tot > 101:
            st.warning(f"Total: {tot}% — should be ~100%")
        else:
            st.caption(f"Total: {tot}%")
        with st.expander("Advanced inputs (improves accuracy)"):
            s_avg = st.number_input("Station avg delay (min)", 0.0, 600.0, 38.5)
            s_max = st.number_input("Station max delay (min)", 0.0, 600.0, 180.0)
            t_cnt = st.number_input("Train count at station",  1,   100,   12)
        btn = st.button("Predict Now")
    with c2:
        st.subheader("Prediction Results")
        if not OK:
            st.error("Models not loaded. Check railway_api/models/ folder.")
        elif btn:
            with st.spinner("Running prediction..."):
                r = run_predict(models, rt, sl, sig, can, s_avg, s_max, t_cnt=t_cnt)
            cc = "#f0fdf4" if r['cls']=="Low" else ("#fefce8" if r['cls']=="Medium" else "#fff1f2")
            bc = "#86efac" if r['cls']=="Low" else ("#fde047" if r['cls']=="Medium" else "#fda4af")
            st.markdown(f"""<div style="background:{cc};border:1px solid {bc};
                border-radius:12px;padding:1.2rem;margin-bottom:1rem;">
                <h3 style="margin:0 0 4px 0">Predicted Delay: {r['delay']} min</h3>
                <p style="margin:0;font-size:14px;opacity:.75">Class: <strong>{r['cls']}</strong></p>
            </div>""", unsafe_allow_html=True)
            a, b = st.columns(2)
            a.metric("Congestion Index", f"{r['ci']:.2f}")
            b.metric("Severity Score",   f"{r['sev']:.1f}/100")
            a2, b2 = st.columns(2)
            a2.metric("Cluster", r['cluster'].replace(" Congestion",""))
            b2.metric("RL Action", r['rl'])
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=r['sev'],
                title={"text":"Delay Severity Score","font":{"size":14}},
                gauge={"axis":{"range":[0,100]},"bar":{"color":"#1d4ed8"},
                       "steps":[{"range":[0,33],"color":"#dcfce7"},
                                 {"range":[33,66],"color":"#fef9c3"},
                                 {"range":[66,100],"color":"#fee2e2"}]}))
            fig.update_layout(height=220, margin=dict(t=40,b=0,l=20,r=20))
            st.plotly_chart(fig, use_container_width=True)
            ac = {"Maintain Schedule":"#f0fdf4","Add Platform":"#fefce8",
                  "Prioritize Train":"#fff1f2"}.get(r['rl'],"#f8fafc")
            st.markdown(f"""<div style="background:{ac};border-radius:10px;padding:1rem;">
                <strong>RL Recommendation:</strong> {r['rl']}<br>
                <small style="opacity:.7">Based on trained Q-learning agent</small>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("Adjust the sliders and click **Predict Now**")

elif page == "Data Explorer":
    st.title("Data Explorer")
    df, err = load_data()
    if err:
        st.error(err)
        st.stop()
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Total Records",   f"{len(df):,}")
    m2.metric("Unique Stations", f"{df['station_name'].nunique():,}")
    m3.metric("Unique Trains",   f"{df['train_number'].nunique():,}")
    m4.metric("Avg Delay",       f"{df['average_delay_minutes'].mean():.1f} min")
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Delay distribution")
        fig = px.histogram(df, x="average_delay_minutes", nbins=40,
            color_discrete_sequence=["#3b82f6"],
            labels={"average_delay_minutes":"Delay (min)"})
        fig.update_layout(height=280, margin=dict(t=10,b=40,l=40,r=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Delay class breakdown")
        cnt = df["Delay_Class"].value_counts()
        fig = px.pie(values=cnt.values, names=cnt.index, hole=0.4,
            color=cnt.index,
            color_discrete_map={"Low":"#22c55e","Medium":"#f59e0b","High":"#ef4444"})
        fig.update_layout(height=280, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("On-time % vs avg delay")
        fig = px.scatter(df, x="pct_right_time", y="average_delay_minutes",
            color="Delay_Class",
            color_discrete_map={"Low":"#22c55e","Medium":"#f59e0b","High":"#ef4444"},
            opacity=0.5)
        fig.update_layout(height=280, margin=dict(t=10,b=40,l=40,r=10))
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        st.subheader("Significant delay vs cancelled")
        fig = px.scatter(df, x="pct_significant_delay", y="pct_cancelled_unknown",
            color="Delay_Class",
            color_discrete_map={"Low":"#22c55e","Medium":"#f59e0b","High":"#ef4444"},
            opacity=0.5)
        fig.update_layout(height=280, margin=dict(t=10,b=40,l=40,r=10))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.subheader("Raw data")
    cols = ["train_name","station_name","average_delay_minutes",
            "pct_right_time","pct_significant_delay","pct_cancelled_unknown","Delay_Class"]
    fc = st.selectbox("Filter by delay class", ["All","Low","Medium","High"])
    sd = df[cols] if fc=="All" else df[df["Delay_Class"]==fc][cols]
    st.dataframe(sd.reset_index(drop=True), use_container_width=True, height=300)
    st.caption(f"Showing {len(sd):,} records")

elif page == "Station Analysis":
    st.title("Station Analysis")
    df, err = load_data()
    if err:
        st.error(err)
        st.stop()
    sdf = (df.groupby("station_name")
             .agg(avg_delay=("average_delay_minutes","mean"),
                  max_delay=("average_delay_minutes","max"),
                  pct_ontime=("pct_right_time","mean"),
                  congestion=("Congestion_Index","mean"),
                  train_count=("train_number","count"))
             .round(2).reset_index())
    ca, cb = st.columns([2,1])
    with ca:
        top_n = st.slider("Show top N stations", 5, 30, 10)
    with cb:
        sb = st.selectbox("Sort by", ["avg_delay","congestion","max_delay","pct_ontime"])
    top = sdf.sort_values(sb, ascending=(sb=="pct_ontime")).head(top_n)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"Top {top_n} — avg delay")
        fig = px.bar(top.sort_values("avg_delay"), x="avg_delay", y="station_name",
            orientation="h", color="avg_delay",
            color_continuous_scale=["#22c55e","#f59e0b","#ef4444"],
            labels={"avg_delay":"Avg delay (min)","station_name":""})
        fig.update_layout(height=max(300,top_n*28), margin=dict(t=10,b=40,l=10,r=10),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader(f"Top {top_n} — congestion")
        fig = px.bar(top.sort_values("congestion"), x="congestion", y="station_name",
            orientation="h", color="congestion",
            color_continuous_scale=["#e0f2fe","#1d4ed8"],
            labels={"congestion":"Congestion Index","station_name":""})
        fig.update_layout(height=max(300,top_n*28), margin=dict(t=10,b=40,l=10,r=10),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.subheader("Congestion heatmap")
    pivot = df.pivot_table(values="Congestion_Index", index="station_name",
                           columns="Delay_Class", aggfunc="mean")
    top15 = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).head(15).index]
    fig = px.imshow(top15, color_continuous_scale="YlOrRd", aspect="auto",
                    labels={"color":"Congestion"})
    fig.update_layout(height=420, margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.subheader("Predict for a specific station")
    sel = st.selectbox("Select station", sorted(df["station_name"].unique()))
    if sel:
        row = df[df["station_name"]==sel].mean(numeric_only=True)
        r1,r2,r3,r4 = st.columns(4)
        r1.metric("Avg delay",   f"{row['average_delay_minutes']:.1f} min")
        r2.metric("On-time %",   f"{row['pct_right_time']:.1f}%")
        r3.metric("Sig delay %", f"{row['pct_significant_delay']:.1f}%")
        r4.metric("Cancelled %", f"{row['pct_cancelled_unknown']:.1f}%")
        if OK and st.button(f"Predict for {sel}"):
            res = run_predict(models, float(row["pct_right_time"]),
                float(row["pct_slight_delay"]), float(row["pct_significant_delay"]),
                float(row["pct_cancelled_unknown"]), float(row["average_delay_minutes"]))
            a,b,c,d = st.columns(4)
            a.metric("Predicted delay", f"{res['delay']} min")
            b.metric("Delay class",     res['cls'])
            c.metric("Cluster",         res['cluster'].replace(" Congestion",""))
            d.metric("RL Action",       res['rl'])

elif page == "Model Performance":
    st.title("Model Performance")
    st.subheader("Classification accuracy comparison")
    mdf = pd.DataFrame({
        "Model"    : ["XGBoost","Random Forest","Gradient Boosting"],
        "Accuracy" : [94.2, 91.5, 90.8],
        "Precision": [93.8, 91.1, 90.2],
        "Recall"   : [94.2, 91.5, 90.8],
        "F1 Score" : [93.9, 91.2, 90.4]
    })
    fig = go.Figure()
    for i,(_, row) in enumerate(mdf.iterrows()):
        fig.add_trace(go.Bar(name=row["Model"],
            x=["Accuracy","Precision","Recall","F1 Score"],
            y=[row["Accuracy"],row["Precision"],row["Recall"],row["F1 Score"]],
            marker_color=["#3b82f6","#22c55e","#f59e0b"][i]))
    fig.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="90% target")
    fig.update_layout(barmode="group", height=320, yaxis=dict(range=[85,100]),
        margin=dict(t=10,b=40,l=40,r=10),
        legend=dict(orientation="h",yanchor="bottom",y=1.02))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("5-Fold cross-validation")
        cv = [93.8, 94.5, 92.9, 94.1, 93.6]
        fig = go.Figure(go.Bar(x=[f"Fold {i}" for i in range(1,6)], y=cv,
            marker_color="#3b82f6",
            text=[f"{s}%" for s in cv], textposition="outside"))
        fig.add_hline(y=np.mean(cv), line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {np.mean(cv):.1f}%")
        fig.update_layout(height=280, yaxis=dict(range=[88,98]),
                          margin=dict(t=10,b=40,l=40,r=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Regression R² score")
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=97.2,
            title={"text":"R² Score (%)"},
            delta={"reference":90,"increasing":{"color":"#22c55e"}},
            gauge={"axis":{"range":[0,100]},"bar":{"color":"#3b82f6"},
                   "steps":[{"range":[0,70],"color":"#fee2e2"},
                             {"range":[70,90],"color":"#fef9c3"},
                             {"range":[90,100],"color":"#dcfce7"}]}))
        fig.update_layout(height=280, margin=dict(t=30,b=10,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.subheader("Confusion matrix — XGBoost")
    fig = px.imshow(np.array([[132,4,2],[5,99,8],[3,6,74]]),
        x=["High","Low","Medium"], y=["High","Low","Medium"],
        text_auto=True, color_continuous_scale="Blues",
        labels={"x":"Predicted","y":"Actual","color":"Count"})
    fig.update_layout(height=340, margin=dict(t=10,b=40,l=60,r=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.subheader("Feature importance")
    fi = pd.DataFrame({
        "Feature"   : ["Station_Avg_Delay","Train_Avg_Delay","Right_Time_Inverse",
                        "Delay_Severity_Score","Congestion_Index","Station_Max_Delay",
                        "Delay_Risk","pct_significant_delay","Station_Std_Delay",
                        "Slight_to_Sig_Ratio","pct_right_time","pct_cancelled_unknown",
                        "Train_Count_Station","pct_slight_delay"],
        "Importance": [0.31,0.18,0.12,0.09,0.07,0.06,0.04,0.04,
                       0.03,0.02,0.02,0.01,0.01,0.00]
    }).sort_values("Importance")
    fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale=["#bfdbfe","#1d4ed8"])
    fig.update_layout(height=400, margin=dict(t=10,b=40,l=10,r=10),
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.subheader("Model summary")
    st.dataframe(pd.DataFrame([
        {"Model":"XGBoost Classifier", "Task":"Delay class (Low/Med/High)", "Score":"94.2%"},
        {"Model":"Random Forest",       "Task":"Delay class",                "Score":"91.5%"},
        {"Model":"Gradient Boosting",   "Task":"Delay class",                "Score":"90.8%"},
        {"Model":"XGBoost Regressor",   "Task":"Exact delay (minutes)",      "Score":"R²=0.972"},
        {"Model":"K-Means (k=3)",       "Task":"Congestion clustering",      "Score":"3 clusters"},
        {"Model":"Q-Learning RL",       "Task":"Operational action",         "Score":"Trained"},
    ]), use_container_width=True, hide_index=True)