"""
test_api.py — Test all Railway API endpoints
=============================================
Run AFTER starting app.py:
    python test_api.py
"""

import json
import urllib.request
import urllib.error

BASE = 'http://127.0.0.1:5000'

def call(method, path, body=None):
    url  = BASE + path
    data = json.dumps(body).encode() if body else None
    req  = urllib.request.Request(
        url, data=data,
        headers={'Content-Type': 'application/json'},
        method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as res:
            result = json.loads(res.read())
            return res.status, result
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
    except Exception as e:
        return None, str(e)


def print_result(label, status, result):
    ok = '✅' if status and 200 <= status < 300 else '❌'
    print(f'\n{ok}  {label}  (HTTP {status})')
    print(json.dumps(result, indent=2))


print('='*55)
print('  Railway API — Test Suite')
print('='*55)

# 1. Health check
status, result = call('GET', '/')
print_result('GET /', status, result)

# 2. Full prediction — high congestion
status, result = call('POST', '/predict', {
    'pct_right_time'        : 40.0,
    'pct_slight_delay'      : 25.0,
    'pct_significant_delay' : 25.0,
    'pct_cancelled_unknown' : 10.0
})
print_result('POST /predict  (high congestion)', status, result)

# 3. Full prediction — low congestion
status, result = call('POST', '/predict', {
    'pct_right_time'        : 95.0,
    'pct_slight_delay'      : 3.0,
    'pct_significant_delay' : 1.0,
    'pct_cancelled_unknown' : 1.0
})
print_result('POST /predict  (low congestion)', status, result)

# 4. Delay only
status, result = call('POST', '/predict/delay', {
    'pct_right_time'        : 70.0,
    'pct_slight_delay'      : 15.0,
    'pct_significant_delay' : 10.0,
    'pct_cancelled_unknown' : 5.0
})
print_result('POST /predict/delay', status, result)

# 5. Class only
status, result = call('POST', '/predict/class', {
    'pct_right_time'        : 55.0,
    'pct_slight_delay'      : 20.0,
    'pct_significant_delay' : 18.0,
    'pct_cancelled_unknown' : 7.0
})
print_result('POST /predict/class', status, result)

# 6. Stations list
status, result = call('GET', '/stations?top=5')
print_result('GET /stations?top=5', status, result)

# 7. Validation error test
status, result = call('POST', '/predict', {
    'pct_right_time': 110,   # invalid — > 100
    'pct_slight_delay': 0,
    'pct_significant_delay': 0,
    'pct_cancelled_unknown': 0
})
print_result('POST /predict  (validation error — should be 422)', status, result)

# 8. Missing field test
status, result = call('POST', '/predict', {
    'pct_right_time': 80
    # missing 3 required fields
})
print_result('POST /predict  (missing fields — should be 422)', status, result)

print('\n' + '='*55)
print('  Tests complete')
print('='*55)
