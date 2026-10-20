import json, joblib, numpy as np
from flask import Flask, jsonify, request, send_from_directory
from features import feats, parts

app = Flask(__name__, static_folder='static')
bundle = joblib.load('model.joblib')
cv, model, domains = bundle['cv'], bundle['model'], bundle['domains']
stats = json.load(open('stats.json'))
inv = {i: k for k, i in cv.vocabulary_.items()}
coef = model.coef_[0]

def normalize(u):
    u = (u or '').strip()
    if u and '://' not in u:
        u = 'https://' + u
    return u

def check(u):
    u = normalize(u)
    if not u or len(u) > 2048:
        return None
    X = cv.transform([u])
    p = float(model.predict_proba(X)[0, 1])
    # top contributing features
    idx = X.indices
    contrib = sorted(((float(coef[i]), inv[i]) for i in idx), key=lambda t: -abs(t[0]))[:6]
    host = parts(u)[0].split(':')[0]
    d = domains.get(host)
    return dict(url=u, host=host, spam_probability=round(p, 4), is_spam=p >= .5,
                label='spam' if p >= .5 else 'not spam',
                domain_history=dict(seen=d[0], spam=d[1], spam_rate=round(d[1] / d[0], 3)) if d else None,
                reasons=[dict(feature=f, weight=round(w, 2), direction='spam' if w > 0 else 'not spam') for w, f in contrib])

@app.get('/')
def index():
    return send_from_directory('static', 'index.html')

@app.get('/api/health')
def health():
    return jsonify(status='ok')

@app.get('/api/stats')
def get_stats():
    return jsonify(stats)

@app.post('/api/check')
def api_check():
    r = check((request.get_json(silent=True) or {}).get('url'))
    return (jsonify(r), 200) if r else (jsonify(error='Provide a valid "url" (max 2048 chars).'), 400)

@app.post('/api/batch')
def api_batch():
    urls = (request.get_json(silent=True) or {}).get('urls')
    if not isinstance(urls, list) or not urls or len(urls) > 500:
        return jsonify(error='Provide "urls": a list of 1-500 URLs.'), 400
    return jsonify(results=[check(u) or dict(url=u, error='invalid') for u in urls])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
