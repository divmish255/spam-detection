# URL Spam Checker
Flask backend + web UI trained on `data/url_spam_classification.csv`.

    pip install -r requirements.txt
    python train.py      # builds model.joblib + stats.json (already included)
    python app.py        # http://localhost:8000

API: `POST /api/check {"url": "..."}`, `POST /api/batch {"urls": [...]}`, `GET /api/stats`, `GET /api/health`.
