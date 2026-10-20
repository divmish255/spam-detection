"""Train the URL spam model from data/url_spam_classification.csv -> model.joblib, stats.json"""
import json, joblib, numpy as np, pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from features import feats, parts

d = pd.read_csv('data/url_spam_classification.csv').drop_duplicates('url').reset_index(drop=True)
d['host'] = d.url.map(lambda u: parts(u)[0].split(':')[0])

def fit(df):
    cv = CountVectorizer(analyzer=feats, binary=True, min_df=3)
    X = cv.fit_transform(df.url)
    return cv, LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000).fit(X, df.is_spam)

tr, te = next(GroupShuffleSplit(1, test_size=.2, random_state=0).split(d, groups=d.host))
cv, m = fit(d.iloc[tr])
p = m.predict_proba(cv.transform(d.url.iloc[te]))[:, 1]; y = d.is_spam.iloc[te]
metrics = dict(precision=precision_score(y, p > .5), recall=recall_score(y, p > .5), auc=roc_auc_score(y, p),
               note='held-out evaluation on domains never seen in training')
print(metrics)

cv, m = fit(d)
hs = d.groupby('host').is_spam.agg(['size', 'sum'])
domain_stats = {h: [int(r['size']), int(r['sum'])] for h, r in hs.iterrows()}
joblib.dump(dict(cv=cv, model=m, domains=domain_stats), 'model.joblib', compress=3)
top = hs[hs['size'] >= 30].sort_values('size', ascending=False).head(15)
json.dump(dict(rows=len(d), spam=int(d.is_spam.sum()), hosts=len(hs), metrics=metrics,
               top_domains=[dict(host=h, n=int(r['size']), spam_rate=round(r['sum'] / r['size'], 3)) for h, r in top.iterrows()]),
          open('stats.json', 'w'), indent=1)
