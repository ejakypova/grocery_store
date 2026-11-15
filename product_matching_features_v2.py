"""Признаки пар товаров для train_product_matching_v2.py."""
import re
from difflib import SequenceMatcher
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

UNKNOWN = {'', 'none', 'null', 'nan', 'nat', 'не определено', 'неизвестно', 'unknown', '-'}

def clean(value):
    if value is None:
        return ''
    try:
        if np.isnan(value):
            return ''
    except (TypeError, ValueError):
        pass
    value = re.sub(r'\s+', ' ', str(value).lower().replace('ё', 'е')).strip()
    return '' if value in UNKNOWN else value

def numeric(value):
    try:
        v = float(value)
        return v if np.isfinite(v) and v > 0 else None
    except (ValueError, TypeError):
        return None

def canonical_pair(a, b):
    return ' [SEP] '.join(sorted((clean(a), clean(b))))

def pair_features(row):
    a, b = clean(row['name_a']), clean(row['name_b'])
    ba, bb = clean(row['brand_a']), clean(row['brand_b'])
    ca, cb = clean(row['category_a']), clean(row['category_b'])
    ua, ub = clean(row['unit_a']), clean(row['unit_b'])
    qa, qb = numeric(row['quantity_a']), numeric(row['quantity_b'])
    ta, tb = set(a.split()), set(b.split())
    overlap = len(ta & tb) / max(1, len(ta | tb))
    # Наличие данных и сравнение значений разделены: unknown != mismatch.
    brand_known = bool(ba and bb)
    category_known = bool(ca and cb)
    unit_known = bool(ua and ub)
    quantity_known = qa is not None and qb is not None
    quantity_comparable = quantity_known and unit_known and ua == ub
    return [
        SequenceMatcher(None, a, b).ratio(), overlap,
        float(brand_known), float(brand_known and ba == bb), float(brand_known and ba != bb),
        float(category_known), float(category_known and ca == cb), float(category_known and ca != cb),
        float(unit_known), float(unit_known and ua == ub), float(unit_known and ua != ub),
        float(quantity_known), float(quantity_comparable),
        min(qa, qb) / max(qa, qb) if quantity_comparable else 0.0,
        float(quantity_comparable and abs(qa-qb) < 1e-6),
    ]

class PairFeatureExtractor(BaseEstimator, TransformerMixin):
    """Работает с DataFrame; объект можно загружать joblib при наличии этого модуля."""
    def __init__(self, max_features=15000):
        self.max_features = max_features

    def fit(self, X, y=None):
        self.vectorizer_ = TfidfVectorizer(analyzer='char', ngram_range=(2, 4),
                                           max_features=self.max_features, sublinear_tf=True)
        self.vectorizer_.fit([canonical_pair(a,b) for a,b in zip(X['name_a'], X['name_b'])])
        return self

    def transform(self, X):
        text = self.vectorizer_.transform([canonical_pair(a,b) for a,b in zip(X['name_a'], X['name_b'])])
        extra = np.array([pair_features(row) for _, row in X.iterrows()], dtype=np.float64)
        return hstack([text, csr_matrix(extra)], format='csr')
