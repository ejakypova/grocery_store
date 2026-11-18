"""Признаки для сопоставления товаров."""
import re
import numpy as np
from difflib import SequenceMatcher
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

UNKNOWN={'','none','null','nan','nat','не определено','неизвестно','unknown','-'}
def clean(x):
    if x is None: return ''
    s=re.sub(r'\s+',' ',str(x).lower().replace('ё','е')).strip()
    return '' if s in UNKNOWN else s
def num(x):
    try:
        f=float(x)
        return f if np.isfinite(f) and f>0 else None
    except (TypeError,ValueError):return None

def canonical_pair(a,b):return ' [SEP] '.join(sorted((clean(a),clean(b))))
def name_properties(s):
    s=clean(s)
    fat=re.search(r'(?<!\d)(\d{1,2}(?:[.,]\d+)?)\s*%',s)
    return float(fat.group(1).replace(',','.')) if fat else None

def extra(row):
    a,b=clean(row['name_a']),clean(row['name_b'])
    ba,bb=clean(row['brand_a']),clean(row['brand_b'])
    ca,cb=clean(row['category_a']),clean(row['category_b'])
    ua,ub=clean(row['unit_a']),clean(row['unit_b'])
    qa,qb=num(row['quantity_a']),num(row['quantity_b'])
    fa,fb=name_properties(a),name_properties(b)
    brand=bool(ba and bb);category=bool(ca and cb)
    unit=bool(ua and ub);quantity=qa is not None and qb is not None
    comparable=quantity and unit and ua==ub
    fat_known=fa is not None and fb is not None
    ta,tb=set(a.split()),set(b.split())
    return [SequenceMatcher(None,a,b).ratio(),len(ta&tb)/max(1,len(ta|tb)),
            float(brand),float(brand and ba==bb),float(brand and ba!=bb),
            float(category),float(category and ca==cb),float(category and ca!=cb),
            float(unit),float(unit and ua==ub),float(unit and ua!=ub),
            float(quantity),float(comparable),min(qa,qb)/max(qa,qb) if comparable else 0.,
            float(comparable and abs(qa-qb)<1e-6),
            float(fat_known),float(fat_known and fa==fb),float(fat_known and fa!=fb)]

class PairFeatureExtractorV3(BaseEstimator,TransformerMixin):
    def __init__(self,max_features=15000):self.max_features=max_features
    def fit(self,X,y=None):
        self.vectorizer_=TfidfVectorizer(analyzer='char',ngram_range=(2,4),max_features=self.max_features,sublinear_tf=True)
        self.vectorizer_.fit([canonical_pair(a,b) for a,b in zip(X.name_a,X.name_b)])
        return self
    def transform(self,X):
        text=self.vectorizer_.transform([canonical_pair(a,b) for a,b in zip(X.name_a,X.name_b)])
        dense=np.asarray([extra(row) for _,row in X.iterrows()],dtype=float)
        return hstack([text,csr_matrix(dense)],format='csr')

def same_conflict(row):
    """Conservative rule: explicit known differences forbid automatic 'same'."""
    ba,bb=clean(row['brand_a']),clean(row['brand_b'])
    ua,ub=clean(row['unit_a']),clean(row['unit_b'])
    qa,qb=num(row['quantity_a']),num(row['quantity_b'])
    fa,fb=name_properties(row['name_a']),name_properties(row['name_b'])
    return bool((ba and bb and ba!=bb) or
                (ua and ub and ua!=ub) or
                (ua and ub and ua==ub and qa is not None and qb is not None and abs(qa-qb)>1e-6) or
                (fa is not None and fb is not None and fa!=fb))
