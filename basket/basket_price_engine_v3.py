"""ML-поиск кандидатов для Умной корзины. Только чтение PostgreSQL.

Требуются рядом: category_model_optimized.joblib, product_matching_model.joblib,
product_matching_features_v3.py, .env.

Примеры:
 python basket_price_engine_v3.py --search "молоко" --limit 20
 python basket_price_engine_v3.py --basket "молоко,сахар" --export basket_v3.csv
 python basket_price_engine_v3.py --search "молоко" --include-uncategorized
"""
import argparse
import os
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
from product_matching_features_v3 import same_conflict

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / '.env')
SQL = """
WITH latest AS (
 SELECT h.*, ROW_NUMBER() OVER (
   PARTITION BY h.product_id ORDER BY h.collected_at DESC NULLS LAST, h.history_id DESC
 ) AS rn
 FROM grocery_clean.price_history_clean h
 WHERE h.eligible_price IS TRUE AND h.price_som > 0
 AND (%(mode)s='all' OR (%(mode)s='public' AND h.source_status='public_page')
                    OR (%(mode)s='demo' AND h.source_status='demo_catalog'))
)
SELECT p.product_id, s.name AS store, p.original_name, p.name_clean,
 p.brand, p.category, p.total_quantity, p.normalized_unit,
 h.price_som, h.price_per_unit, h.collected_at, h.source_status
FROM latest h
JOIN grocery_clean.products_clean p ON p.product_id=h.product_id
JOIN public.stores s ON s.id=p.store_id
WHERE h.rn=1
"""
COLS = ['name_a','name_b','brand_a','brand_b','category_a','category_b',
        'quantity_a','quantity_b','unit_a','unit_b']
INVALID = {'','none','null','nan','nat','не определено','unknown','неизвестно'}

def clean(x):
    if x is None or pd.isna(x): return ''
    s = re.sub(r'\s+',' ',str(x).lower().replace('ё','е')).strip()
    return '' if s in INVALID else s

def connection():
    cfg = dict(host=os.getenv('DB_HOST','localhost'),port=int(os.getenv('DB_PORT','5432')),
               dbname=os.getenv('DB_NAME','grocery_store'),user=os.getenv('DB_USER','postgres'))
    if os.getenv('DB_PASSWORD'):cfg['password']=os.getenv('DB_PASSWORD')
    return psycopg.connect(**cfg)

def load_prices(mode):
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(SQL,{'mode':mode})
            df=pd.DataFrame(cur.fetchall(),columns=[x.name for x in cur.description])
    if df.empty:return df
    for c in ('price_som','price_per_unit','total_quantity'):
        df[c]=pd.to_numeric(df[c],errors='coerce')
    df['display_name']=df.original_name.fillna('').replace('',pd.NA).fillna(df.name_clean).fillna('')
    df['name_key']=df.display_name.map(clean)
    return df

def models():
    category=joblib.load(BASE/'category_model_optimized.joblib')
    matching=joblib.load(BASE/'product_matching_model.joblib')
    if not {'features','model'}.issubset(matching.named_steps):
        raise ValueError('Неожиданный формат модели сопоставления')
    return category,matching

def find(df,query,category_model,matching_model,limit=20,include_uncategorized=False,
         category_threshold=.45,same_threshold=.8):
    """ML категории + ML сопоставления. Вывод кандидатов, не доказанных аналогов."""
    if df.empty or not clean(query):return pd.DataFrame()
    query_category=str(category_model.predict([clean(query)])[0])
    query_conf=float(np.max(category_model.predict_proba([clean(query)])))
    # Категория запроса с низкой уверенностью не должна автоматически фильтровать каталог.
    if query_conf < category_threshold:
        print(f'ВНИМАНИЕ: категория запроса неуверенная ({query_category}: {query_conf:.2f}).')
        return pd.DataFrame()
    known=df.category.map(clean)
    allowed=known.eq(clean(query_category))
    missing=known.eq('')
    pool=df.loc[allowed | (missing if include_uncategorized else False)].copy()
    if pool.empty:return pool
    pool['category_source']=np.where(pool.category.map(clean).eq(''),'ml_predicted','database')
    # Неизвестные категории классифицируются ML, но не выдаются за проверенную разметку.
    unknown=pool.category_source.eq('ml_predicted')
    if unknown.any():
        names=pool.loc[unknown,'name_key'].tolist()
        pred=category_model.predict(names)
        conf=category_model.predict_proba(names).max(axis=1)
        pool.loc[unknown,'category']=pred
        pool.loc[unknown,'category_confidence']=conf
        pool=pool.loc[~unknown | ((pool.category.map(clean)==clean(query_category)) &
                                   (pool.category_confidence.fillna(1)>=category_threshold))].copy()
    if pool.empty:return pool
    # Не сравниваем текстовый запрос с каталогом как «same»: у запроса может не быть бренда/веса.
    # Вместо этого ML ранжирует пары с опорным товаром из каталога.
    query_words=set(clean(query).split())
    pool['query_overlap']=pool.name_key.map(lambda s: len(query_words & set(s.split()))/max(1,len(query_words)))
    pool=pool.loc[pool.query_overlap.gt(0)].copy()
    if pool.empty:return pool
    pool=pool.sort_values(['query_overlap','price_som'],ascending=[False,True]).head(250).copy()
    # Опорный товар выбирается из совпадений по названию; при отсутствии точного названия
    # результат носит ознакомительный характер.
    anchor=pool.iloc[0]
    pairs=pd.DataFrame([{
        'name_a':anchor.display_name,'name_b':r.display_name,
        'brand_a':anchor.brand,'brand_b':r.brand,
        'category_a':anchor.category,'category_b':r.category,
        'quantity_a':anchor.total_quantity,'quantity_b':r.total_quantity,
        'unit_a':anchor.normalized_unit,'unit_b':r.normalized_unit
    } for r in pool.itertuples(index=False)],columns=COLS)
    probs=matching_model.predict_proba(pairs)
    classes=list(matching_model.classes_)
    raw=matching_model.predict(pairs)
    pool['ml_label']=raw
    pool['same_probability']=probs[:,classes.index('same')]
    pool['needs_review']=[bool((r=='same' and (p<same_threshold or same_conflict(row))))
                          for r,p,(_,row) in zip(raw,pool.same_probability,pairs.iterrows())]
    pool['safe_label']=np.where(pool.needs_review,'review',pool.ml_label)
    pool['anchor_product_id']=int(anchor.product_id)
    pool['anchor_name']=anchor.display_name
    pool['query_category']=query_category
    pool['query_category_confidence']=query_conf
    pool['unit_price_label']=pool.normalized_unit.map(lambda u:'сом/кг' if u=='кг' else ('сом/л' if u=='л' else 'сом/ед.'))
    # Приоритет по совпадению слов, затем цене; не выдаём результаты за подтверждённую идентичность.
    return pool.sort_values(['query_overlap','price_som'],ascending=[False,True]).head(limit)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--search');g.add_argument('--basket')
    ap.add_argument('--mode',choices=['public','demo','all'],default='public')
    ap.add_argument('--include-uncategorized',action='store_true')
    ap.add_argument('--limit',type=int,default=20)
    ap.add_argument('--category-threshold',type=float,default=.45)
    ap.add_argument('--same-threshold',type=float,default=.8)
    ap.add_argument('--export')
    a=ap.parse_args()
    if a.limit<1 or not 0<=a.category_threshold<=1 or not 0<=a.same_threshold<=1:
        ap.error('Проверьте limit и пороги [0,1]')
    df=load_prices(a.mode)
    category_model,matching_model=models()
    print(f'Ценовых предложений: {len(df)}; источник: {a.mode}')
    items=[a.search] if a.search else [x.strip() for x in a.basket.split(',') if x.strip()]
    results=[]
    for item in items:
        r=find(df,item,category_model,matching_model,a.limit,a.include_uncategorized,
               a.category_threshold,a.same_threshold)
        if r.empty:
            print('Нет надёжных кандидатов для:',item)
            continue
        r.insert(0,'requested_item',item)
        results.append(r)
    cols=['requested_item','store','product_id','display_name','brand','category','category_source',
          'total_quantity','normalized_unit','price_som','price_per_unit','unit_price_label',
          'collected_at','source_status','anchor_product_id','anchor_name',
          'ml_label','safe_label','same_probability','needs_review','query_category_confidence']
    output=pd.concat(results,ignore_index=True) if results else pd.DataFrame(columns=cols)
    output=output[[c for c in cols if c in output.columns]]
    print(output.to_string(index=False,max_colwidth=40) if not output.empty else 'Совпадений нет')
    if a.export:
        output.to_csv(a.export,index=False,encoding='utf-8-sig')
        print('CSV сохранён:',a.export)
    print('ВАЖНО: ML-метки предварительные. Сравнение общей стоимости корзины не выполнено.')

if __name__=='__main__':main()
