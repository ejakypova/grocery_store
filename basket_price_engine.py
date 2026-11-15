"""Система получения данных о ценах на продукты (режим чтения).
Запуск : python basket_price_engine.py --search "молоко" - для одного товара
     python basket_price_engine.py --basket "молоко,сахар,масло" --mode public - для корзины товаров

"""
import argparse
import os
import re
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import psycopg
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / '.env')

SQL = """
WITH latest AS (
 SELECT h.*, ROW_NUMBER() OVER (
   PARTITION BY h.product_id ORDER BY h.collected_at DESC NULLS LAST, h.history_id DESC
 ) AS rn
 FROM grocery_clean.price_history_clean h
 WHERE h.eligible_price IS TRUE AND h.price_som > 0
)
SELECT p.product_id, s.name AS store, p.original_name, p.name_clean,
       p.brand, p.category, p.total_quantity, p.normalized_unit,
       h.price_som, h.price_per_unit, h.collected_at, h.source_status
FROM latest h
JOIN grocery_clean.products_clean p ON p.product_id=h.product_id
JOIN public.stores s ON s.id=p.store_id
WHERE h.rn=1
"""

def connection():
    kwargs = dict(host=os.getenv('DB_HOST','localhost'),
                  port=int(os.getenv('DB_PORT','5432')),
                  dbname=os.getenv('DB_NAME','grocery_store'),
                  user=os.getenv('DB_USER','postgres'))
    if os.getenv('DB_PASSWORD'):
        kwargs['password'] = os.getenv('DB_PASSWORD')
    return psycopg.connect(**kwargs)

def load_prices(mode='public'):
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(SQL)
            data = cur.fetchall()
            columns = [c.name for c in cur.description]
    df = pd.DataFrame(data, columns=columns)
    if df.empty:
        return df
    if mode == 'public':
        df = df[df.source_status.eq('public_page')].copy()
    elif mode == 'demo':
        df = df[df.source_status.eq('demo_catalog')].copy()
    elif mode != 'all':
        raise ValueError('mode must be public, demo or all')
    df['price_som'] = pd.to_numeric(df.price_som, errors='coerce')
    df['price_per_unit'] = pd.to_numeric(df.price_per_unit, errors='coerce')
    df['display_name'] = df.original_name.fillna('').replace('', pd.NA).fillna(df.name_clean).fillna('')
    df['name_key'] = df.display_name.map(normalize)
    return df.reset_index(drop=True)

def normalize(value):
    return re.sub(r'\s+', ' ', re.sub(r'[^\wа-яё]+', ' ', str(value).lower())).strip()

def search_products(df, query, limit=15):
    if df.empty or not normalize(query):
        return df.head(0)
    tokens = normalize(query).split()
    mask = df.name_key.map(lambda name: all(token in name for token in tokens))
    result = df.loc[mask].sort_values(['price_som','store','product_id'])
    return result.head(limit).drop(columns=['name_key'])

def basket_candidates(df, items, limit_per_item=10):
    """Return candidate offers, NOT a mathematically valid basket total.
    Matching names alone cannot establish equivalence or required quantity.
    """
    parts = []
    for item in items:
        found = search_products(df, item, limit_per_item).copy()
        found.insert(0, 'requested_item', item)
        parts.append(found)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--search', help='Product search phrase')
    parser.add_argument('--basket', help='Comma-separated requested items')
    parser.add_argument('--mode', choices=['public','demo','all'], default='public')
    parser.add_argument('--limit', type=int, default=15)
    parser.add_argument('--export', help='Optional CSV output path')
    args = parser.parse_args()
    df = load_prices(args.mode)
    print(f'Доступно последних допустимых цен ({args.mode}): {len(df)}')
    if args.search:
        result = search_products(df, args.search, args.limit)
    elif args.basket:
        result = basket_candidates(df, [x.strip() for x in args.basket.split(',') if x.strip()], args.limit)
    else:
        print('Укажите --search или --basket'); return
    fields = ['requested_item','store','product_id','display_name','brand','category',
              'total_quantity','normalized_unit','price_som','price_per_unit','collected_at','source_status']
    result = result[[c for c in fields if c in result.columns]]
    if result.empty:
        print('Подходящих предложений не найдено.')
    else:
        print(result.to_string(index=False, max_colwidth=55))
    if args.export:
        result.to_csv(args.export, index=False, encoding='utf-8-sig')
        print('CSV сохранён:', args.export)
    print('ВАЖНО: совпадение слов не подтверждает идентичность товаров. Итог корзины пока не вычисляется.')

if __name__ == '__main__':
    main()
