"""Incremental public -> grocery_clean. No DDL, DELETE or UPDATE.

Run: python clean_to_postgres_v2.py           # preview, rollback
     python clean_to_postgres_v2.py --apply   # insert new rows
Requires existing grocery_clean.products_clean and price_history_clean.
"""
import argparse
import getpass
import importlib.util
import json
from decimal import Decimal
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

MAX_PRICE = Decimal('100000')
PRODUCT_COLUMNS = {'product_id', 'store_id', 'original_name', 'name_clean', 'brand', 'category',
                   'total_quantity', 'normalized_unit', 'pack_count', 'attributes',
                   'needs_review', 'review_reasons', 'rules_version'}
PRICE_COLUMNS = {'history_id', 'product_id', 'price_som', 'old_price_som', 'collected_at',
                 'source_status', 'price_per_unit', 'eligible_price', 'review_reasons'}


def load_cleaner():
    folder = Path(__file__).resolve().parent
    for name in ('clean_grocery_store.py'):
        path = folder / name
        if path.exists():
            spec = importlib.util.spec_from_file_location('grocery_clean_rules', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError('Положите clean_grocery_store.py рядом с этим скриптом')


def safe_number(value):
    if value is None:
        return None
    try:
        result = Decimal(str(value))
        return result if result.is_finite() else None
    except (ValueError, ArithmeticError):
        return None


def ensure_columns(cur, schema, table, required):
    cur.execute('''SELECT column_name FROM information_schema.columns
                   WHERE table_schema=%s AND table_name=%s''', (schema, table))
    actual = {r['column_name'] for r in cur.fetchall()}
    missing = required - actual
    if missing:
        raise RuntimeError(f'{schema}.{table}: отсутствуют колонки {sorted(missing)}; структура не изменена')


def price_key(row, cleaned=False):
    return (int(row['product_id']), safe_number(row['price_som']),
            safe_number(row['old_price_som']), row['collected_at'],
            row['source_status'] if cleaned else row['status'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--apply', action='store_true', help='Записать изменения; без параметра только проверка')
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=5432)
    ap.add_argument('--database', default='grocery_store')
    ap.add_argument('--user', default='postgres')
    args = ap.parse_args()
    cleaner = load_cleaner()
    password = getpass.getpass('Пароль PostgreSQL: ')
    stats = {'new_products': 0, 'existing_products_skipped': 0,
             'new_price_observations': 0, 'duplicate_prices_skipped': 0,
             'price_id_conflicts': 0, 'eligible_new_prices': 0}

    with psycopg.connect(host=args.host, port=args.port, dbname=args.database,
                         user=args.user, password=password, row_factory=dict_row) as conn:
        try:
            with conn.cursor() as cur:
                cur.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
                cur.execute('SELECT pg_advisory_xact_lock(72603142)')
                ensure_columns(cur, 'grocery_clean', 'products_clean', PRODUCT_COLUMNS)
                ensure_columns(cur, 'grocery_clean', 'price_history_clean', PRICE_COLUMNS)
                ensure_columns(cur, 'public', 'products', {'id', 'store_id', 'name'})
                ensure_columns(cur, 'public', 'price_history', {'id', 'product_id', 'price_som',
                               'old_price_som', 'collected_at', 'status'})
                cur.execute('SELECT * FROM public.products ORDER BY id')
                source_products = cur.fetchall()
                cur.execute('SELECT * FROM public.price_history ORDER BY id')
                source_prices = cur.fetchall()
                cur.execute('SELECT * FROM grocery_clean.products_clean')
                existing_products = {int(r['product_id']): r for r in cur.fetchall()}
                cur.execute('SELECT history_id, product_id, price_som, old_price_som, collected_at, source_status FROM grocery_clean.price_history_clean')
                existing_prices = cur.fetchall()
                existing_price_ids = {int(r['history_id']) for r in existing_prices}
                seen_prices = {price_key(r, cleaned=True) for r in existing_prices}

                confirmed = {}
                cur.execute("SELECT to_regclass('grocery_clean.product_category_labels') AS tbl")
                if cur.fetchone()['tbl'] is not None:
                    ensure_columns(cur, 'grocery_clean', 'product_category_labels',
                                   {'product_id', 'category', 'review_status'})
                    cur.execute('''SELECT product_id, category FROM grocery_clean.product_category_labels
                                   WHERE review_status = %s''', ('подтверждено',))
                    confirmed = {int(r['product_id']): r['category'] for r in cur.fetchall()}

                products = dict(existing_products)
                for source in source_products:
                    pid = int(source['id'])
                    if pid in products:
                        stats['existing_products_skipped'] += 1
                        continue  # Preserve manual edits, category, brand and existing metadata.
                    p = cleaner.extract_product(source, forced_category=confirmed.get(pid))
                    quantity = safe_number(p['total_quantity'])
                    attributes = p.get('attributes') or {}
                    if isinstance(attributes, str):
                        attributes = json.loads(attributes)
                    new_row = dict(product_id=pid, store_id=p['store_id'], original_name=p['original_name'],
                                   name_clean=p['name_clean'], brand=p['brand'], category=p['category'],
                                   total_quantity=quantity, normalized_unit=p['normalized_unit'],
                                   pack_count=p['pack_count'], attributes=attributes,
                                   needs_review=p['needs_review'], review_reasons=p['review_reasons'],
                                   rules_version=p['rules_version'])
                    products[pid] = new_row
                    if args.apply:
                        cur.execute('''INSERT INTO grocery_clean.products_clean
                            (product_id,store_id,original_name,name_clean,brand,category,total_quantity,
                             normalized_unit,pack_count,attributes,needs_review,review_reasons,rules_version)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            ON CONFLICT (product_id) DO NOTHING''',
                            (pid, p['store_id'], p['original_name'], p['name_clean'], p['brand'],
                             p['category'], quantity, p['normalized_unit'], p['pack_count'],
                             Jsonb(attributes), p['needs_review'], p['review_reasons'], p['rules_version']))
                    stats['new_products'] += 1

                for source in source_prices:
                    pid = int(source['product_id'])
                    if pid not in products:
                        raise RuntimeError(f'История {source["id"]} ссылается на отсутствующий product_id={pid}')
                    key = price_key(source)
                    if key in seen_prices:
                        stats['duplicate_prices_skipped'] += 1
                        continue
                    history_id = int(source['id'])
                    if history_id in existing_price_ids:
                        stats['price_id_conflicts'] += 1
                        raise RuntimeError(f'Конфликт history_id={history_id}: запись уже есть с другими значениями. Ничего не сохранено.')
                    seen_prices.add(key)
                    existing_price_ids.add(history_id)
                    p = products[pid]
                    price = safe_number(source['price_som'])
                    old = safe_number(source['old_price_som'])
                    quantity = safe_number(p['total_quantity'])
                    reasons = []
                    if price is None or price <= 0:
                        reasons.append('invalid_price')
                    elif price > MAX_PRICE:
                        reasons.append('price_above_review_threshold')
                    if source['status'] != 'public_page':
                        reasons.append('unverified_source_status')
                    if quantity is None or quantity <= 0:
                        reasons.append('unknown_package')
                    if p['normalized_unit'] not in {'kg', 'l', 'pcs', 'кг', 'л', 'шт'}:
                        reasons.append('unknown_unit')
                    flags = set((p['review_reasons'] or '').split(';'))
                    if flags & {'multiple_measures', 'ambiguous_pack_count', 'source_quantity_conflict'}:
                        reasons.append('ambiguous_package')
                    eligible = not reasons
                    per_unit = (price / quantity).quantize(Decimal('0.01')) if eligible else None
                    if args.apply:
                        cur.execute('''INSERT INTO grocery_clean.price_history_clean
                            (history_id,product_id,price_som,old_price_som,collected_at,source_status,
                             price_per_unit,eligible_price,review_reasons)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            ON CONFLICT (history_id) DO NOTHING''',
                            (history_id, pid, price, old, source['collected_at'], source['status'],
                             per_unit, eligible, ';'.join(reasons)))
                    stats['new_price_observations'] += 1
                    stats['eligible_new_prices'] += int(eligible)
            if args.apply:
                conn.commit()
            else:
                conn.rollback()
        except Exception:
            conn.rollback()
            raise

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print('ГОТОВО: изменения записаны.' if args.apply else 'ПРОВЕРКА: база не изменена. Для записи добавьте --apply.')


if __name__ == '__main__':
    main()
