"""ML-поиск и предварительное сравнение стоимости.

Читает PostgreSQL, не изменяет данные. Сначала ищет по ВСЕЙ базе, затем применяет ML v5.

Запуск (с альтернативными вариантами товаров):
 python basket_price_engine_v5.py --basket "Котлеты Атахан говядина 1кг,молоко Белая Река" --match-mode alternatives --export basket_v5.csv
"""
import argparse
import os
import re
from difflib import SequenceMatcher
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
OUTPUT_COLS = ['requested_item','store','product_id','display_name','brand','category',
 'category_source','total_quantity','normalized_unit','price_som','price_per_unit',
 'unit_price_label','collected_at','source_status','anchor_product_id','anchor_name',
 'ml_label','safe_label','same_probability','needs_review','query_category_confidence',
 'query_overlap','eligible_for_basket','basket_exclusion_reason',
 'direct_found','match_source','query_match_score']


def clean(x):
    if x is None or pd.isna(x):
        return ''
    s = re.sub(r'\s+', ' ', str(x).lower().replace('ё','е')).strip()
    return '' if s in INVALID else s


def connection():
    cfg = dict(host=os.getenv('DB_HOST','localhost'),
               port=int(os.getenv('DB_PORT','5432')),
               dbname=os.getenv('DB_NAME','grocery_store'),
               user=os.getenv('DB_USER','postgres'))
    if os.getenv('DB_PASSWORD'):
        cfg['password'] = os.getenv('DB_PASSWORD')
    return psycopg.connect(**cfg)


def load_prices(mode):
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(SQL, {'mode': mode})
            df = pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])
    if df.empty:
        return df
    for c in ('price_som','price_per_unit','total_quantity'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['display_name'] = df.original_name.fillna('').replace('',pd.NA).fillna(df.name_clean).fillna('')
    df['name_key'] = df.display_name.map(clean)
    return df


def models():
    category = joblib.load(BASE / 'category_model_optimized.joblib')
    matching = joblib.load(BASE / 'product_matching_model.joblib')
    if not {'features','model'}.issubset(matching.named_steps):
        raise ValueError('Неожиданный формат модели сопоставления')
    if 'same' not in matching.classes_:
        raise ValueError('Модель сопоставления не содержит класса same')
    return category, matching


def search_tokens(value):
    """Ключевые слова для поиска, допускающие разные пробелы и знаки препинания."""
    return re.findall(r'[а-яa-z0-9]+', clean(value))


def word_matches(word, candidate):
    if word == candidate:
        return True
    # небольшие различия окончаний: «котлеты» / «котлет»
    if len(word) >= 5 and len(candidate) >= 5 and word[:4] == candidate[:4]:
        return True
    return False


def name_score(query, name):
    """Оценка покрытия слов запроса, без требования совпадения ML-категории."""
    qt, nt = search_tokens(query), search_tokens(name)
    if not qt or not nt:
        return 0.0
    matched = sum(any(word_matches(w, t) for t in nt) for w in qt)
    coverage = matched / len(qt)
    similarity = SequenceMatcher(None, clean(query), clean(name)).ratio()
    return 0.85 * coverage + 0.15 * similarity


def find(df, query, category_model, matching_model, limit=20,
         include_uncategorized=False, category_threshold=.45, same_threshold=.8):
    """Ищем по ВСЕМ загруженным товарам; ML уточняет, но не блокирует цену."""
    if df.empty or not clean(query):
        return pd.DataFrame()

    q = clean(query)
    # 1. Поиск по всей базе: ни категория, ни product_pair_candidates не ограничивают его.
    pool = df.copy()
    pool['query_overlap'] = pool.name_key.map(
        lambda name: sum(any(word_matches(w, t) for t in search_tokens(name))
                         for w in search_tokens(q)) / max(1, len(search_tokens(q))))
    pool = pool.loc[pool.query_overlap.gt(0)].copy()
    if pool.empty:
        return pool
    pool['query_match_score'] = pool.name_key.map(lambda name: name_score(q, name))
    pool = pool.sort_values(['query_overlap', 'query_match_score', 'price_som'],
                            ascending=[False, False, True]).copy()

    # 2. Категория — информация, а НЕ фильтр поиска.
    query_category, query_conf = '', 0.0
    if category_model is not None:
        try:
            query_category = str(category_model.predict([q])[0])
            query_conf = float(np.max(category_model.predict_proba([q])))
        except Exception as exc:
            print(f'Категория не определена для «{query}»: {exc}')
    pool['category_source'] = np.where(pool.category.map(clean).eq(''), 'unknown', 'database')
    pool['category'] = pool.category.fillna('')
    if category_model is not None:
        unknown = pool.category.map(clean).eq('')
        if unknown.any():
            try:
                names = pool.loc[unknown, 'name_key'].tolist()
                pred = category_model.predict(names)
                conf = category_model.predict_proba(names).max(axis=1)
                accepted = conf >= category_threshold
                idx = pool.index[unknown]
                pool.loc[idx[accepted], 'category'] = pred[accepted]
                pool.loc[idx[accepted], 'category_source'] = 'ml_predicted'
            except Exception as exc:
                print(f'Не удалось заполнить категории для «{query}»: {exc}')

    anchor = pool.iloc[0]
    pool['anchor_product_id'] = int(anchor.product_id)
    pool['anchor_name'] = anchor.display_name
    pool['query_category'] = query_category
    pool['query_category_confidence'] = query_conf
    pool['unit_price_label'] = pool.normalized_unit.map(
        lambda u: 'сом/кг' if u == 'кг' else ('сом/л' if u == 'л' else 'сом/ед.'))
    pool['ml_label'] = 'not_evaluated'
    pool['safe_label'] = 'found'
    pool['same_probability'] = np.nan
    pool['needs_review'] = False
    pool['match_source'] = 'database_search'

    # Прямое совпадение названия: всегда показывать цену, даже если ML не согласен.
    # Порог поиска мягче для коротких запросов; подтверждение покупки строже.
    pool['direct_found'] = pool.query_overlap.ge(0.8) & pool.query_match_score.ge(0.73)
    exact = pool.name_key.eq(q)
    pool.loc[exact, 'direct_found'] = True

    # 3. ML для оценки кандидатов относительно лучшего найденного товара.
    # Сравнение НЕ зависит от ранее созданных product_pair_candidates.
    if matching_model is not None:
        pairs = pd.DataFrame([{
            'name_a': anchor.display_name, 'name_b': r.display_name,
            'brand_a': anchor.brand, 'brand_b': r.brand,
            'category_a': anchor.category, 'category_b': r.category,
            'quantity_a': anchor.total_quantity, 'quantity_b': r.total_quantity,
            'unit_a': anchor.normalized_unit, 'unit_b': r.normalized_unit
        } for r in pool.itertuples(index=False)], columns=COLS)
        try:
            probs = matching_model.predict_proba(pairs)
            raw = matching_model.predict(pairs)
            classes = list(matching_model.classes_)
            same_probs = probs[:, classes.index('same')]
            conflict = [bool(same_conflict(row)) for _, row in pairs.iterrows()]
            review = [str(label) == 'same' and (float(prob) < same_threshold or bad)
                      for label, prob, bad in zip(raw, same_probs, conflict)]
            pool['ml_label'] = raw
            pool['same_probability'] = same_probs
            pool['needs_review'] = review
            pool['safe_label'] = np.where(review, 'review', raw)
            pool['match_source'] = 'ml'
        except Exception as exc:
            print(f'ML-сопоставление для «{query}» недоступно: {exc}. Показываю найденные цены.')

    # Если ML отклонил прямое найденное название, оно всё равно видно как found.
    # Отдельные противоречивые пары не считаем автоматически одинаковыми.
    pool.loc[pool.direct_found & pool.safe_label.isin(['different', 'review', 'not_evaluated']),
             'match_source'] = 'database_search'
    return pool.sort_values(['query_overlap', 'query_match_score', 'price_som'],
                            ascending=[False, False, True])


def prepare_candidates(output, match_mode):
    """Показываем найденные цены независимо от ML; корзина — осторожнее."""
    if output.empty:
        output = output.copy()
        output['eligible_for_basket'] = pd.Series(dtype=bool)
        output['basket_exclusion_reason'] = pd.Series(dtype=str)
        return output
    output = output.copy()
    allowed = {'same'} if match_mode == 'same' else {'same', 'alternative'}

    def reason(r):
        if not np.isfinite(r.price_som) or r.price_som <= 0:
            return 'нет корректной цены'
        # Полное точное название — безопасный fallback на цену базы.
        if clean(r.display_name) == clean(r.requested_item):
            return ''
        # Когда ML не нашёл пару, прямой текстовый матч всё равно может
        # попасть в корзину; для однословных запросов не выбираем случайный товар.
        informative = len(search_tokens(r.requested_item)) >= 2
        if r.direct_found and informative and r.safe_label in {'different', 'review', 'found', 'not_evaluated'}:
            if r.safe_label == 'review':
                return 'ML требует проверки; цена показана, но не включена в итог'
            if r.safe_label == 'different':
                return 'ML считает товары разными; цена показана, но не включена в итог'
            return ''
        if r.safe_label == 'review':
            return 'требует проверки сопоставления'
        if r.safe_label not in allowed:
            return 'не подтверждено сопоставление (цена показана)'
        # Слишком широкие запросы вроде «масло» не должны выдавать случайно
        # самую дешёвую упаковку за выигрышную корзину.
        if not informative:
            return 'уточните название товара для расчёта корзины'
        if r.query_overlap < 0.5:
            return 'слабое совпадение с запросом'
        return ''

    output['basket_exclusion_reason'] = [reason(r) for r in output.itertuples(index=False)]
    output['eligible_for_basket'] = output.basket_exclusion_reason.eq('')
    return output


def summarize(output, requested_items, all_stores):
    eligible = output.loc[output.eligible_for_basket].copy() if not output.empty else output.copy()
    # Для повторяющихся запросов сохраняем отдельные позиции корзины по порядку.
    # item_index заполняется вызывающим кодом, а не по названию товара.
    rows = []
    for store in all_stores:
        selected = []
        for idx in range(len(requested_items)):
            group = eligible.loc[(eligible.item_index == idx) & (eligible.store == store)]
            if not group.empty:
                selected.append(group.sort_values('price_som').iloc[0])
        rows.append({'store':store, 'found_items':len(selected),
                     'requested_items':len(requested_items),
                     'missing_items':', '.join(requested_items[i] for i in range(len(requested_items))
                         if not any(int(r.item_index)==i for r in selected)),
                     'total_som':round(sum(float(r.price_som) for r in selected),2)
                         if len(selected)==len(requested_items) else np.nan,
                     'complete':len(selected)==len(requested_items)})
    store_summary = pd.DataFrame(rows, columns=['store','found_items','requested_items',
                                                  'missing_items','total_som','complete'])
    mixed = []
    for idx,item in enumerate(requested_items):
        group = eligible.loc[eligible.item_index==idx]
        if group.empty:
            mixed.append({'requested_item':item,'store':'НЕ НАЙДЕНО',
                          'product_id':pd.NA,'display_name':'', 'price_som':np.nan,
                          'collected_at':pd.NaT})
        else:
            r = group.sort_values('price_som').iloc[0]
            mixed.append({'requested_item':item,'store':r.store,
                          'product_id':r.product_id,'display_name':r.display_name,
                          'price_som':r.price_som,'collected_at':r.collected_at})
    mixed_df = pd.DataFrame(mixed)
    mixed_complete = bool(len(mixed_df)==len(requested_items) and mixed_df.price_som.notna().all())
    mixed_total = float(mixed_df.price_som.sum()) if mixed_complete else None
    return store_summary, mixed_df, mixed_total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--search')
    g.add_argument('--basket')
    ap.add_argument('--mode',choices=['public','demo','all'],default='all')
    ap.add_argument('--match-mode',choices=['same','alternatives'],default='same',
                    help='same: только same; alternatives: same и alternative')
    ap.add_argument('--include-uncategorized',action='store_true')
    ap.add_argument('--limit',type=int,default=100)
    ap.add_argument('--category-threshold',type=float,default=.45)
    ap.add_argument('--same-threshold',type=float,default=.8)
    ap.add_argument('--export',help='CSV кандидатов; рядом сохраняются сводки')
    a = ap.parse_args()
    if a.limit<1 or not 0<=a.category_threshold<=1 or not 0<=a.same_threshold<=1:
        ap.error('Проверьте limit и пороги [0,1]')
    df = load_prices(a.mode)
    category_model, matching_model = models()
    print(f'Ценовых предложений: {len(df)}; источник: {a.mode}; режим: {a.match_mode}')
    items = [a.search] if a.search else [x.strip() for x in a.basket.split(',') if x.strip()]
    if not items:
        ap.error('Корзина пустая')
    results = []
    for idx,item in enumerate(items):
        r = find(df,item,category_model,matching_model,a.limit,a.include_uncategorized,
                 a.category_threshold,a.same_threshold)
        if r.empty:
            print('В базе не найдено название с общими словами:',item)
            continue
        r.insert(0,'requested_item',item)
        r['item_index'] = idx
        results.append(r)
    output = pd.concat(results,ignore_index=True) if results else pd.DataFrame(columns=OUTPUT_COLS+['item_index'])
    output = prepare_candidates(output,a.match_mode)
    print('\nКандидаты (предварительные ML-метки):')
    view_cols = ['requested_item','store','display_name','price_som','safe_label','direct_found','eligible_for_basket']
    print(output[view_cols].to_string(index=False,max_colwidth=42) if not output.empty else 'Совпадений нет')
    stores = sorted(df.store.dropna().unique().tolist()) if not df.empty else []
    store_summary,mixed_df,mixed_total = summarize(output,items,stores)
    print('\nПРЕДВАРИТЕЛЬНО: покупка всей корзины в одном магазине (1 упаковка на позицию):')
    print(store_summary.to_string(index=False) if not store_summary.empty else 'Нет магазинов')
    complete_stores = store_summary.loc[store_summary.complete].sort_values('total_som')
    if complete_stores.empty:
        print('Нет магазина с полной корзиной по выбранным условиям.')
    else:
        best = complete_stores.iloc[0]
        print(f'Минимальная полная корзина в одном магазине: {best.store}, {best.total_som:.2f} сом')
    print('\nПРЕДВАРИТЕЛЬНО: смешанная корзина (по одному предложению на позицию):')
    print(mixed_df.to_string(index=False,max_colwidth=42))
    if mixed_total is None:
        print('Итог смешанной корзины не рассчитывается: есть ненайденные позиции.')
    else:
        print(f'Итого смешанная корзина: {mixed_total:.2f} сом (без доставки)')
        if not complete_stores.empty:
            savings = float(complete_stores.iloc[0].total_som) - mixed_total
            print(f'Разница с лучшим полным магазином: {savings:.2f} сом')
    print('\nВНИМАНИЕ: ML-метки не подтверждают идентичность. Широкие запросы (например, «масло»)')
    print('могут объединять разные виды товаров. Цена относится к 1 упаковке, а не к 1 кг/л.')
    print('Все найденные товары с ценой видны в кандидатах, даже если ML не подтвердил пару.')
    print('Проверьте фасовку, вид, бренд, свежесть цен и исходные статусы перед выводом о выгоде.')
    if a.export:
        target = Path(a.export)
        target.parent.mkdir(parents=True,exist_ok=True)
        output[[c for c in OUTPUT_COLS if c in output.columns]].to_csv(target,index=False,encoding='utf-8-sig')
        store_file = target.with_name(target.stem+'_stores.csv')
        mixed_file = target.with_name(target.stem+'_mixed.csv')
        store_summary.to_csv(store_file,index=False,encoding='utf-8-sig')
        mixed_df.to_csv(mixed_file,index=False,encoding='utf-8-sig')
        print('Сохранены:',target,store_file,mixed_file)

if __name__ == '__main__':
    main()
