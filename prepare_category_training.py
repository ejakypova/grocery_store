"""Подготовка справочников и таблицы ручной проверки категорий/брендов.

Запуск: python prepare_category_training.py
Зависимости: pip install pandas openpyxl
Не изменяет PostgreSQL и не обучает модель.
"""
from pathlib import Path
import re
import unicodedata
import pandas as pd

ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / 'Справочник бренд и категорий.xlsx'
PRODUCTS = ROOT / 'clean_report' / 'products_clean.csv'
OUTPUT = ROOT / 'category_training_prepared'


def normalize(value):
    if pd.isna(value):
        return ''
    text = unicodedata.normalize('NFKC', str(value)).casefold().replace('ё', 'е')
    text = re.sub(r'[\u00ad\u200b-\u200d]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def main():
    if not CATALOG.exists():
        raise FileNotFoundError(f'Нет справочника: {CATALOG}')
    if not PRODUCTS.exists():
        raise FileNotFoundError(f'Нет файла товаров: {PRODUCTS}')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference = pd.read_excel(CATALOG, dtype=str)
    required = {'тип', 'значение_из_excel'}
    if not required.issubset(reference.columns):
        raise ValueError(f'В Excel должны быть столбцы: {sorted(required)}')
    reference['тип'] = reference['тип'].map(normalize)
    values = reference['значение_из_excel'].dropna().astype(str).str.strip()
    reference['значение_из_excel'] = reference['значение_из_excel'].astype('string').str.strip()
    categories = sorted(set(reference.loc[reference['тип'].eq('category'), 'значение_из_excel'].dropna()))
    if not categories:
        raise ValueError('В справочнике не найдены строки с типом category')
    pd.DataFrame({'category': categories}).to_csv(OUTPUT / 'categories_reference.csv', index=False, encoding='utf-8-sig')

    aliases = []
    for raw in reference.loc[reference['тип'].eq('brand'), 'значение_из_excel'].dropna():
        parts = [x.strip() for x in str(raw).split('/') if x.strip()]
        if not parts:
            continue
        # Для известных сокращений явно выбираем полное каноническое имя.
        canonical = {'вм/веселый молочник': 'Веселый молочник',
                     'домик в деревне/двд': 'Домик в деревне'}.get(normalize(raw), parts[0])
        for part in parts:
            aliases.append({'alias_normalized': normalize(part), 'brand_canonical': canonical,
                            'source_excel': raw})
    alias_df = pd.DataFrame(aliases).drop_duplicates()
    collisions = alias_df.groupby('alias_normalized')['brand_canonical'].nunique()
    ambiguous = set(collisions[collisions > 1].index)
    if ambiguous:
        print('Неоднозначные псевдонимы (пропускаем):', sorted(ambiguous))
        alias_df = alias_df[~alias_df.alias_normalized.isin(ambiguous)]
    alias_df.to_csv(OUTPUT / 'brand_aliases.csv', index=False, encoding='utf-8-sig')

    products = pd.read_csv(PRODUCTS, dtype={'product_id': 'Int64'}, low_memory=False)
    expected = {'product_id', 'original_name', 'category', 'brand', 'store_name'}
    if not expected.issubset(products.columns):
        raise ValueError(f'В products_clean.csv нет столбцов: {sorted(expected - set(products.columns))}')

    category_by_key = {normalize(c): c for c in categories}
    # Строгие синонимы категорий: НЕ назначаем подробные категории по общему названию.
    category_synonyms = {
        'непродовольственные: подгузники': 'Памперсы и трусики',
        'безалкокольные напитки': 'Безалкокольные напитки',
    }
    for old, target in category_synonyms.items():
        if normalize(target) in category_by_key:
            category_by_key[normalize(old)] = category_by_key[normalize(target)]

    # Длинные названия проверяем первыми; сокращения — только по границам слов.
    patterns = []
    for row in alias_df.itertuples(index=False):
        alias = row.alias_normalized
        if alias:
            pattern = re.compile(r'(?<!\w)' + re.escape(alias).replace(r'\ ', r'\s+') + r'(?!\w)', re.I)
            patterns.append((len(alias), pattern, row.brand_canonical))
    patterns.sort(key=lambda item: -item[0])

    records = []
    for item in products.itertuples(index=False):
        name = str(item.original_name) if pd.notna(item.original_name) else ''
        old_category = str(item.category).strip() if pd.notna(item.category) else ''
        old_brand = str(item.brand).strip() if pd.notna(item.brand) else ''
        proposed_category = category_by_key.get(normalize(old_category), '')
        if not old_category:
            category_status = 'missing_category'
        elif not proposed_category:
            category_status = 'not_in_reference_needs_review'
        elif normalize(old_category) == normalize(proposed_category):
            category_status = 'existing_exact_unverified'
        else:
            category_status = 'category_alias_needs_review'
        normalized_name = normalize(name)
        matched_brands = list(dict.fromkeys(canonical for _, pattern, canonical in patterns if pattern.search(normalized_name)))
        proposed_brand = matched_brands[0] if len(matched_brands) == 1 else ''
        brand_status = ('dictionary_match' if len(matched_brands) == 1
                        else 'ambiguous_multiple_brands' if len(matched_brands) > 1
                        else 'not_detected')
        records.append({
            'product_id': item.product_id, 'store_name': item.store_name,
            'original_name': name, 'old_category': old_category,
            'proposed_category': proposed_category, 'category_status': category_status,
            'old_brand': old_brand, 'proposed_brand': proposed_brand,
            'brand_status': brand_status, 'human_category': '', 'human_brand': '',
            'review_status': 'pending',
        })
    review = pd.DataFrame(records)
    review.to_csv(OUTPUT / 'products_for_review.csv', index=False, encoding='utf-8-sig')
    priority = review[(review.category_status != 'existing_exact_unverified') |
                      (review.brand_status == 'ambiguous_multiple_brands')]
    priority.to_csv(OUTPUT / 'priority_review.csv', index=False, encoding='utf-8-sig')
    print(f'Готово. Товаров: {len(review)}; категорий: {len(categories)}; '
          f'псевдонимов брендов: {len(alias_df)}; приоритетная проверка: {len(priority)}')
    print(f'Результаты: {OUTPUT}')
    print('ВАЖНО: предложенные значения не подтверждены вручную; PostgreSQL не изменён.')


if __name__ == '__main__':
    main()
