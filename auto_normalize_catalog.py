# Консервативная нормализация категорий и брендов для обучения модели классификации.

from pathlib import Path
import re
import unicodedata
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'category_training_prepared'
OUTPUT = ROOT / 'category_normalized'
OUTPUT.mkdir(exist_ok=True)

def norm(s):
    s = unicodedata.normalize('NFKC', str(s or '')).casefold().replace('ё','е')
    s = re.sub(r'[\u00ad\u200b-\u200d]', '', s)
    return re.sub(r'\s+', ' ', s).strip()

def present(text, alias):
    # Не ищем короткие аббревиатуры внутри других слов.
    return re.search(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', text) is not None

products = pd.read_csv(SOURCE / 'products_for_review.csv', dtype=str, keep_default_na=False, encoding='utf-8-sig')
aliases = pd.read_csv(SOURCE / 'brand_aliases.csv', dtype=str, keep_default_na=False, encoding='utf-8-sig')
categories = set(pd.read_csv(SOURCE / 'categories_reference.csv', dtype=str, keep_default_na=False, encoding='utf-8-sig')['category'])

brand_entries = [(norm(row.alias_normalized), row.brand_canonical) for row in aliases.itertuples(index=False) if norm(row.alias_normalized)]
brand_entries.sort(key=lambda x: len(x[0]), reverse=True)

out = []
for row in products.to_dict('records'):
    name = norm(row['original_name'])
    hits = [(alias, canonical) for alias, canonical in brand_entries if present(name, alias)]
    # Короткое название отбрасываем, если оно содержится в длинном совпавшем названии.
    maximal = [(alias, canonical) for alias, canonical in hits if not any(alias != other and present(other, alias) for other, _ in hits)]
    brands = sorted({canonical for _, canonical in maximal})
    if len(brands) == 1:
        new_brand, brand_status = brands[0], 'auto_dictionary'
    elif len(brands) > 1:
        new_brand, brand_status = '', 'manual_multiple_brands'
    else:
        new_brand, brand_status = '', 'manual_brand_not_found'

    old_category = row['old_category']
    if old_category == 'Непродовольственные: подгузники' and 'Памперсы и трусики' in categories:
        new_category, category_status = 'Памперсы и трусики', 'auto_category_alias'
    elif old_category in categories:
        new_category, category_status = old_category, 'existing_unverified'
    else:
        new_category, category_status = '', 'manual_category_review'

    row.update(normalized_category=new_category, normalized_brand=new_brand,
               category_decision=category_status, brand_decision=brand_status,
               human_category='', human_brand='', review_status='pending')
    out.append(row)

result = pd.DataFrame(out)
result.to_csv(OUTPUT / 'products_normalized.csv', index=False, encoding='utf-8-sig')
review = result[(result.category_decision == 'manual_category_review') | result.brand_decision.str.startswith('manual_')]
review.to_csv(OUTPUT / 'manual_review.csv', index=False, encoding='utf-8-sig')
print('Всего товаров:', len(result))
print('Категории:', result.category_decision.value_counts().to_dict())
print('Бренды:', result.brand_decision.value_counts().to_dict())
print('На ручную проверку (категория или бренд):', len(review))
print('Результаты:', OUTPUT)
print('PostgreSQL не изменён. Метки existing_unverified требуют проверки перед обучением.')
