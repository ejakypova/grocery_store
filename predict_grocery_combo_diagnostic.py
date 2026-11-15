
import argparse
import csv
import getpass
import os
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from dotenv import load_dotenv

ALIASES = {'вм': 'Веселый молочник', 'двд': 'Домик в деревне'}


def normalize(value):
    value = unicodedata.normalize('NFKC', str(value or '')).casefold().replace('ё', 'е')
    return re.sub(r'\s+', ' ', re.sub(r'[^\w]+', ' ', value, flags=re.UNICODE)).strip()


def predict_batch(model, names):
    if model is None:
        return [(None, None)] * len(names)
    # Совместимо с нормализацией train_category_optimized.py для обычных названий.
    texts = [normalize(name) for name in names]
    probabilities = model.predict_proba(texts)
    classes = getattr(model, 'classes_', None)
    if classes is None and hasattr(model, 'named_steps'):
        classes = model.steps[-1][1].classes_
    if classes is None:
        raise ValueError('Модель не предоставляет классы для predict_proba')
    indices = np.argmax(probabilities, axis=1)
    return [(str(classes[i]), float(probabilities[j, i])) for j, i in enumerate(indices)]




def diagnose_model(model, names, out_dir, batch_size=300):
    """Проверяет распределение уверенности и сохраняет три наиболее вероятные категории."""
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = model.named_steps['model'].classes_ if hasattr(model, 'named_steps') else model.classes_
    path = out_dir / 'prediction_diagnostics.csv'
    confidence_values = []
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Название', 'Топ-1', 'Вероятность 1', 'Топ-2', 'Вероятность 2', 'Топ-3', 'Вероятность 3'])
        for start in range(0, len(names), batch_size):
            chunk = names[start:start + batch_size]
            probs = model.predict_proba([normalize(n) for n in chunk])
            for name, row in zip(chunk, probs):
                best = np.argsort(row)[::-1][:3]
                top = [(str(classes[i]), float(row[i])) for i in best]
                confidence_values.append(top[0][1])
                flattened = [v for label, conf in top for v in (label, conf)]
                writer.writerow([name, *flattened, *([''] * (6 - len(flattened)))])
    vals = np.asarray(confidence_values)
    print(f'Диагностика: средняя уверенность {vals.mean():.1%}; медиана {np.median(vals):.1%}')
    print(f'Ниже 60%: {(vals < 0.6).sum()} из {len(vals)}')
    print(f'Три наиболее вероятные категории: {path.resolve()}')

def load_brands(reference):
    aliases = {}
    if reference.is_file():
        wb = load_workbook(reference, read_only=True, data_only=True)
        try:
            for sheet in wb.worksheets:
                for row in sheet.iter_rows(values_only=True):
                    if len(row) < 2 or normalize(row[0]) != 'brand' or not row[1]:
                        continue
                    variants = [x.strip() for x in str(row[1]).split('/') if x.strip()]
                    if variants:
                        canonical = max(variants, key=lambda x: len(normalize(x)))
                        for variant in variants:
                            aliases[normalize(variant)] = canonical
        finally:
            wb.close()
    else:
        print(f'Справочник брендов не найден: {reference}; используется ML и встроенные сокращения')
    for alias, canonical in ALIASES.items():
        aliases.setdefault(normalize(alias), canonical)
    return list(aliases.items())


def dictionary_brand(name, entries):
    words = normalize(name).split()
    hits = []
    for alias, canonical in entries:
        tokens = alias.split()
        if tokens and any(words[i:i + len(tokens)] == tokens for i in range(len(words) - len(tokens) + 1)):
            hits.append((len(tokens), len(alias), canonical))
    if not hits:
        return None, 'не найден'
    hits.sort(reverse=True)
    best = hits[0]
    if len({hit[2] for hit in hits if hit[:2] == best[:2]}) > 1:
        return None, 'неоднозначное совпадение'
    return best[2], 'справочник'


def read_uncategorized(run_id):
    import psycopg
    load_dotenv()
    password = os.getenv('DB_PASSWORD') or getpass.getpass('Пароль PostgreSQL: ')
    with psycopg.connect(
        host=os.getenv('DB_HOST', 'localhost'), port=int(os.getenv('DB_PORT', '5432')),
        dbname=os.getenv('DB_NAME', 'grocery_store'),
        user=os.getenv('DB_USER', 'postgres'), password=password,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute('''
                SELECT source_id, data->>'original_name', data->>'category_source'
                FROM grocery_clean.records
                WHERE run_id = %s AND kind = 'product'
                  AND NULLIF(BTRIM(data->>'category'), '') IS NULL
                ORDER BY source_id
            ''', (run_id,))
            rows = cur.fetchall()
    return [{'product_id': str(pid), 'name': name, 'category_source': source or ''}
            for pid, name, source in rows if name and normalize(name)]


def quality(confidence, medium, high):
    if confidence is None:
        return 'Нет модели'
    if confidence >= high:
        return 'Высокая — проверить'
    if confidence >= medium:
        return 'Средняя — проверить'
    return 'Низкая — проверить'


def export_excel(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = 'Предсказания'
    headers = ['product_id', 'Название товара', 'Категория ML', 'Уверенность категории',
               'Статус категории', 'Бренд', 'Источник бренда', 'Уверенность бренда',
               'Категория проверена вручную', 'Комментарий']
    ws.append(headers)
    for row in rows:
        ws.append([row['product_id'], row['name'], row['category'], row['category_conf'],
                   row['category_status'], row['brand'], row['brand_source'], row['brand_conf'], '', ''])
    ws.freeze_panes = 'C2'
    ws.auto_filter.ref = ws.dimensions
    for cell in ws[1]:
        cell.font = Font(bold=True, color='FFFFFF')
        cell.fill = PatternFill('solid', fgColor='1F4E78')
        cell.alignment = Alignment(wrap_text=True)
    widths = [16, 55, 35, 24, 26, 25, 24, 23, 35, 35]
    for i, width in enumerate(widths, start=1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = width
    for row in ws.iter_rows(min_row=2):
        for i in (4, 8):
            row[i-1].number_format = '0.0%'
    info = wb.create_sheet('Инструкция')
    info.append(['Поле', 'Описание'])
    info.append(['Категория ML', 'Прогноз модели, не подтверждённая категория'])
    info.append(['Уверенность', 'Оценка predict_proba; не калибрована и не гарантирует точность'])
    info.append(['Категория проверена вручную', 'Впишите правильную категорию после проверки'])
    info.append(['База PostgreSQL', 'Не изменяется при запуске скрипта'])
    wb.save(path)
    print(f'Excel сохранён: {path.resolve()}')


def main():
    parser = argparse.ArgumentParser(description='ML категории и бренды; массовый экспорт из PostgreSQL')
    parser.add_argument('names', nargs='*', help='Названия товаров для разового предсказания')
    parser.add_argument('--source', choices=['manual', 'db'], default='manual')
    parser.add_argument('--run-id', type=int, default=2)
    parser.add_argument('--export-excel', action='store_true')
    parser.add_argument('--output', default='category_predictions/uncategorized_predictions.xlsx')
    parser.add_argument('--models', default='grocery_models')
    parser.add_argument('--category-model', default='category_optimization/category_model_optimized.joblib')
    parser.add_argument('--reference', default='Справочник бренд и категорий.xlsx')
    parser.add_argument('--threshold', type=float, default=0.60)
    parser.add_argument('--high-threshold', type=float, default=0.80)
    parser.add_argument('--diagnose', action='store_true', help='Сохранить три вероятные категории и статистику уверенности')
    args = parser.parse_args()
    if not (0 <= args.threshold < args.high_threshold <= 1):
        parser.error('Укажите пороги: 0 <= --threshold < --high-threshold <= 1')
    category_file = Path(args.category_model)
    if not category_file.is_file():
        parser.error(f'Не найдена модель категорий: {category_file}')
    category_model = joblib.load(category_file)
    brand_file = Path(args.models) / 'brand_label_model.joblib'
    brand_model = joblib.load(brand_file) if brand_file.is_file() else None
    entries = load_brands(Path(args.reference))
    if args.source == 'db':
        items = read_uncategorized(args.run_id)
        print(f'Товаров без категории с названием: {len(items)} (run_id={args.run_id})')
    else:
        names = args.names or [input('Название товара: ')]
        items = [{'product_id': '', 'name': name} for name in names if normalize(name)]
    if not items:
        print('Нет товаров для предсказания')
        return
    category_predictions = predict_batch(category_model, [item['name'] for item in items])
    if args.diagnose:
        diagnose_model(category_model, [item['name'] for item in items], Path(args.output).parent)
    dictionary_results = [dictionary_brand(item['name'], entries) for item in items]
    missing_indices = [i for i, (brand, _) in enumerate(dictionary_results) if not brand]
    brand_predictions = predict_batch(brand_model, [items[i]['name'] for i in missing_indices])
    ml_brands = dict(zip(missing_indices, brand_predictions))
    results = []
    for i, item in enumerate(items):
        category, category_conf = category_predictions[i]
        brand, brand_source = dictionary_results[i]
        brand_conf = None
        if brand is None:
            brand, brand_conf = ml_brands.get(i, (None, None))
            brand_source = 'ML — проверить' if brand else 'Не определён'
        results.append({
            **item, 'category': category, 'category_conf': category_conf,
            'category_status': quality(category_conf, args.threshold, args.high_threshold),
            'brand': brand or '', 'brand_source': brand_source, 'brand_conf': brand_conf,
        })
    if args.source == 'manual':
        for row in results:
            print(f"\nТовар: {row['name']}\nКатегория: {row['category']} "
                  f"({row['category_conf']:.1%}; {row['category_status']})\n"
                  f"Бренд: {row['brand'] or 'не определён'} ({row['brand_source']})")
    if args.export_excel or args.source == 'db':
        export_excel(results, Path(args.output))
    print('PostgreSQL не изменён.')


if __name__ == '__main__':
    main()
