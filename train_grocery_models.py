"""Обучение моделей категории и бренда для Grocery Store.
Запуск: python train_grocery_models.py --csv grocery_merged_for_training.csv --review "Сверка категорий_заполнено.xlsx"
По умолчанию обучается только на ручных метках. --include-unverified добавляет исходные авто-метки (с риском ошибок).
"""
import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


def norm(x):
    x = unicodedata.normalize('NFKC', str(x or '')).lower().replace('ё', 'е')
    x = re.sub(r'[^\w\s]', ' ', x)
    return re.sub(r'\s+', ' ', x).strip()


def rows_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def read_review(path):
    # Excel: read_only preserves memory for large sheets
    from openpyxl import load_workbook
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    it = ws.iter_rows(values_only=True)
    headers = [str(x).strip() if x is not None else '' for x in next(it)]
    out = {}
    for values in it:
        r = dict(zip(headers, values))
        if r.get('product_id') is not None:
            out[str(r['product_id']).strip()] = {
                'human_category': str(r.get('human_category') or '').strip(),
                'human_brand': str(r.get('human_brand') or '').strip(),
            }
    return out


def train_task(data, field, folder, threshold=0.60):
    records = [(norm(r['original_name']), r[field]) for r in data if r.get(field) and norm(r.get('original_name'))]
    # Never let the same normalized product name occur in train and test.
    by_name = {}
    conflicts = set()
    for name, label in records:
        if name in by_name and by_name[name] != label:
            conflicts.add(name)
        else:
            by_name[name] = label
    clean = [(n, lab) for n, lab in by_name.items() if n not in conflicts]
    counts = Counter(lab for _, lab in clean)
    print(f'\n{field}: {len(clean)} unique names, {len(counts)} classes, {len(conflicts)} conflicting names excluded')
    if len(counts) < 2:
        print(f'{field}: insufficient distinct labels; skipping model')
        return
    # Keep rare labels in training; evaluate on classes with at least 3 examples.
    eligible = [i for i, (_, label) in enumerate(clean) if counts[label] >= 3]
    rare = [i for i, (_, label) in enumerate(clean) if counts[label] < 3]
    test_idx = []
    train_idx = list(range(len(clean)))
    if len(eligible) >= 12 and len({clean[i][1] for i in eligible}) >= 2:
        x = [clean[i][0] for i in eligible]
        y = [clean[i][1] for i in eligible]
        from sklearn.model_selection import train_test_split
        a, b = train_test_split(list(range(len(eligible))), test_size=0.2, random_state=42, stratify=y if min(Counter(y).values()) >= 5 else None)
        test_idx = [eligible[i] for i in b]
        train_idx = [eligible[i] for i in a] + rare
    train = [clean[i] for i in train_idx]
    model = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2, 5), sublinear_tf=True, max_features=40000)),
        ('model', LogisticRegression(max_iter=1500, class_weight='balanced')),
    ])
    model.fit([x for x, _ in train], [y for _, y in train])
    report = {'task': field, 'train_count': len(train), 'test_count': len(test_idx), 'classes': len(counts), 'conflicting_names_excluded': len(conflicts), 'threshold': threshold}
    if test_idx:
        truth = [clean[i][1] for i in test_idx]
        pred = model.predict([clean[i][0] for i in test_idx])
        report.update({'accuracy': float(accuracy_score(truth, pred)), 'macro_f1': float(f1_score(truth, pred, average='macro', zero_division=0)), 'classification_report': classification_report(truth, pred, zero_division=0, output_dict=True)})
        print(f"  Test accuracy={report['accuracy']:.3f}; macro F1={report['macro_f1']:.3f} (small test set; interpret cautiously)")
    else:
        report['warning'] = 'Not enough data for a meaningful holdout test'
    # Refit final deployable model on all available nonconflicting examples.
    model.fit([x for x, _ in clean], [y for _, y in clean])
    joblib.dump(model, folder / f'{field}_model.joblib')
    (folder / f'{field}_metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='grocery_merged_for_training.csv')
    ap.add_argument('--review', default='Сверка категорий_заполнено.xlsx')
    ap.add_argument('--out', default='grocery_models')
    ap.add_argument('--include-unverified', action='store_true', help='Use automatic labels as well; may introduce label noise')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data = rows_csv(args.csv)
    reviews = read_review(args.review)
    for r in data:
        manual = reviews.get(str(r.get('product_id', '')).strip())
        if manual:
            r['human_category'] = manual['human_category']
            r['human_brand'] = manual['human_brand']
        r['category_label'] = (r.get('human_category') or '').strip() or ((r.get('category') or '').strip() if args.include_unverified else '')
        r['brand_label'] = (r.get('human_brand') or '').strip() or ((r.get('brand') or '').strip() if args.include_unverified and not manual else '')
    # Remove obvious misspelling only if still present in old labels.
    for r in data:
        if r['category_label'] == 'Шокалад':
            r['category_label'] = 'Шоколад'
    for field in ('category_label', 'brand_label'):
        train_task(data, field, out)
    print(f'\nModels and evaluation reports saved in {out.resolve()}')

if __name__ == '__main__':
    main()
