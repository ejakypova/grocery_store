"""Универсальное предварительное обучение категорий: PostgreSQL / CSV+Excel / все источники.

Примеры:
 python train_category_optimized_2.py --source db --run-id 2
 python train_category_optimized_2.py --source files --csv grocery_merged_for_training.csv --review "Сверка категорий_заполнено.xlsx"
 python train_category_optimized_2.py --source all --run-id 2 --csv grocery_merged_for_training.csv --review "Сверка категорий_заполнено.xlsx"

"""
import argparse
import csv
import getpass
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline


def norm(value):
    value = unicodedata.normalize('NFKC', str(value or '')).lower().replace('ё', 'е')
    value = re.sub(r'[^\w\s]', ' ', value)
    return re.sub(r'\s+', ' ', value).strip()

def valid_category(value):
    if value is None:
        return False
    value = str(value).strip()
    return value.lower() not in {'', 'null', 'none', 'nan', 'nat', 'не определено'}


# Разрешаем только заранее согласованные конфликты общего и детального классов.
# Подтвержденные вручную метки всегда имеют приоритет.
SPECIFIC_OVER_GENERIC = {
    'Непродовольственные: гигиена и быт': {
        'Мыло', 'Шампунь', 'Зубная паста', 'Влажные салфетки',
        'Зубная щетка', 'Жидкое мыло', 'Гель для душа', 'Гель для стирки',
    },
    'Непродовольственные: подгузники': {'Памперсы и трусики'},
}


def resolve_known_conflict(labels):
    labels = set(labels)
    if len(labels) != 2:
        return None
    for generic, specific in SPECIFIC_OVER_GENERIC.items():
        if generic in labels:
            detailed = labels - {generic}
            if len(detailed) == 1 and next(iter(detailed)) in specific:
                return next(iter(detailed))
    return None


def read_db(run_id):
    import psycopg
    load_dotenv()
    password = os.getenv('DB_PASSWORD') or getpass.getpass('Пароль PostgreSQL: ')
    with psycopg.connect(host=os.getenv('DB_HOST', 'localhost'),
                         port=int(os.getenv('DB_PORT', '5432')),
                         dbname=os.getenv('DB_NAME', 'grocery_store'),
                         user=os.getenv('DB_USER', 'postgres'), password=password) as conn:
        with conn.cursor() as cur:
            cur.execute("""SELECT source_id, data->>'original_name', data->>'category',
                                  data->>'category_source'
                           FROM grocery_clean.records
                           WHERE run_id=%s AND kind='product'""", (run_id,))
            rows = cur.fetchall()
    return [{'id': str(pid), 'name': name, 'label': label, 'verified': source == 'human_verified',
             'origin': 'db'} for pid, name, label, source in rows if valid_category(label) and name]


def read_files(csv_path, review_path):
    records = []
    if csv_path:
        with open(csv_path, encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                name = row.get('original_name') or row.get('product_name') or row.get('name')
                label = row.get('category')
                verified = (row.get('review_status') or '').strip().lower() in ('подтверждено', 'confirmed', 'verified')
                if name and valid_category(label):
                    records.append({'id': str(row.get('product_id') or ''), 'name': name,
                                    'label': label, 'verified': verified, 'origin': 'csv'})
    if review_path:
        frame = pd.read_excel(review_path, dtype=str).fillna('')
        if 'product_id' not in frame.columns:
            raise ValueError('В Excel необходим столбец product_id')
        reviews = {}
        for _, row in frame.iterrows():
            manual = row.get('human_category')
            proposed = row.get('proposed_category')
            label = str(manual if valid_category(manual) else proposed).strip() if (valid_category(manual) or valid_category(proposed)) else ''
            status = str(row.get('review_status', '')).strip().lower()
            # Файл с human_category — ручная разметка; proposed_category требует подтверждения.
            verified = valid_category(label) and (('human_category' in frame.columns and status in ('', 'подтверждено', 'confirmed', 'verified')) or status in ('подтверждено', 'confirmed', 'verified'))
            if verified:
                pid = str(row['product_id']).strip()
                if pid in reviews and reviews[pid] != label:
                    raise ValueError(f'Противоречивая ручная разметка product_id={pid}')
                reviews[pid] = label
        # Для Excel с именами используем их напрямую; иначе накладываем на CSV по product_id.
        for _, row in frame.iterrows():
            pid = str(row['product_id']).strip()
            name = row.get('original_name') or row.get('product_name') or row.get('name') or ''
            if pid in reviews and name:
                records.append({'id': pid, 'name': name, 'label': reviews[pid], 'verified': True, 'origin': 'excel'})
        for record in records:
            if record['id'] in reviews:
                record['label'] = reviews[record['id']]
                record['verified'] = True
                record['origin'] = 'excel_verified'
        missing = set(reviews) - {r['id'] for r in records}
        if missing:
            print(f'Предупреждение: {len(missing)} подтверждённых Excel-ID без названия в доступных источниках; пропущены')
    return records


def merge_records(records):
    """Объединение по ID и имени, с журналом автоматических решений."""
    by_id = defaultdict(list)
    for row in records:
        name = norm(row['name'])
        label = str(row['label'] or '').strip()
        if name and valid_category(label):
            key = ('id', row['id']) if row['id'] else ('name', name)
            by_id[key].append({**row, 'name': name, 'label': label})

    resolved, conflicts, auto_resolved = [], [], []
    for key, rows in by_id.items():
        verified = {r['label'] for r in rows if r['verified']}
        labels = {r['label'] for r in rows}
        if len(verified) > 1:
            conflicts.append({'key': str(key), 'labels': ' | '.join(sorted(verified)), 'reason': 'verified_conflict'})
            continue
        if verified:
            label = next(iter(verified))
            chosen = next(r for r in rows if r['verified'] and r['label'] == label)
        else:
            label = next(iter(labels)) if len(labels) == 1 else resolve_known_conflict(labels)
            if label is None:
                conflicts.append({'key': str(key), 'labels': ' | '.join(sorted(labels)), 'reason': 'rule_conflict'})
                continue
            chosen = next(r for r in rows if r['label'] == label)
            if len(labels) > 1:
                auto_resolved.append({'key': str(key), 'name': chosen['name'],
                                      'original_labels': ' | '.join(sorted(labels)),
                                      'selected_category': label, 'rule': 'specific_over_generic'})
        resolved.append({**chosen, 'label': label})

    by_name = defaultdict(list)
    for row in resolved:
        by_name[row['name']].append(row)
    unique = []
    for name, rows in by_name.items():
        verified = {r['label'] for r in rows if r['verified']}
        labels = {r['label'] for r in rows}
        if len(verified) > 1:
            conflicts.append({'key': name, 'labels': ' | '.join(sorted(labels)), 'reason': 'verified_name_conflict'})
            continue
        if verified:
            label = next(iter(verified))
        elif len(labels) == 1:
            label = next(iter(labels))
        else:
            label = resolve_known_conflict(labels)
            if label is None:
                conflicts.append({'key': name, 'labels': ' | '.join(sorted(labels)), 'reason': 'name_conflict'})
                continue
            auto_resolved.append({'key': name, 'name': name,
                                  'original_labels': ' | '.join(sorted(labels)),
                                  'selected_category': label, 'rule': 'specific_over_generic_name'})
        chosen = next((r for r in rows if r['verified'] and r['label'] == label),
                      next(r for r in rows if r['label'] == label))
        unique.append({**chosen, 'label': label, 'verified': bool(verified)})
    return unique, conflicts, auto_resolved


def make_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=30000, sublinear_tf=True)),
        ('model', LogisticRegression(max_iter=2000, class_weight='balanced')),
    ])


def score(model, rows):
    if not rows:
        return None
    y = [r['label'] for r in rows]
    pred = model.predict([r['name'] for r in rows])
    return {'n': len(rows), 'accuracy': float(accuracy_score(y, pred)),
            'precision_macro': float(precision_score(y, pred, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(y, pred, average='macro', zero_division=0)),
            'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
            'classification_report': classification_report(y, pred, output_dict=True, zero_division=0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['db', 'files', 'all'], default='db')
    parser.add_argument('--run-id', type=int, default=2)
    parser.add_argument('--csv')
    parser.add_argument('--review')
    parser.add_argument('--out', default='category_optimization_final')
    args = parser.parse_args()
    if args.source == 'files' and not (args.csv or args.review):
        parser.error('Для --source files нужен --csv и/или --review')
    rows = []
    if args.source in ('db', 'all'):
        rows += read_db(args.run_id)
    if args.source in ('files', 'all'):
        rows += read_files(args.csv, args.review)
    clean, conflicts, auto_resolved = merge_records(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(conflicts, columns=['key', 'labels', 'reason']).to_csv(out/'conflicting_labels.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(auto_resolved, columns=['key', 'name', 'original_labels', 'selected_category', 'rule']).to_csv(
        out/'auto_resolved_conflicts.csv', index=False, encoding='utf-8-sig')
    if len(clean) < 30:
        raise SystemExit('Недостаточно размеченных названий (минимум 30).')
    counts = Counter(r['label'] for r in clean)
    common = [r for r in clean if counts[r['label']] >= 3]
    rare = [r for r in clean if counts[r['label']] < 3]
    n_classes = len({r['label'] for r in common})
    if n_classes < 2:
        raise SystemExit('Недостаточно категорий с 3+ примерами')
    test_n = max(int(np.ceil(len(common)*0.2)), n_classes)
    if len(common) - test_n < n_classes:
        raise SystemExit('Недостаточно данных для стратифицированного разбиения')
    train, test = train_test_split(common, test_size=test_n, random_state=42,
                                   stratify=[r['label'] for r in common])
    train += rare
    print(f'Уникальных названий: {len(clean)}; Train: {len(train)}; Test: {len(test)}; конфликтов: {len(conflicts)}')
    X_train, y_train = [r['name'] for r in train], [r['label'] for r in train]
    baseline = make_pipeline().fit(X_train, y_train)
    train_counts = Counter(y_train)
    cv_rows = [r for r in train if train_counts[r['label']] >= 3]
    if len({r['label'] for r in cv_rows}) < 2:
        raise SystemExit('Недостаточно классов для GridSearchCV')
    grid = {'tfidf__ngram_range': [(2, 4), (2, 5), (3, 5)],
            'tfidf__max_features': [15000, 30000], 'model__C': [0.5, 1.0, 3.0]}
    search = GridSearchCV(make_pipeline(), grid, scoring='f1_macro', n_jobs=-1,
                          cv=StratifiedKFold(3, shuffle=True, random_state=42), refit=True)
    search.fit([r['name'] for r in cv_rows], [r['label'] for r in cv_rows])
    optimized = clone(search.best_estimator_).fit(X_train, y_train)
    metrics = {'source': args.source, 'run_id': args.run_id if args.source != 'files' else None,
               'unique_names': len(clean), 'train': len(train), 'test': len(test),
               'rare_train_only': len(rare), 'conflicts': len(conflicts), 'auto_resolved_conflicts': len(auto_resolved),
               'verified_train': sum(r['verified'] for r in train),
               'verified_test': sum(r['verified'] for r in test),
               'best_params': search.best_params_, 'cv_f1_macro': float(search.best_score_),
               'baseline_test': score(baseline, test), 'optimized_test': score(optimized, test),
               'verified_test_only': score(optimized, [r for r in test if r['verified']]),
               'warning': 'Автоматические метки rules не являются истиной. Verified test может быть малым и смещённым по категориям.'}
    (out/'comparison_metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    joblib.dump(baseline, out/'category_baseline.joblib')
    joblib.dump(optimized, out/'category_model_optimized.joblib')
    pd.DataFrame(search.cv_results_).to_csv(out/'gridsearch_results.csv', index=False, encoding='utf-8-sig')
    pred = optimized.predict([r['name'] for r in test])
    prob = optimized.predict_proba([r['name'] for r in test]).max(axis=1)
    report = pd.DataFrame([{'name': r['name'], 'true_category': r['label'], 'predicted_category': p,
                            'confidence': float(q), 'verified': r['verified'], 'correct': r['label'] == p}
                           for r, p, q in zip(test, pred, prob)])
    report.to_csv(out/'test_predictions.csv', index=False, encoding='utf-8-sig')
    report[~report['correct']].to_csv(out/'errors_for_review.csv', index=False, encoding='utf-8-sig')
    with pd.ExcelWriter(out/'category_test_error_report.xlsx', engine='openpyxl') as writer:
        pd.DataFrame([{'model': 'baseline', **{k:v for k,v in metrics['baseline_test'].items() if k != 'classification_report'}},
                      {'model': 'optimized', **{k:v for k,v in metrics['optimized_test'].items() if k != 'classification_report'}}]).to_excel(writer, sheet_name='Итоги', index=False)
        report.to_excel(writer, sheet_name='Тестовые товары', index=False)
        report[~report['correct']].to_excel(writer, sheet_name='Ошибки', index=False)
    print(f'Автоматически разрешено конфликтов: {len(auto_resolved)}; осталось на проверку: {len(conflicts)}')
    print('Baseline macro-F1:', round(metrics['baseline_test']['macro_f1'], 4))
    print('Optimized macro-F1:', round(metrics['optimized_test']['macro_f1'], 4))
    print('Лучшие параметры:', search.best_params_)
    print('Отчёты:', out.resolve())


if __name__ == '__main__':
    main()
