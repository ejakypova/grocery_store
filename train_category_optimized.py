"""Сравнение исходной и оптимизированной модели категорий Grocery Store.

Запуск:
 python train_category_optimized.py --include-unverified
 python train_category_optimized.py --csv grocery_merged_for_training.csv --review 'Сверка категорий_заполнено.xlsx' --include-unverified

Не изменяет существующие модели в grocery_models.
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
from openpyxl import load_workbook, Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline


def norm(value):
    value = unicodedata.normalize('NFKC', str(value or '')).lower().replace('ё', 'е')
    value = re.sub(r'[^\w\s]', ' ', value)
    return re.sub(r'\s+', ' ', value).strip()


def load_reviews(path):
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    rows = ws.iter_rows(values_only=True)
    headers = [str(v).strip() if v is not None else '' for v in next(rows)]
    reviews = {}
    for vals in rows:
        row = dict(zip(headers, vals))
        if row.get('product_id') is not None:
            reviews[str(row['product_id']).strip()] = str(row.get('human_category') or '').strip()
    wb.close()
    return reviews


def load_data(csv_path, review_path, include_unverified):
    reviews = load_reviews(review_path)
    by_name = {}
    conflicts = set()
    with open(csv_path, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            name = norm(row.get('original_name'))
            if not name:
                continue
            pid = str(row.get('product_id', '')).strip()
            manual = reviews.get(pid, (row.get('human_category') or '').strip())
            label = manual or ((row.get('category') or '').strip() if include_unverified else '')
            if label == 'Шокалад':
                label = 'Шоколад'
            if not label:
                continue
            if name in by_name and by_name[name] != label:
                conflicts.add(name)
            else:
                by_name[name] = label
    for name in conflicts:
        by_name.pop(name, None)
    return list(by_name.items()), len(conflicts)


def metrics(model, X, y):
    pred = model.predict(X)
    return {
        'accuracy': float(accuracy_score(y, pred)),
        'precision_macro': float(precision_score(y, pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y, pred, average='macro', zero_division=0)),
        'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
        'classification_report': classification_report(y, pred, zero_division=0, output_dict=True),
    }

def export_test_report(
    out, X_test, y_test, base, optimized,
    base_metrics, opt_metrics
):
    wb = Workbook()

    # Лист 1. Итоги
    ws = wb.active
    ws.title = "Итоги"
    ws.append([
        "Модель", "Accuracy", "Precision",
        "Recall", "Macro-F1"
    ])

    for name, m in [
        ("Исходная", base_metrics),
        ("Оптимизированная", opt_metrics)
    ]:
        ws.append([
            name,
            m["accuracy"],
            m["precision_macro"],
            m["recall_macro"],
            m["macro_f1"]
        ])

    # Лист 2. Тестовые товары
    ws_test = wb.create_sheet("Тестовые товары")
    ws_test.append([
        "Название товара",
        "Категория из разметки",
        "Категория ML",
        "Уверенность",
        "Результат"
    ])

    # Лист 3. Ошибки
    ws_errors = wb.create_sheet("Ошибки")
    ws_errors.append([
        "Название товара",
        "Категория из разметки",
        "Категория ML",
        "Уверенность",
        "Результат"
    ])

    predictions = optimized.predict(X_test)
    probabilities = optimized.predict_proba(X_test)

    for name, actual, predicted, probs in zip(
        X_test, y_test, predictions, probabilities
    ):
        index = list(optimized.classes_).index(predicted)
        confidence = float(probs[index])

        result = (
            "Верно" if actual == predicted
            else "Ошибка"
        )

        row = [
            name,
            actual,
            predicted,
            confidence,
            result
        ]

        ws_test.append(row)

        if result == "Ошибка":
            ws_errors.append(row)

    # Лист 4. Метрики категорий
    ws_metrics = wb.create_sheet("Метрики категорий")
    ws_metrics.append([
        "Категория",
        "Precision",
        "Recall",
        "F1-score",
        "Support"
    ])

    report = opt_metrics["classification_report"]

    for category, values in report.items():
        if not isinstance(values, dict):
            continue

        if category in ("macro avg", "weighted avg"):
            continue

        ws_metrics.append([
            category,
            values["precision"],
            values["recall"],
            values["f1-score"],
            values["support"]
        ])

    # Оформление
    for sheet in wb.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions

        for cell in sheet[1]:
            cell.font = Font(
                bold=True,
                color="FFFFFF"
            )
            cell.fill = PatternFill(
                "solid",
                fgColor="1F4E78"
            )
            cell.alignment = Alignment(
                wrap_text=True
            )

        for column in sheet.columns:
            letter = column[0].column_letter
            max_length = max(
                len(str(cell.value or ""))
                for cell in column
            )
            sheet.column_dimensions[letter].width = min(
                max(max_length + 3, 15), 55
            )

    path = out / "category_test_error_report.xlsx"
    wb.save(path)

    print(f"Excel-отчёт сохранён: {path}")

def make_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2, 5), sublinear_tf=True, max_features=40000)),
        ('model', LogisticRegression(max_iter=1500, class_weight='balanced')),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='grocery_merged_for_training.csv')
    ap.add_argument('--review', default='Сверка категорий_заполнено.xlsx')
    ap.add_argument('--include-unverified', action='store_true')
    ap.add_argument('--out', default='category_optimization')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    clean, conflicts = load_data(args.csv, args.review, args.include_unverified)
    counts = Counter(label for _, label in clean)
    print(f'Уникальных названий: {len(clean)}; категорий: {len(counts)}; конфликтов исключено: {conflicts}')
    if len(clean) < 30 or len(counts) < 2:
        raise SystemExit('Недостаточно данных для сравнения. Попробуйте --include-unverified.')

    # Категории с <3 примерами оставляем только в Train.
    common = [(x, y) for x, y in clean if counts[y] >= 3]
    rare = [(x, y) for x, y in clean if counts[y] < 3]
    if len(common) < 20 or len(set(y for _, y in common)) < 2:
        raise SystemExit('Недостаточно категорий с 3+ примерами для тестирования.')

    Xc, yc = zip(*common)
    # Не стратифицируем, если 20% теста меньше количества классов.
    n_classes = len(set(yc))
    n_test = int(np.ceil(0.2 * len(common)))
    stratify = list(yc) if n_test >= n_classes and len(common) - n_test >= n_classes else None
    X_train, X_test, y_train, y_test = train_test_split(
        list(Xc), list(yc), test_size=0.2, random_state=42, stratify=stratify
    )
    # При отсутствии стратификации гарантируем, что все классы Test присутствуют в Train.
    missing = set(y_test) - set(y_train)
    if missing:
        retained = [(x, y) for x, y in zip(X_test, y_test) if y not in missing]
        moved = [(x, y) for x, y in zip(X_test, y_test) if y in missing]
        X_test, y_test = [x for x, _ in retained], [y for _, y in retained]
        X_train += [x for x, _ in moved]
        y_train += [y for _, y in moved]
    X_train += [x for x, _ in rare]
    y_train += [y for _, y in rare]
    if not X_test:
        raise SystemExit('Тестовая выборка пуста.')

    print(f'Train: {len(X_train)}, Test: {len(X_test)}, редких только в Train: {len(rare)}')
    base = make_pipeline()
    base.fit(X_train, y_train)
    base_metrics = metrics(base, X_test, y_test)
    print(f"Исходная: accuracy={base_metrics['accuracy']:.3f}; macro-F1={base_metrics['macro_f1']:.3f}")

    # GridSearchCV выполняется только на Train, Test не участвует в подборе.
    train_counts = Counter(y_train)
    cv_classes = {k for k, v in train_counts.items() if v >= 3}
    # Чтобы CV не обучался на классах с единственным примером, подбираем параметры
    # на достаточно частых классах; затем переобучаем лучшую конфигурацию на всём Train.
    X_cv = [x for x, y in zip(X_train, y_train) if y in cv_classes]
    y_cv = [y for y in y_train if y in cv_classes]
    if len(cv_classes) < 2:
        raise SystemExit('Недостаточно классов с 3+ примерами для GridSearchCV.')
    grid = {
        'tfidf__ngram_range': [(2, 4), (2, 5), (3, 5)],
        'tfidf__max_features': [20000, 40000],
        'model__C': [0.5, 1.0, 3.0],
    }
    search = GridSearchCV(
        make_pipeline(), grid, scoring='f1_macro',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        n_jobs=-1, refit=True, error_score='raise', verbose=1,
    )
    search.fit(X_cv, y_cv)
    optimized = make_pipeline()
    optimized.set_params(**search.best_params_)
    optimized.fit(X_train, y_train)
    opt_metrics = metrics(optimized, X_test, y_test)
    print(f"Оптимизированная: accuracy={opt_metrics['accuracy']:.3f}; macro-F1={opt_metrics['macro_f1']:.3f}")
    print('Лучшие параметры:', search.best_params_)

    # Сохраняем оцененные модели, обученные только на Train, чтобы результат был воспроизводим.
    joblib.dump(base, out / 'category_baseline.joblib')
    joblib.dump(optimized, out / 'category_model_optimized.joblib')
    report = {
        'train_count': len(X_train), 'test_count': len(X_test),
        'rare_train_only': len(rare), 'conflicts_excluded': conflicts,
        'include_unverified': args.include_unverified,
        'cv_best_params': search.best_params_, 'cv_best_macro_f1': float(search.best_score_),
        'baseline': base_metrics, 'optimized': opt_metrics,
        'note': 'Test использован только для финальной оценки; метрики применимы к этому разбиению и качеству его меток.',
    }
    (out / 'comparison_metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with open(out / 'comparison.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'accuracy', 'precision_macro', 'recall_macro', 'macro_f1'])
        for key in ('baseline', 'optimized'):
            m = report[key]
            w.writerow([key, m['accuracy'], m['precision_macro'], m['recall_macro'], m['macro_f1']])
            export_test_report(
                out=out,
                X_test=X_test,
                y_test=y_test,
                base=base,
                optimized=optimized,
                base_metrics=base_metrics,
                opt_metrics=opt_metrics
            )
    print('Результаты сохранены в:', out.resolve())
   


if __name__ == '__main__':
    main()
