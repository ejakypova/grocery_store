"""Обучение классификатора пар товаров: same / alternative / different.

Запуск: python train_product_matching.py
Зависимости: pip install pandas numpy scipy scikit-learn psycopg[binary] python-dotenv joblib

"""
import json
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

load_dotenv()
OUT = Path(__file__).resolve().parent / 'matching_results'
OUT.mkdir(exist_ok=True)
LABELS = ['same', 'alternative', 'different']


def normalize(s):
    s = '' if pd.isna(s) else str(s)
    return re.sub(r'\s+', ' ', s.lower().replace('ё', 'е')).strip()


def pair_text(a, b):
    # Канонический порядок: сравнение товар с ID а, ID b, тоже самое что и товар с ID b, ID a, 
    # это для исключения дублирования.
    a, b = sorted((normalize(a), normalize(b)))
    return f'{a} [SEP] {b}'


def main():
    config = dict(host=os.getenv('DB_HOST', 'localhost'),
                  port=int(os.getenv('DB_PORT', '5432')),
                  dbname=os.getenv('DB_NAME', 'grocery_store'),
                  user=os.getenv('DB_USER', 'postgres'))
    if os.getenv('DB_PASSWORD'):
        config['password'] = os.getenv('DB_PASSWORD')
    sql = '''
        SELECT a.product_id_a, a.product_id_b, a.human_label,
               COALESCE(p.name_a, x.original_name, x.name_clean, '') AS name_a,
               COALESCE(p.name_b, y.original_name, y.name_clean, '') AS name_b
        FROM grocery_clean.product_pair_annotations a
        LEFT JOIN grocery_clean.product_pair_candidates p
          ON p.product_id_a = a.product_id_a AND p.product_id_b = a.product_id_b
        LEFT JOIN grocery_clean.products_clean x ON x.product_id = a.product_id_a
        LEFT JOIN grocery_clean.products_clean y ON y.product_id = a.product_id_b
    '''
    with psycopg.connect(**config) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [c.name for c in cur.description]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        raise ValueError('Нет размеченных пар в product_pair_annotations.')
    df = df[df.human_label.isin(LABELS)].copy()
    df['text_pair'] = [pair_text(a, b) for a, b in zip(df.name_a, df.name_b)]
    df = df[(df.name_a.map(normalize) != '') & (df.name_b.map(normalize) != '')]
    df['pair_key'] = df.apply(lambda r: tuple(sorted((int(r.product_id_a), int(r.product_id_b)))), axis=1)
    if df.pair_key.duplicated().any():
        raise ValueError('Обнаружены повторяющиеся пары product_id; проверьте разметку.')
    counts = df.human_label.value_counts()
    print('Размеченные пары:', len(df), '\n', counts.to_string())
    if len(counts) < 3 or counts.min() < 2:
        raise ValueError('Для train/test нужно минимум 2 примера каждого из трёх классов.')

    train, test = train_test_split(df, test_size=0.20, random_state=42,
                                   stratify=df.human_label)
    train.to_csv(OUT / 'train_pairs.csv', index=False, encoding='utf-8-sig')
    test.to_csv(OUT / 'test_pairs.csv', index=False, encoding='utf-8-sig')
    # Обе модели используют одинаковый векторизатор и одинаковое разбиение.
    models = {
        'logistic_regression': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample',
                                                 random_state=42, n_jobs=-1, min_samples_leaf=2)
    }
    results = {}
    for name, estimator in models.items():
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2, 4),
                                      max_features=15000, sublinear_tf=True)),
            ('model', estimator)
        ])
        pipeline.fit(train.text_pair, train.human_label)
        predicted = pipeline.predict(test.text_pair)
        report = classification_report(test.human_label, predicted, labels=LABELS,
                                       output_dict=True, zero_division=0)
        same_precision = report['same']['precision']
        results[name] = {
            'accuracy': accuracy_score(test.human_label, predicted),
            'macro_f1': report['macro avg']['f1-score'],
            'same_precision': same_precision,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(test.human_label, predicted, labels=LABELS).tolist()
        }
        joblib.dump(pipeline, OUT / f'{name}.joblib')
        prediction_df = test[['product_id_a', 'product_id_b', 'name_a', 'name_b', 'human_label']].copy()
        prediction_df['predicted_label'] = predicted
        prediction_df['correct'] = prediction_df.human_label == prediction_df.predicted_label
        prediction_df.to_csv(OUT / f'{name}_predictions.csv', index=False, encoding='utf-8-sig')
        print(f'{name}: accuracy={results[name]["accuracy"]:.3f}, '
              f'macro-F1={results[name]["macro_f1"]:.3f}, '
              f'precision(same)={same_precision:.3f}')
    best = max(results, key=lambda name: (results[name]['same_precision'], results[name]['macro_f1']))
    # Сохраняем копию лучшей модели для приложения.
    joblib.dump(joblib.load(OUT / f'{best}.joblib'), OUT / 'product_matching_model.joblib')
    results['selected_model'] = best
    results['train_count'] = len(train)
    results['test_count'] = len(test)
    results['labels_order'] = LABELS
    results['note'] = ('Маленький тест (около 60 пар) и мало different: '
                       'метрики предварительные. Пары с общими товарами могут '
                       'попадать в train и test, поэтому оценка может быть оптимистичной.')
    (OUT / 'comparison_metrics.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Лучшая модель:', best)
    print('Результаты сохранены в:', OUT)


if __name__ == '__main__':
    main()
