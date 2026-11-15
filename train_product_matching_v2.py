"""Обучение Product Matching v2 на ручной разметке из PostgreSQL.
Запуск: python train_product_matching_v2.py
product_matching_features_v2.py должен лежать в одной папке. Таблицы БД не изменяются.
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

from product_matching_features_v2 import PairFeatureExtractor, clean

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / '.env')
OUT = BASE / 'matching_results_v2'
LABELS = ['same', 'alternative', 'different']

SQL = '''
SELECT a.product_id_a, a.product_id_b, a.human_label,
       COALESCE(NULLIF(p.name_a, ''), NULLIF(x.original_name, ''), x.name_clean, '') AS name_a,
       COALESCE(NULLIF(p.name_b, ''), NULLIF(y.original_name, ''), y.name_clean, '') AS name_b,
       x.brand AS brand_a, y.brand AS brand_b,
       x.category AS category_a, y.category AS category_b,
       x.total_quantity AS quantity_a, y.total_quantity AS quantity_b,
       x.normalized_unit AS unit_a, y.normalized_unit AS unit_b
FROM grocery_clean.product_pair_annotations a
LEFT JOIN grocery_clean.product_pair_candidates p
    ON p.product_id_a = a.product_id_a AND p.product_id_b = a.product_id_b
LEFT JOIN grocery_clean.products_clean x ON x.product_id = a.product_id_a
LEFT JOIN grocery_clean.products_clean y ON y.product_id = a.product_id_b
'''

FEATURE_COLUMNS = ['name_a','name_b','brand_a','brand_b','category_a','category_b',
                   'quantity_a','quantity_b','unit_a','unit_b']


def connected_groups(df):
    """Все пары с общим product_id попадают в одну компоненту связности."""
    parent = {}
    def root(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[rb] = ra
    for a,b in zip(df.product_id_a, df.product_id_b):
        union(int(a), int(b))
    return np.array([root(int(a)) for a in df.product_id_a])


def split_pairs(df):
    groups = connected_groups(df)
    n_groups = len(set(groups))
    # Ищем групповой split, где каждый класс представлен и в train, и в test.
    if n_groups >= 3:
        splitter = GroupShuffleSplit(n_splits=500, test_size=0.2, random_state=42)
        choices = []
        for tr, te in splitter.split(df, df.human_label, groups):
            ct, cv = df.iloc[tr].human_label.value_counts(), df.iloc[te].human_label.value_counts()
            if all(ct.get(k,0) >= 2 and cv.get(k,0) >= 2 for k in LABELS):
                # Примерно 20% test и приблизительно сохранённые пропорции классов.
                error = abs(len(te)/len(df)-0.2)
                error += sum(abs(cv.get(k,0)/len(te) - df.human_label.eq(k).mean()) for k in LABELS)
                choices.append((error, tr, te))
        if choices:
            _, tr, te = min(choices, key=lambda x:x[0])
            return df.iloc[tr].copy(), df.iloc[te].copy(), 'group_disjoint', n_groups
    # Честно сообщаем об ограничении; не выдаём случайный split за независимый по товарам.
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df.human_label)
    return train.copy(), test.copy(), 'stratified_pair_split_possible_product_overlap', n_groups


def main():
    config = dict(host=os.getenv('DB_HOST','localhost'), port=int(os.getenv('DB_PORT','5432')),
                  dbname=os.getenv('DB_NAME','grocery_store'), user=os.getenv('DB_USER','postgres'))
    if os.getenv('DB_PASSWORD'):
        config['password'] = os.getenv('DB_PASSWORD')
    with psycopg.connect(**config) as conn:
        with conn.cursor() as cur:
            cur.execute(SQL)
            rows = cur.fetchall()
            cols = [c.name for c in cur.description]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        raise ValueError('Нет размеченных пар в product_pair_annotations.')
    df = df[df.human_label.isin(LABELS)].copy()
    df = df[df.name_a.map(clean).ne('') & df.name_b.map(clean).ne('')].copy()
    df['pair_key'] = df.apply(lambda r: tuple(sorted((int(r.product_id_a), int(r.product_id_b)))), axis=1)
    if df.pair_key.duplicated().any():
        raise ValueError('Повторяющиеся пары product_id. Проверьте базу.')
    counts = df.human_label.value_counts()
    if len(counts) != 3 or counts.min() < 3:
        raise ValueError('Для обучения нужны все три класса, минимум 3 примера каждого.')
    print('Размеченные пары:',len(df),'\n',counts.to_string())
    print('Пропущен бренд хотя бы у одного товара:', int((df.brand_a.map(clean).eq('') | df.brand_b.map(clean).eq('')).sum()))
    print('Пропущена категория хотя бы у одного товара:', int((df.category_a.map(clean).eq('') | df.category_b.map(clean).eq('')).sum()))

    train, test, split_method, n_groups = split_pairs(df)
    overlap = (set(train.product_id_a) | set(train.product_id_b)) & (set(test.product_id_a) | set(test.product_id_b))
    print(f'Split: {split_method}; train={len(train)}, test={len(test)}, общих product_id={len(overlap)}')
    if overlap:
        print('ВНИМАНИЕ: товарные ID пересекаются между train/test; оценка может быть оптимистичной.')
    OUT.mkdir(exist_ok=True)
    train.drop(columns=['pair_key']).to_csv(OUT/'train_pairs.csv',index=False,encoding='utf-8-sig')
    test.drop(columns=['pair_key']).to_csv(OUT/'test_pairs.csv',index=False,encoding='utf-8-sig')
    models = {
        'logistic_regression': LogisticRegression(max_iter=3000, class_weight='balanced',random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=350,min_samples_leaf=2,
                              class_weight='balanced_subsample',random_state=42,n_jobs=-1)
    }
    results = {}
    for name, estimator in models.items():
        pipe = Pipeline([('features',PairFeatureExtractor(max_features=15000)),('model',estimator)])
        pipe.fit(train[FEATURE_COLUMNS],train.human_label)
        pred = pipe.predict(test[FEATURE_COLUMNS])
        proba = pipe.predict_proba(test[FEATURE_COLUMNS])
        report = classification_report(test.human_label,pred,labels=LABELS,output_dict=True,zero_division=0)
        results[name] = {
            'accuracy':float(accuracy_score(test.human_label,pred)),
            'macro_f1':float(report['macro avg']['f1-score']),
            'same_precision':float(report['same']['precision']),
            'same_recall':float(report['same']['recall']),
            'classification_report':report,
            'confusion_matrix':confusion_matrix(test.human_label,pred,labels=LABELS).tolist()
        }
        joblib.dump(pipe,OUT/f'{name}.joblib')
        prediction = test[['product_id_a','product_id_b','name_a','name_b','human_label']].copy()
        prediction['predicted_label'] = pred
        prediction['correct'] = prediction.human_label.eq(prediction.predicted_label)
        for i,label in enumerate(pipe.named_steps['model'].classes_):
            prediction[f'prob_{label}'] = proba[:,i]
        prediction.to_csv(OUT/f'{name}_predictions.csv',index=False,encoding='utf-8-sig')
        print(f'{name}: accuracy={results[name]["accuracy"]:.3f}, '
              f'macro-F1={results[name]["macro_f1"]:.3f}, '
              f'precision(same)={results[name]["same_precision"]:.3f}')
    best = max(results,key=lambda name:(results[name]['same_precision'],results[name]['macro_f1']))
    joblib.dump(joblib.load(OUT/f'{best}.joblib'),OUT/'product_matching_model.joblib')
    results.update(selected_model=best,train_count=len(train),test_count=len(test),
                   total_pairs=len(df),split_method=split_method,connected_components=n_groups,
                   overlapping_product_ids=len(overlap),labels_order=LABELS,
                   note='Метрики предварительные; выбор лучшей модели по тесту делает оценку выбранной модели оптимистичной. '
                        'Для финальной оценки нужен отдельный независимый holdout. '
                        'Пропуски бренда/категории не считаются несовпадением.')
    (OUT/'comparison_metrics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Лучшая модель:',best)
    print('Результаты сохранены:',OUT)

if __name__ == '__main__':
    main()
