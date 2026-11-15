"""Генерация кандидатов пар для ручной разметки в Grocery Store.

Запуск: python generate_product_pairs.py --limit 1000

"""
import argparse
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def clean_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).lower().replace("ё", "е")).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--min-similarity", type=float, default=0.12)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    if args.limit < 1 or args.neighbors < 1:
        parser.error("--limit и --neighbors должны быть положительными")

    password = os.getenv("DB_PASSWORD")
    conn_args = dict(host=os.getenv("DB_HOST", "localhost"),
                     port=int(os.getenv("DB_PORT", "5432")),
                     dbname=os.getenv("DB_NAME", "grocery_store"),
                     user=os.getenv("DB_USER", "postgres"))
    if password:
        conn_args["password"] = password

    with psycopg.connect(**conn_args) as conn:
        rows = conn.execute("""
            SELECT p.product_id, p.store_id, p.original_name, p.name_clean,
                   p.brand, p.category, p.total_quantity, p.normalized_unit,
                   l.category AS confirmed_category
            FROM grocery_clean.products_clean p
            LEFT JOIN grocery_clean.product_category_labels l
              ON l.product_id = p.product_id AND l.review_status = 'подтверждено'
            WHERE p.store_id IS NOT NULL
              AND COALESCE(NULLIF(TRIM(p.name_clean), ''), NULLIF(TRIM(p.original_name), '')) IS NOT NULL
            ORDER BY p.product_id
        """).fetchall()
        cols = ["product_id", "store_id", "original_name", "name_clean", "brand", "category", "total_quantity", "normalized_unit", "confirmed_category"]
        df = pd.DataFrame(rows, columns=cols).drop_duplicates("product_id")
        if len(df) < 2 or df.store_id.nunique() < 2:
            print("Недостаточно товаров из разных магазинов")
            return
        df["text"] = df.apply(lambda r: clean_text(r["name_clean"] or r["original_name"]), axis=1)
        df["effective_category"] = df["confirmed_category"].fillna(df["category"]).map(clean_text)
        df["brand_key"] = df["brand"].map(clean_text)
        df["unit_key"] = df["normalized_unit"].map(clean_text)
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1, sublinear_tf=True)
        matrix = vectorizer.fit_transform(df["text"])
        n_neighbors = min(len(df), args.neighbors + 1)
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
        knn.fit(matrix)
        distances, indices = knn.kneighbors(matrix)
        candidates = {}
        for i in range(len(df)):
            a = df.iloc[i]
            for dist, j in zip(distances[i], indices[i]):
                if i == j:
                    continue
                b = df.iloc[int(j)]
                if int(a.store_id) == int(b.store_id):
                    continue
                sim = float(np.clip(1 - dist, 0, 1))
                if sim < args.min_similarity:
                    continue
                aid, bid = sorted((int(a.product_id), int(b.product_id)))
                key = (aid, bid)
                if key in candidates and candidates[key]["similarity"] >= sim:
                    continue
                cat_a, cat_b = a.effective_category, b.effective_category
                category = cat_a if cat_a and cat_a == cat_b else None
                reason = "tfidf_char_cross_store"
                if cat_a and cat_b and cat_a != cat_b:
                    reason += ";category_disagreement"
                if a.brand_key and b.brand_key and a.brand_key != b.brand_key:
                    reason += ";brand_disagreement"
                if a.unit_key and b.unit_key and a.unit_key != b.unit_key:
                    reason += ";unit_disagreement"
                first, second = (a, b) if int(a.product_id) == aid else (b, a)
                candidates[key] = dict(product_id_a=aid, product_id_b=bid,
                                       name_a=str(first.original_name or first.name_clean),
                                       name_b=str(second.original_name or second.name_clean),
                                       category=category, similarity=round(sim, 5), reason=reason)
        pairs = list(candidates.values())
        # Разные уровни похожести для более разнообразной разметки.
        bands = [(0.75, 1.01), (0.4, 0.75), (args.min_similarity, 0.4)]
        selected = []
        seen = set()
        per_band = max(1, args.limit // len(bands))
        for low, high in bands:
            pool = sorted((p for p in pairs if low <= p["similarity"] < high),
                          key=lambda p: (-p["similarity"], p["product_id_a"], p["product_id_b"]))
            for p in pool[:per_band]:
                key = (p["product_id_a"], p["product_id_b"])
                if key not in seen:
                    selected.append(p)
                    seen.add(key)
        for p in sorted(pairs, key=lambda x: -x["similarity"]):
            if len(selected) >= args.limit:
                break
            key = (p["product_id_a"], p["product_id_b"])
            if key not in seen:
                selected.append(p)
                seen.add(key)
        selected = selected[:args.limit]
        pd.DataFrame(selected, columns=["product_id_a", "product_id_b", "name_a", "name_b", "category", "similarity", "reason"]).to_csv(
            "pairs_for_review.csv", index=False, encoding="utf-8-sig")
        print(f"Товаров: {len(df)}; найдено пар: {len(pairs)}; отобрано: {len(selected)}")
        if args.dry_run:
            print("Пробный запуск: PostgreSQL не изменён. CSV сохранён.")
            return
        with conn.transaction():
            with conn.cursor() as cur:
                for p in selected:
                    cur.execute("""
                        INSERT INTO grocery_clean.product_pair_candidates
                        (product_id_a, product_id_b, name_a, name_b, category, similarity, reason)
                        VALUES (%(product_id_a)s, %(product_id_b)s, %(name_a)s, %(name_b)s,
                                %(category)s, %(similarity)s, %(reason)s)
                        ON CONFLICT (product_id_a, product_id_b) DO NOTHING
                    """, p)
        print("Кандидаты сохранены. Существующие пары и ручная разметка не изменены.")


if __name__ == "__main__":
    main()
