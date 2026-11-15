import os
import getpass
from pathlib import Path

import openpyxl
import psycopg
from dotenv import load_dotenv

load_dotenv()

FILE = Path(__file__).resolve().parent / "category_175_review.xlsx"

wb = openpyxl.load_workbook(FILE, read_only=True, data_only=True)
ws = wb.active

rows = list(ws.values)
headers = [str(x).strip() for x in rows[0]]
data = [dict(zip(headers, row)) for row in rows[1:]]

labels = []

for row in data:
    if str(row["review_status"]).strip().lower() != "подтверждено":
        continue

    product_id = int(row["product_id"])
    category = str(row["proposed_category"] or "").strip()

    if category:
        labels.append((product_id, category))

wb.close()

if len(labels) != len(set(pid for pid, _ in labels)):
    raise ValueError("Обнаружены повторяющиеся product_id")

print("Подтвержденных категорий:", len(labels))

password = os.getenv("DB_PASSWORD") or getpass.getpass(
    "Пароль PostgreSQL: "
)

with psycopg.connect(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    dbname=os.getenv("DB_NAME", "grocery_store"),
    user=os.getenv("DB_USER", "postgres"),
    password=password
) as conn:

    with conn.cursor() as cur:
        cur.execute("""
            CREATE SCHEMA IF NOT EXISTS grocery_clean
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS
            grocery_clean.product_category_labels (
                product_id BIGINT PRIMARY KEY,
                category TEXT NOT NULL,
                review_status TEXT NOT NULL
                    DEFAULT 'подтверждено',
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        ids = [pid for pid, _ in labels]

        cur.execute("""
            SELECT id
            FROM public.products
            WHERE id = ANY(%s)
        """, (ids,))

        found = {row[0] for row in cur.fetchall()}
        missing = set(ids) - found

        if missing:
            raise ValueError(
                f"Товары отсутствуют в базе: {sorted(missing)}"
            )

        cur.executemany("""
            INSERT INTO grocery_clean.product_category_labels
                (product_id, category, review_status)
            VALUES (%s, %s, 'подтверждено')
            ON CONFLICT (product_id)
            DO UPDATE SET
                category = EXCLUDED.category,
                review_status = EXCLUDED.review_status,
                updated_at = NOW()
        """, labels)

print("Загрузка завершена!")
