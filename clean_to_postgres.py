import getpass
import json
from decimal import Decimal

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from clean_grocery_store import extract_product, number


# Настройки подключения
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "grocery_store"
DB_USER = "postgres"

MAX_PRICE = Decimal("100000")


def safe_number(value):
    """Преобразует число; NaN и бесконечность заменяет на None."""
    result = number(value)

    if result is None:
        return None

    result = Decimal(str(result))
    return result if result.is_finite() else None


def main():
    password = getpass.getpass("Пароль PostgreSQL: ")

    with psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=password,
        row_factory=dict_row,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ"
            )

            # Защита от одновременного запуска двух копий.
            cur.execute(
                "SELECT pg_advisory_xact_lock(72603142)"
            )

            cur.execute(
                "SELECT * FROM public.products ORDER BY id"
            )
            source_products = cur.fetchall()

            cur.execute(
                "SELECT * FROM public.price_history ORDER BY id"
            )
            source_prices = cur.fetchall()

            cur.execute("""
                CREATE SCHEMA IF NOT EXISTS grocery_clean;

                CREATE TABLE IF NOT EXISTS
                    grocery_clean.products_clean (
                    product_id INTEGER PRIMARY KEY,
                    store_id INTEGER NOT NULL,
                    original_name TEXT NOT NULL,
                    name_clean TEXT NOT NULL,
                    brand TEXT,
                    category TEXT,
                    total_quantity NUMERIC,
                    normalized_unit TEXT,
                    pack_count INTEGER,
                    attributes JSONB NOT NULL,
                    needs_review BOOLEAN NOT NULL,
                    review_reasons TEXT,
                    rules_version TEXT NOT NULL,
                    processed_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );

                CREATE TABLE IF NOT EXISTS
                    grocery_clean.price_history_clean (
                    history_id INTEGER PRIMARY KEY,
                    product_id INTEGER NOT NULL REFERENCES
                        grocery_clean.products_clean(product_id),
                    price_som NUMERIC,
                    old_price_som NUMERIC,
                    collected_at TIMESTAMPTZ NOT NULL,
                    source_status TEXT,
                    price_per_unit NUMERIC,
                    eligible_price BOOLEAN NOT NULL,
                    review_reasons TEXT,
                    processed_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );

                CREATE INDEX IF NOT EXISTS
                    ix_clean_products_category
                    ON grocery_clean.products_clean(category);

                CREATE INDEX IF NOT EXISTS
                    ix_clean_prices_product_date
                    ON grocery_clean.price_history_clean(
                        product_id, collected_at
                    );
            """)

            # Заменяем только результаты очистки.
            cur.execute("""
                DELETE FROM grocery_clean.price_history_clean;
                DELETE FROM grocery_clean.products_clean;
            """)

            products_by_id = {}

            for row in source_products:
                cleaned = extract_product(row)
                product_id = int(cleaned["product_id"])

                cleaned["total_quantity"] = safe_number(
                    cleaned["total_quantity"]
                )

                products_by_id[product_id] = cleaned

                attributes = cleaned.get("attributes")

                # Поддержка JSON как строки или словаря.
                if isinstance(attributes, str):
                    attributes = json.loads(attributes)
                elif attributes is None:
                    attributes = {}

                cur.execute("""
                    INSERT INTO grocery_clean.products_clean (
                        product_id,
                        store_id,
                        original_name,
                        name_clean,
                        brand,
                        category,
                        total_quantity,
                        normalized_unit,
                        pack_count,
                        attributes,
                        needs_review,
                        review_reasons,
                        rules_version
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    product_id,
                    cleaned["store_id"],
                    cleaned["original_name"],
                    cleaned["name_clean"],
                    cleaned["brand"],
                    cleaned["category"],
                    cleaned["total_quantity"],
                    cleaned["normalized_unit"],
                    cleaned["pack_count"],
                    Jsonb(attributes),
                    cleaned["needs_review"],
                    cleaned["review_reasons"],
                    cleaned["rules_version"],
                ))

            seen = set()
            duplicates = 0
            saved_prices = 0
            eligible_count = 0

            for row in source_prices:
                product_id = int(row["product_id"])
                price = safe_number(row["price_som"])
                old_price = safe_number(row["old_price_som"])

                # Разные даты сбора сохраняются.
                key = (
                    product_id,
                    price,
                    old_price,
                    row["collected_at"],
                    row["status"],
                )

                if key in seen:
                    duplicates += 1
                    continue

                seen.add(key)
                product = products_by_id.get(product_id)

                if product is None:
                    raise ValueError(
                        f"Цена {row['id']} ссылается "
                        f"на отсутствующий товар {product_id}"
                    )

                quantity = product["total_quantity"]
                reasons = []

                if price is None or price <= 0:
                    reasons.append("invalid_price")
                elif price > MAX_PRICE:
                    reasons.append("price_above_review_threshold")

                if row["status"] != "public_page":
                    reasons.append("unverified_source_status")

                product_flags = set(
                    (product.get("review_reasons") or "").split(";")
                )

                if quantity is None or quantity <= 0:
                    reasons.append("unknown_package")

                if product.get("normalized_unit") not in {
                    "kg", "l", "pcs", "кг", "л", "шт"
                }:
                    reasons.append("unknown_unit")

                ambiguous_flags = {
                    "multiple_measures",
                    "ambiguous_pack_count",
                    "source_quantity_conflict",
                }

                if product_flags & ambiguous_flags:
                    reasons.append("ambiguous_package")

                eligible = not reasons
                price_per_unit = None

                if eligible:
                    price_per_unit = (
                        price / quantity
                    ).quantize(Decimal("0.01"))
                    eligible_count += 1

                cur.execute("""
                    INSERT INTO grocery_clean.price_history_clean (
                        history_id,
                        product_id,
                        price_som,
                        old_price_som,
                        collected_at,
                        source_status,
                        price_per_unit,
                        eligible_price,
                        review_reasons
                    )
                    VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s
                    )
                """, (
                    row["id"],
                    product_id,
                    price,
                    old_price,
                    row["collected_at"],
                    row["status"],
                    price_per_unit,
                    eligible,
                    ";".join(reasons),
                ))

                saved_prices += 1

        # Успешный выход фиксирует изменения.
        # При ошибке транзакция откатывается.

    print("\nГотово. Результаты сохранены в PostgreSQL.")
    print(json.dumps({
        "products_saved": len(products_by_id),
        "source_price_rows": len(source_prices),
        "duplicates_excluded": duplicates,
        "price_rows_saved": saved_prices,
        "eligible_price_rows": eligible_count,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()