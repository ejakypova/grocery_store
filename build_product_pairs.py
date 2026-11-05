import getpass
import json
import re
import unicodedata
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


DB_NAME = "grocery_store"
RULES_VERSION = "pairs_auto_v2"

# Максимум похожих кандидатов для каждого товара
# в каждом другом магазине.
TOP_K = 5

# При изменении правил увеличивай RULES_VERSION.
CATEGORIES = {
    "Чай",
    "Вода",
    "Соки и нектары",
    "Макаронные изделия",
}

FLAVORS = {
    "смородина": r"смород",
    "шиповник": r"шипов",
    "груша": r"груш",
    "клюква": r"клюкв",
    "брусника": r"брусник",
    "клубника": r"клубник",
    "малина": r"малин",
    "облепиха": r"облепих",
    "томат": r"томат",
    "морковь": r"морков",
    "тыква": r"тыкв",
    "лимон": r"лимон",
    "грейпфрут": r"грейпфрут",
    "яблоко": r"ябл",
    "вишня": r"виш",
    "апельсин": r"апельс",
    "персик": r"перс",
    "ананас": r"ананас",
    "манго": r"манго",
    "виноград": r"виноград",
    "гранат": r"гранат",
    "абрикос": r"абрикос",
    "мультифрукт": r"муль?тифрукт",
}


def normalize(value):
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[\u00ad\u200b\ufeff]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def positive_number(value):
    try:
        result = Decimal(str(value))
        if result.is_finite() and result > 0:
            return result
    except (InvalidOperation, ValueError):
        pass
    return None


def first_match(text, patterns):
    found = [
        name
        for name, pattern in patterns.items()
        if re.search(pattern, text)
    ]
    return found[0] if len(found) == 1 else None


def extract_features(product):
    text = normalize(product["original_name"])
    category = product["category"]

    features = {}

    if category == "Вода":
        if re.search(r"б/газ|б/г\b|негаз|без газа", text):
            features["carbonation"] = "still"
        elif re.search(r"газ", text):
            features["carbonation"] = "sparkling"
        else:
            features["carbonation"] = None

        features["water_type"] = first_match(text, {
            "mineral": r"минер",
            "drinking": r"стол|родник",
        })

        features["mint"] = True if "мята" in text else None

    elif category == "Чай":
        features["tea_type"] = first_match(text, {
            "green": r"зелен|\bgreen\b(?!field)",
            "black": r"черн|\bblack\b",
            "herbal": r"травян|фиточай",
        })

        features["form"] = first_match(text, {
            "bags": r"\d+\s*пак|пакетик|в пакетах|в пакетиках|пирамид|саше",
            "loose": r"лист",
            "granules": r"гранул|\bгран\.",
        })

        features["aroma"] = first_match(text, {
            "bergamot": r"бергамот|earl grey|lord grey",
            "jasmine": r"жасмин|jasmine",
            "lemon": r"лимон|lemon",
            "mint": r"мят|mojito",
            "grape": r"виноград|festive grape",
            "berries": r"ягод|berries|summer bouq",
            "mango": r"манго|mango",
        })

    elif category == "Соки и нектары":
        flavors = sorted(
            name
            for name, pattern in FLAVORS.items()
            if re.search(pattern, text)
        )
        features["flavors"] = flavors or None

        # Название «сок» иногда ошибочно используется
        # для нектара, поэтому это только подсказка.
        features["drink_type"] = (
            "nectar" if "нектар" in text
            else "juice" if re.search(r"^сок\b", text)
            else None
        )

        features["no_sugar"] = (
            True if "без сахара" in text else None
        )

        features["pulp"] = (
            True if "с мякотью" in text else None
        )

    elif category == "Макаронные изделия":
        features["shape"] = first_match(text, {
            "spaghetti": r"спагет",
            "vermicelli": r"вермиш",
            "spirals": r"спирал|пружин",
            "shells": r"ракуш|conchiglie",
            "penne": r"перья",
            "horns": r"рожки",
            "mampar": r"мампар|манпар",
            "soba": r"соба|soba",
            "udon": r"удон",
            "ptitim": r"птитим",
        })

        features["composition"] = first_match(text, {
            "buckwheat": r"греч",
            "egg": r"яич",
            "vegetable": r"свекл|шпинат|морковь",
            "wheat": r"пшенич",
        })

        features["gluten_free"] = (
            True
            if re.search(r"б/глют|без глют", text)
            else None
        )

    if category == "Вода":
        features["water_number"] = first_match(text, {
            "4": r"№\s*4\b",
            "17": r"№\s*17\b",
            "27": r"№\s*27\b",
        })
        features["purpose"] = first_match(text, {
            "children": r"детск",
            "medicinal": r"лечебн(?!о)",
            "medicinal_table": r"лечебно[ -]*столов",
        })
    return features


def classify(a, b):
    """Предварительная разметка без использования ручных меток."""
    fa = a["_features"]
    fb = b["_features"]

    # Сначала ищем явные различия.
    for key in sorted(set(fa) | set(fb)):
        va, vb = fa.get(key), fb.get(key)

        if va is not None and vb is not None and va != vb:
            return "different", f"Различается свойство: {key}"

    if a["category"] == "Вода" and (
        fa.get("water_type") == "mineral"
        or fb.get("water_type") == "mineral"
    ):
        return "unsure", "Для минеральной воды нужны состав, минерализация и назначение"

    if a["category"] == "Чай" and (
        fa.get("aroma") is None or fb.get("aroma") is None
    ):
        return "unsure", "Ароматизация чая не подтверждена; отсутствие упоминания не означает отсутствие добавок"

    # Отсутствующее свойство нельзя считать совпавшим.
    required = {
        "Вода": ["carbonation", "water_type"],
        "Чай": ["tea_type", "form"],
        "Соки и нектары": ["flavors", "drink_type"],
        "Макаронные изделия": ["shape", "composition"],
    }[a["category"]]

    missing = [
        key for key in required
        if fa.get(key) is None or fb.get(key) is None
    ]

    if missing:
        return (
            "unsure",
            "Не хватает свойств: " + ", ".join(missing),
        )

    # Одностороннее упоминание сахара, мякоти,
    # ароматизации или безглютенового состава —
    # повод для ручной проверки.
    for key in sorted(set(fa) | set(fb)):
        if (fa.get(key) is None) != (fb.get(key) is None):
            return "unsure", f"Нужно уточнить свойство: {key}"

    if a["_truncated"] or b["_truncated"]:
        return "unsure", "Название товара обрезано"

    brand_a = normalize(a["brand"])
    brand_b = normalize(b["brand"])

    quantity_a = positive_number(a["total_quantity"])
    quantity_b = positive_number(b["total_quantity"])

    package_flags = {
        "unknown_package",
        "multiple_measures",
        "ambiguous_pack_count",
        "source_quantity_conflict",
    }

    flags_a = set((a["review_reasons"] or "").split(";"))
    flags_b = set((b["review_reasons"] or "").split(";"))

    package_ok = not (
        (flags_a | flags_b) & package_flags
    )

    # Консервативное правило same.
    # Похожие названия сами по себе недостаточны.
    if (
        brand_a
        and brand_a == brand_b
        and a["_text"] == b["_text"]
        and quantity_a is not None
        and quantity_a == quantity_b
        and a["normalized_unit"] == b["normalized_unit"]
        and package_ok
    ):
        return "same", "Совпали название, бренд, фасовка и свойства"

    return (
        "alternative",
        "Совпали извлеченные свойства; требуется подтверждение",
    )


def similarity(a, b):
    return SequenceMatcher(
        None,
        a["_text"],
        b["_text"],
        autojunk=False,
    ).ratio()


def collect_pairs(products):
    """
    Для каждого товара выбирает TOP_K наиболее похожих
    товаров той же категории из каждого другого магазина.
    """
    groups = defaultdict(lambda: defaultdict(list))

    for product in products:
        groups[product["category"]][product["store_id"]].append(
            product
        )

    pairs = {}

    for category, stores in groups.items():
        print(f"Подбор пар: {category}", flush=True)

        for store_id, source_products in stores.items():
            for other_store_id, other_products in stores.items():
                if store_id == other_store_id:
                    continue

                for a in source_products:
                    ranked = sorted(
                        (
                            (similarity(a, b), b["product_id"], b)
                            for b in other_products
                        ),
                        key=lambda item: (-item[0], item[1]),
                    )[:TOP_K]

                    for score, _, b in ranked:
                        # Порядок ID исключает пары A–B / B–A.
                        left, right = sorted(
                            (a, b),
                            key=lambda product: product["product_id"],
                        )

                        key = (
                            left["product_id"],
                            right["product_id"],
                        )

                        pairs[key] = (left, right, score)

    return pairs


def main():
    if TOP_K < 1:
        raise ValueError("TOP_K должен быть больше нуля")

    password = getpass.getpass("Пароль PostgreSQL: ")

    with psycopg.connect(
        host="localhost",
        port=5432,
        dbname=DB_NAME,
        user="postgres",
        password=password,
        row_factory=dict_row,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ"
            )

            # Защита от одновременного запуска двух копий.
            cur.execute(
                "SELECT pg_advisory_xact_lock(72603143)"
            )

            cur.execute("""
                SELECT
                    p.product_id,
                    p.store_id,
                    s.name AS store_name,
                    p.original_name,
                    p.brand,
                    p.category,
                    p.total_quantity,
                    p.normalized_unit,
                    p.review_reasons
                FROM grocery_clean.products_clean AS p
                JOIN public.stores AS s
                    ON s.id = p.store_id
                WHERE p.attributes->>'product_scope' = 'food'
                ORDER BY p.product_id
            """)

            products = [
                dict(row)
                for row in cur.fetchall()
                if row["category"] in CATEGORIES
            ]

            if not products:
                raise ValueError(
                    "Нет товаров выбранных категорий. "
                    "Проверь очищенные таблицы."
                )

            for product in products:
                product["_text"] = normalize(
                    product["original_name"]
                )
                product["_features"] = extract_features(product)
                product["_truncated"] = bool(
                    re.search(
                        r"…|\.\.\.",
                        product["original_name"],
                    )
                )

            print(f"Прочитано товаров: {len(products)}")
            pairs = collect_pairs(products)

            if not pairs:
                raise ValueError(
                    "Нет пар из разных магазинов "
                    "внутри выбранных категорий."
                )

            cur.execute("""
                CREATE TABLE IF NOT EXISTS
                    grocery_clean.product_pair_candidates (
                    product_id_a BIGINT NOT NULL,
                    product_id_b BIGINT NOT NULL,
                    rules_version TEXT NOT NULL,
                    category TEXT NOT NULL,
                    store_a TEXT NOT NULL,
                    store_b TEXT NOT NULL,
                    name_a TEXT NOT NULL,
                    name_b TEXT NOT NULL,
                    similarity DOUBLE PRECISION NOT NULL,
                    auto_label TEXT NOT NULL CHECK (
                        auto_label IN (
                            'same',
                            'alternative',
                            'different',
                            'unsure'
                        )
                    ),
                    reason TEXT NOT NULL,
                    features_a JSONB NOT NULL,
                    features_b JSONB NOT NULL,
                    needs_review BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

                    PRIMARY KEY (
                        product_id_a,
                        product_id_b,
                        rules_version
                    ),

                    CHECK (product_id_a < product_id_b)
                );
            """)

            # Подтвержденные решения хранятся отдельно и не удаляются
            # при повторном построении автоматических кандидатов.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS grocery_clean.product_pair_annotations (
                    product_id_a BIGINT NOT NULL,
                    product_id_b BIGINT NOT NULL,
                    human_label TEXT NOT NULL CHECK (
                        human_label IN ('same','alternative','different','unsure')
                    ),
                    name_a TEXT NOT NULL,
                    name_b TEXT NOT NULL,
                    comment TEXT NOT NULL DEFAULT '',
                    reviewed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (product_id_a, product_id_b),
                    CHECK (product_id_a < product_id_b)
                );
            """)

            # Обновляем только автоматические кандидаты
            # текущей версии. Ручную разметку нужно хранить
            # в отдельной таблице.
            cur.execute("""
                DELETE FROM grocery_clean.product_pair_candidates
                WHERE rules_version = %s
            """, (RULES_VERSION,))

            counts = Counter()
            insert_rows = []

            for a, b, score in pairs.values():
                label, reason = classify(a, b)
                counts[label] += 1

                insert_rows.append((
                    a["product_id"],
                    b["product_id"],
                    RULES_VERSION,
                    a["category"],
                    a["store_name"],
                    b["store_name"],
                    a["original_name"],
                    b["original_name"],
                    score,
                    label,
                    reason,
                    Jsonb(a["_features"]),
                    Jsonb(b["_features"]),
                ))

            cur.executemany("""
                INSERT INTO grocery_clean.product_pair_candidates (
                    product_id_a,
                    product_id_b,
                    rules_version,
                    category,
                    store_a,
                    store_b,
                    name_a,
                    name_b,
                    similarity,
                    auto_label,
                    reason,
                    features_a,
                    features_b
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s
                )
            """, insert_rows)

            # Ручное решение применяется только к тому же снимку названий.
            cur.execute("""
                CREATE OR REPLACE VIEW grocery_clean.product_pairs_review AS
                SELECT c.*,
                    h.human_label,
                    h.comment AS human_comment,
                    h.reviewed_at,
                    CASE WHEN h.name_a = c.name_a AND h.name_b = c.name_b
                         THEN h.human_label END AS confirmed_label,
                    (h.human_label IS NOT NULL AND
                     (h.name_a <> c.name_a OR h.name_b <> c.name_b))
                     AS annotation_stale
                FROM grocery_clean.product_pair_candidates c
                LEFT JOIN grocery_clean.product_pair_annotations h
                  ON h.product_id_a = c.product_id_a
                 AND h.product_id_b = c.product_id_b;
            """)

        # При успешном завершении изменения фиксируются.
        # При ошибке вся транзакция откатывается.

    print("\nГотово. Пары сохранены в PostgreSQL.")
    print(json.dumps({
        "products": len(products),
        "pairs": len(pairs),
        "labels": dict(counts),
        "rules_version": RULES_VERSION,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()