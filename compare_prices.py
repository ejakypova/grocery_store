import getpass
from datetime import datetime, timezone, timedelta
from decimal import Decimal

import psycopg
from psycopg.rows import dict_row


DB_NAME = "grocery_store"

# Более старые цены не участвуют в выборе.
MAX_AGE_DAYS = 7


def money(value):
    return f"{value:,.2f}".replace(",", " ")


def choose_product(cur, query):
    # Поиск по названию без учета регистра.
    cur.execute("""
        SELECT
            p.product_id,
            p.original_name,
            p.category,
            p.total_quantity,
            p.normalized_unit,
            s.name AS store
        FROM grocery_clean.products_clean AS p
        JOIN public.stores AS s
            ON s.id = p.store_id
        WHERE p.original_name ILIKE %s
          AND p.attributes->>'product_scope' = 'food'
        ORDER BY p.original_name, s.name, p.product_id
        LIMIT 50
    """, (f"%{query}%",))

    products = cur.fetchall()

    if not products:
        print("Товары не найдены. Попробуй более короткое название.")
        return None

    print("\nНайденные товары:")

    for index, product in enumerate(products, start=1):
        quantity = product["total_quantity"]
        unit = product["normalized_unit"] or ""

        package = (
            f"{quantity} {unit}"
            if quantity is not None
            else "фасовка неизвестна"
        )

        print(
            f"{index}. {product['original_name']}\n"
            f"   {product['store']} | {package}"
        )

    if len(products) == 50:
        print("\nПоказаны первые 50 результатов. Можно уточнить поиск.")

    while True:
        answer = input("\nНомер товара или 0 для отмены: ").strip()

        if answer == "0":
            return None

        if answer.isdigit() and 1 <= int(answer) <= len(products):
            return products[int(answer) - 1]

        print("Введи номер из списка.")


def matching_products(cur, selected, allow_alternatives):
    labels = (
        ["same", "alternative"]
        if allow_alternatives
        else ["same"]
    )

    # Берем только прямые подтвержденные связи.
    # Не строим цепочки замен A -> B -> C.
    # Названия сверяем с текущим каталогом.
    cur.execute("""
        SELECT
            h.product_id_a,
            h.product_id_b,
            h.human_label
        FROM grocery_clean.product_pair_annotations AS h
        JOIN grocery_clean.products_clean AS a
            ON a.product_id = h.product_id_a
        JOIN grocery_clean.products_clean AS b
            ON b.product_id = h.product_id_b
        WHERE (
            h.product_id_a = %s
            OR h.product_id_b = %s
        )
          AND h.human_label = ANY(%s::text[])
          AND h.name_a = a.original_name
          AND h.name_b = b.original_name
          AND a.category = b.category
    """, (
        selected["product_id"],
        selected["product_id"],
        labels,
    ))

    matches = {selected["product_id"]: "selected"}

    for pair in cur.fetchall():
        other_id = (
            pair["product_id_b"]
            if pair["product_id_a"] == selected["product_id"]
            else pair["product_id_a"]
        )
        matches[other_id] = pair["human_label"]

    return matches


def get_offers(cur, matches):
    # Сначала берем последнюю запись цены каждого товара.
    # Затем проверяем ее пригодность — не подменяем
    # плохую новую цену старой подходящей.
    cur.execute("""
        SELECT
            p.product_id,
            p.original_name,
            p.total_quantity,
            p.normalized_unit,
            p.review_reasons AS product_review,
            s.name AS store,
            raw.url,
            h.price_som,
            h.collected_at,
            h.eligible_price,
            h.source_status,
            h.review_reasons AS price_review
        FROM grocery_clean.products_clean AS p
        JOIN public.stores AS s
            ON s.id = p.store_id
        LEFT JOIN public.products AS raw
            ON raw.id = p.product_id
        LEFT JOIN LATERAL (
            SELECT *
            FROM grocery_clean.price_history_clean AS ph
            WHERE ph.product_id = p.product_id
            ORDER BY
                ph.collected_at DESC,
                ph.history_id DESC
            LIMIT 1
        ) AS h ON TRUE
        WHERE p.product_id = ANY(%s::integer[])
    """, (list(matches),))

    return cur.fetchall()


def compare(cur, selected, allow_alternatives):
    matches = matching_products(
        cur, selected, allow_alternatives
    )

    if len(matches) == 1:
        print(
            "\nПодтвержденные предложения этого товара "
            "в других магазинах пока не найдены."
        )

    offers = get_offers(cur, matches)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=MAX_AGE_DAYS)

    valid = []
    rejected = []

    for offer in offers:
        reasons = []
        price = offer["price_som"]
        date = offer["collected_at"]
        quantity = offer["total_quantity"]

        if date is None:
            reasons.append("нет цены")
        else:
            if date.tzinfo is None:
                reasons.append("не определен часовой пояс даты")
            else:
                if date < cutoff:
                    reasons.append("цена устарела")
                if date > now:
                    reasons.append("дата сбора находится в будущем")

        if (
            price is None
            or not price.is_finite()
            or price <= 0
        ):
            reasons.append("некорректная цена")

        if offer["source_status"] != "public_page":
            reasons.append("источник цены не подтвержден")

        if offer["eligible_price"] is not True:
            reasons.append(
                offer["price_review"]
                or "цена или фасовка требуют проверки"
            )

        if (
            quantity is None
            or not quantity.is_finite()
            or quantity <= 0
        ):
            reasons.append("фасовка неизвестна")

        unit = offer["normalized_unit"]

        if unit not in {"кг", "л", "шт"}:
            reasons.append("неизвестная единица")

        if reasons:
            rejected.append((offer, "; ".join(dict.fromkeys(reasons))))
            continue

        offer["unit_price"] = price / quantity
        offer["match_type"] = matches[offer["product_id"]]
        valid.append(offer)

    print(f"\nВыбран товар: {selected['original_name']}")
    print(f"Актуальность цен: не старше {MAX_AGE_DAYS} дней.")

    if not valid:
        print("\nНет предложений, прошедших проверки.")
    else:
        # Сравниваем только одинаковые единицы.
        selected_unit = selected["normalized_unit"]

        comparable = [
            offer for offer in valid
            if offer["normalized_unit"] == selected_unit
        ]

        for offer in valid:
            if offer["normalized_unit"] != selected_unit:
                rejected.append((
                    offer,
                    "единица фасовки отличается от выбранного товара",
                ))

        comparable.sort(
            key=lambda offer: (
                offer["unit_price"],
                offer["price_som"],
                offer["product_id"],
            )
        )

        if not comparable:
            print("\nНет предложений с сопоставимой единицей фасовки.")
        else:
            best_price = comparable[0]["unit_price"]

            print("\nСравнение по цене за кг, литр или штуку:")

            for offer in comparable:
                is_best = offer["unit_price"] == best_price
                mark = "ВЫГОДНЕЕ ЗА ЕДИНИЦУ" if is_best else ""

                match_name = {
                    "selected": "выбранный товар",
                    "same": "одинаковый товар",
                    "alternative": "подтвержденная замена",
                }[offer["match_type"]]

                print(
                    f"\n{offer['store']} — {mark}\n"
                    f"  {offer['original_name']}\n"
                    f"  Сопоставление: {match_name}\n"
                    f"  Цена упаковки: "
                    f"{money(offer['price_som'])} сом\n"
                    f"  Фасовка: {offer['total_quantity']} "
                    f"{offer['normalized_unit']}\n"
                    f"  Цена за {offer['normalized_unit']}: "
                    f"{money(offer['unit_price'])} сом\n"
                    f"  Собрано: "
                    f"{offer['collected_at'].isoformat(timespec='minutes')}"
                )

                if offer["url"]:
                    print(f"  Ссылка: {offer['url']}")

            cheapest_package = min(
                comparable,
                key=lambda offer: offer["price_som"],
            )

            print(
                "\nМинимальная стоимость одной упаковки: "
                f"{cheapest_package['store']} — "
                f"{money(cheapest_package['price_som'])} сом."
            )

            print(
                "Стоимость доставки, минимальный заказ "
                "и наличие товара в расчет не включены."
            )

    if rejected:
        print("\nНе участвовали в сравнении:")

        for offer, reason in rejected:
            print(
                f"- {offer['store']}: "
                f"{offer['original_name']} — {reason}"
            )


def main():
    password = getpass.getpass("Пароль PostgreSQL: ")

    with psycopg.connect(
        host="localhost",
        port=5432,
        dbname=DB_NAME,
        user="postgres",
        password=password,
        row_factory=dict_row,
    ) as conn:
        # Скрипт только читает данные.
        with conn.cursor() as cur:
            cur.execute(
                "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ"
            )
            cur.execute("SET TRANSACTION READ ONLY")

            query = input(
                "Введи товар, например: сок яблоко: "
            ).strip()

            if not query:
                print("Поисковая строка пустая.")
                return

            selected = choose_product(cur, query)

            if selected is None:
                return

            answer = input(
                "\nВключать подтвержденные замены "
                "других брендов и фасовок? (да/нет): "
            ).strip().lower()

            compare(
                cur,
                selected,
                allow_alternatives=answer in {"да", "yes", "y"},
            )


if __name__ == "__main__":
    main()