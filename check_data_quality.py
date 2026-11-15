
import pandas as pd

# Загружаем очищенные товары
df = pd.read_csv("clean_report/products_clean.csv")

print("=== ОБЩАЯ ИНФОРМАЦИЯ ===")
print("Всего товаров:", len(df))

# Проверяем пропуски
columns = [
    "category",
    "brand",
    "total_quantity",
    "normalized_unit"
]

print("\n=== ПРОПУСКИ ===")

for col in columns:
    missing = df[col].isna().sum()
    percent = df[col].isna().mean() * 100

    print(f"{col}: {missing} ({percent:.1f}%)")

# Распределение товаров по категориям
print("\n=== КАТЕГОРИИ ===")

categories = (
    df["category"]
    .value_counts(dropna=False)
    .reset_index()
)

categories.columns = ["category", "count"]

print(categories.to_string(index=False))

# Проверка по магазинам
print("\n=== КАТЕГОРИИ ПО МАГАЗИНАМ ===")

by_store = (
    df.groupby("store_name")
    .agg(
        total_products=("product_id", "count"),
        with_category=("category", "count")
    )
)

by_store["without_category"] = (
    by_store["total_products"] -
    by_store["with_category"]
)

print(by_store.to_string())

# Причины ручной проверки
print("\n=== ПРИЧИНЫ ПРОВЕРКИ ===")

reasons = (
    df["review_reasons"]
    .dropna()
    .astype(str)
    .str.split(";")
    .explode()
    .str.strip()
)

reasons = reasons[reasons != ""]

print(reasons.value_counts().to_string())

# Примеры товаров без категории
print("\n=== ПРИМЕРЫ ТОВАРОВ БЕЗ КАТЕГОРИИ ===")

without_category = df[df["category"].isna()]

print(
    without_category[
        ["product_id", "store_name", "original_name"]
    ].head(20).to_string(index=False)
)

# Сохраняем результаты
categories.to_csv(
    "clean_report/category_statistics.csv",
    index=False,
    encoding="utf-8-sig"
)

without_category.to_csv(
    "clean_report/products_without_category.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nАнализ завершён.")
print("Отчёты сохранены в clean_report.")
