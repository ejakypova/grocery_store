
import pandas as pd
from creat_database import save_rows

# Читаем собранные товары
df = pd.read_csv(
    "globus_combined_preview.csv",
    encoding="utf-8-sig"
)

# Убираем повторяющиеся товары
df = df.drop_duplicates(
    subset=["store", "name", "url"]
)

# Проверяем данные
df = df[
    df["name"].notna()
    & df["url"].notna()
    & (df["price_som"] > 0)
].copy()

# Преобразуем пустые значения в None
df = df.astype(object).where(pd.notna(df), None)

print("Товаров для импорта:", len(df))

# Сохраняем в PostgreSQL
result = save_rows(df.to_dict(orient="records"))

print("Результат сохранения:", result)
