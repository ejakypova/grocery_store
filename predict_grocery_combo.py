"""Предсказание категории и бренда: справочник брендов -> ML.

Пример:
 python predict_grocery_combined.py "Макароны Макфа 400 г"
 python predict_grocery_combined.py --reference "Справочник бренд и категорий.xlsx" "Молоко ВМ 1 л"
"""
import argparse
import re
import unicodedata
from pathlib import Path

import joblib
from openpyxl import load_workbook

# Известные варианты, которых может не быть в текущем справочнике.
ALIASES = {
    "вм": "Веселый молочник",
    "двд": "Домик в деревне",
}


def normalize(value):
    value = unicodedata.normalize("NFKC", str(value or "")).casefold().replace("ё", "е")
    return re.sub(r"[^\w]+", " ", value, flags=re.UNICODE).strip()


def ml_predict(model, name, threshold):
    if model is None:
        return "Модель не обучена", 0.0, False
    probabilities = model.predict_proba([normalize(name)])[0]
    index = int(probabilities.argmax())
    confidence = float(probabilities[index])
    return str(model.classes_[index]), confidence, confidence >= threshold


def load_brands(reference):
    if not reference.exists():
        print(f"Предупреждение: справочник не найден: {reference}. Используются только встроенные сокращения и ML.")
        return []
    workbook = load_workbook(reference, read_only=True, data_only=True)
    try:
        aliases = {}
        for sheet in workbook.worksheets:
            for row in sheet.iter_rows(values_only=True):
                if len(row) < 2 or normalize(row[0]) != "brand" or not row[1]:
                    continue
                variants = [x.strip() for x in str(row[1]).split("/") if x.strip()]
                if not variants:
                    continue
                canonical = max(variants, key=lambda x: len(normalize(x)))
                for variant in variants:
                    aliases[normalize(variant)] = canonical
        return list(aliases.items())
    finally:
        workbook.close()


def dictionary_brand(name, entries):
    normalized_name = normalize(name)
    words = normalized_name.split()
    hits = []
    for alias, canonical in entries:
        tokens = alias.split()
        if not tokens:
            continue
        # Сравниваем целые слова/фразы, чтобы 'вм' не совпадало с частью другого слова.
        for i in range(len(words) - len(tokens) + 1):
            if words[i:i + len(tokens)] == tokens:
                hits.append((len(tokens), len(alias), canonical))
                break
    if not hits:
        return None, "не найден"
    hits.sort(reverse=True)
    best = hits[0]
    # При нескольких разных подходящих брендах не выбираем произвольно.
    tied = {h[2] for h in hits if h[:2] == best[:2]}
    if len(tied) > 1:
        return None, "неоднозначное совпадение"
    return best[2], "справочник"


def main():
    parser = argparse.ArgumentParser(description="Определение категории и бренда товара")
    parser.add_argument("names", nargs="*")
    parser.add_argument("--models", default="grocery_models", help="Папка прежней модели брендов")
    parser.add_argument("--category-model", default="category_optimization/category_model_optimized.joblib", help="Оптимизированная модель категорий")
    parser.add_argument("--reference", default="Справочник бренд и категорий.xlsx")
    parser.add_argument("--threshold", type=float, default=0.60)
    args = parser.parse_args()

    category_file = Path(args.category_model)
    brand_file = Path(args.models) / "brand_label_model.joblib"
    if not category_file.is_file():
        parser.error(f"Не найдена оптимизированная модель категорий: {category_file}. Сначала запустите train_category_optimized.py")
    models = {
        "category_label": joblib.load(category_file),
        "brand_label": joblib.load(brand_file) if brand_file.is_file() else None,
    }
    print(f"Модель категорий: {category_file}")
    print(f"Модель брендов: {brand_file if brand_file.is_file() else 'не найдена'}")

    entries = load_brands(Path(args.reference))
    for alias, canonical in ALIASES.items():
        if not any(a == normalize(alias) for a, _ in entries):
            entries.append((normalize(alias), canonical))

    for name in (args.names or [input("Название товара: ")]):
        category, cat_conf, cat_ok = ml_predict(models["category_label"], name, args.threshold)
        brand, source = dictionary_brand(name, entries)
        if brand:
            brand_text = f"{brand} (по справочнику; принято)"
        else:
            brand, confidence, brand_ok = ml_predict(models["brand_label"], name, args.threshold)
            brand_text = f"{brand} (ML: {confidence:.1%}; {'принято' if brand_ok else 'проверить'})"
            if source == "неоднозначное совпадение":
                brand_text += " [в справочнике неоднозначность]"
        print(f"\nТовар: {name}")
        print(f"Категория: {category} (ML: {cat_conf:.1%}; {'принято' if cat_ok else 'проверить'})")
        print(f"Бренд: {brand_text}")


if __name__ == "__main__":
    main()
