"""Предсказание категории и бренда: справочник брендов -> ML.

Пример:
 python predict_grocery_updated.py "Макароны Макфа 400 г"
 python predict_grocery_updated.py --reference "Справочник бренд и категорий.xlsx" "Молоко ВМ 1 л"
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
                # Предпочитаем полное наименование, а не аббревиатуру.
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
    parser.add_argument("--models", default="grocery_models")
    parser.add_argument("--reference", default="Справочник бренд и категорий.xlsx")
    parser.add_argument("--threshold", type=float, default=0.60)
    args = parser.parse_args()

    models = {}
    for field in ("category_label", "brand_label"):
        file = Path(args.models) / f"{field}_model.joblib"
        models[field] = joblib.load(file) if file.exists() else None

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
