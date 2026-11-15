"""Predict a category and brand: python predict_grocery.py "Молоко Веселый молочник 1 л" "Сок Rich яблоко 1л"""
import argparse
from pathlib import Path
import joblib
from train_grocery_models import norm

ALIASES = {'вм': 'Веселый молочник', 'двд': 'Домик в деревне'}

def predict(model, name, threshold):
    if model is None:
        return 'Модель не обучена', 0.0, False
    prob = model.predict_proba([norm(name)])[0]
    idx = int(prob.argmax())
    confidence = float(prob[idx])
    return str(model.classes_[idx]), confidence, confidence >= threshold

def brand_alias(name):
    words = set(norm(name).split())
    for key, value in ALIASES.items():
        if key in words or norm(value) in norm(name):
            return value
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('names', nargs='*')
    ap.add_argument('--models', default='grocery_models')
    ap.add_argument('--threshold', type=float, default=0.60)
    args = ap.parse_args()
    path = Path(args.models)
    models = {}
    for field in ('category_label', 'brand_label'):
        file = path / f'{field}_model.joblib'
        models[field] = joblib.load(file) if file.exists() else None
    names = args.names or [input('Название товара: ')]
    for name in names:
        cat, cp, okc = predict(models['category_label'], name, args.threshold)
        brand = brand_alias(name)
        if brand:
            bp, okb = 1.0, True
        else:
            brand, bp, okb = predict(models['brand_label'], name, args.threshold)
        print(f'\nТовар: {name}\nКатегория: {cat} (уверенность {cp:.1%}; {"принято" if okc else "проверить"})\nБренд: {brand} (уверенность {bp:.1%}; {"принято" if okb else "проверить"})')

if __name__ == '__main__':
    main()
