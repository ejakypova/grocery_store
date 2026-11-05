"""Очистка grocery_store. Исходные таблицы не изменяются.

Без PostgreSQL (проверенный plain SQL COPY dump):
 python clean_grocery_store.py --dump grocery_store_dump.sql

Из PostgreSQL (pip install "psycopg[binary]"):
    python clean_grocery_store.py --db
    python clean_grocery_store.py --db --apply
Пароль запрашивается скрыто. --apply записывает отдельную схему grocery_clean.
Магазины читаются из public.stores: Globus, Optobaza, Kampa и новые магазины.
CSV-отчёты всегда сохраняются в --output (по умолчанию clean_report).
"""
import argparse
import csv
import getpass
import json
import re
import unicodedata
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

VERSION = '1.2'
BRANDS = {
 'Простоквашино': ['простоквашино'], 'Домик в деревне': ['домик в деревне'],
 'Чабрец': ['чабрец'], 'Шадринское': ['шадринское'], 'ВМ': ['вм'],
 'Салих': ['салих'], 'Beta': ['beta'], 'Султан': ['султан'],
 'Алтайская сказка': ['алтайская сказка'], 'Митон': ['митон'],
 'Barilla': ['barilla', 'барилла'], 'Richard': ['richard', 'ричард'],
 'Лия': ['лия'], 'Мадина': ['мадина'], 'Greenfield': ['greenfield', 'гринфилд'],
 'Пиала': ['пиала'], 'Савушкин': ['савушкин'], 'ВкусВилл': ['вкусвилл'],
 'Takemura': ['takemura'], 'Риха': ['риха'], 'Coffesso': ['coffesso'],
 'Белая река':['белая река'], 'Фруто няна':['фруто няна'], 'Топтышка':['Топтышка'],
 'Asu':['asu'], 'Шоро Байтик':['шоро байтик'], 'Шоро Legenda':['шоро legenda'],
 'Pepsi':['pepsi'], 'Salam':['salam'], 'Мадина':['мадина'], 'Ал халал':['ал халал'],
}
RULES = [
 ('Сливки', r'^сливки\b'), ('Сгущенное молоко', r'^(сгущенка|молоко\s+сгущ)'),
 ('Колбасные изделия', r'^(колбаса|колбаски|сосиски|сардельки)\b'),
 ('Мясные деликатесы', r'^деликатес\b'),
 ('Растительные напитки', r'(кокосов\w*|миндальн\w*|овсян\w*|соев\w*)\s+молоко|молоко\s+(кокосов|миндальн|овсян|соев)'),
 ('Томатная паста', r'томатн\w*\s+паста'),
 ('Макаронные изделия', r'^(макароны|лапша|спагетти|вермишель)\b'),
 ('Чай', r'^чай\b'), ('Кофе', r'^кофе\b'), ('Молоко', r'^молоко\b'),
 ('Кефир', r'^кефир\b'), ('Сметана', r'^сметана\b'), ('Творог', r'^творог\b'),
 ('Йогурт', r'^йогурт\b'), ('Масло сливочное', r'^масло\s+слив'),
 ('Масло растительное', r'^масло\s+(подсол|олив|кукуруз|растит)'),
 ('Сыр', r'^сыр\b'), ('Сахар', r'^сахар\b'), ('Мука', r'^мука\b'),
 ('Рис', r'^рис\b'), ('Гречка', r'^(греч|крупа\s+греч)'),
 ('Соки и нектары', r'^(сок|нектар|напиток сокосодержащий)\b'),
 ('Вода', r'^вода\b'), ('Яйца', r'^яйц'), ('Печенье', r'^печенье\b'),
 ('Шоколад', r'^шоколад\b'), ('Конфеты', r'^конфеты\b'),
 
]
# Более узкие правила имеют приоритет над общими.
RULES = [
 ('Непродовольственные: подгузники', r'^(подгуз|трусики.*подгуз)'),
 ('Непродовольственные: гигиена и быт', r'^(мыло|шамп|гель|пакеты|салфетки|порошок|зубн|прокладки|чистящ|моющ|средство|набор.*шамп)'),
 ('Мясные полуфабрикаты', r'^(котлет|бифштекс|фрикадель|наг[г]?ет|пельмен|манты|хинкали|фарш)'),
 ('Рыбные полуфабрикаты', r'^(рыбные|лососевые)\s+(палочки|бургеры)'),
 ('Грибы', r'^грибы\b'),
 ('Зелень', r'^(розмарин|джусай|сельдерей|укроп|шпинат|мята|руккола|листья салата|лук зеленый|петрушка)\b'),
 ('Снеки', r'^(гренки|чипсы|сухарики|попкорн)\b'),
 ('Крупы и хлопья', r'^(крупа|хлопья|пшено|овсянка|булгур|перловка)\b'),
 ('Кондитерские изделия', r'^(вафли|пряники|мармелад|зефир|карамель|батончик|кекс|рулет)\b'),
 ('Соусы', r'^(соус|кетчуп|майонез|горчица)\b'),
 ('Приправы', r'^(приправа|специи|перец|соль)\b'),
] + RULES
for _brand in ['Атахан', 'Баркад', 'Мираторг', 'Дияр', 'Аттила', 'Байказан', 'Vici',
               'Egoiste', 'Huggies', 'Pampers', 'Libero', 'Moony', 'Шик']:
    BRANDS.setdefault(_brand, [_brand.lower()])

NUM = r'\d+(?:[.,]\d+)?'
UNIT = r'(?:килограмм(?:а|ов)?|грамм(?:а|ов)?|миллилитр(?:а|ов)?|литр(?:а|ов)?|кг|мл|гр|г|л)'
MEASURE = re.compile(rf'(?<![\w.,])({NUM})\s*({UNIT})(?!\w)')
MULTI = re.compile(rf'(?<!\w)(\d+)\s*[xх×]\s*({NUM})\s*({UNIT})(?!\w)')
BAGS = re.compile(rf'(\d+)\s*(?:пакетиков|пакетика|пак|пакет|шт)\.?\s*(?:по\s*)?({NUM})\s*(г|гр)(?!\w)')

def clean_text(value):
    t = unicodedata.normalize('NFKC', value or '').lower().replace('ё', 'е')
    t = re.sub('[\u00ad\u200b\ufeff]', '', t)
    t = re.sub(r'(\d),(\d)', r'\1.\2', t)
    # Слитное написание Kampa: «раз.20гр», «молоко1л».
    t = re.sub(r'(?<=[а-яa-z])(?=\d)', ' ', t)
    t = re.sub(r'(?<!\d)\.(?=\d+\s*(?:кг|гр|г|мл|л)\b)', '. ', t)
    return re.sub(r'\s+', ' ', t).strip()

def number(value):
    try:
        n = Decimal(str(value))
        return n if n.is_finite() else None
    except (InvalidOperation, ValueError):
        return None

def unit_info(u):
    if u == 'кг' or u.startswith('килограмм'): return 'кг', Decimal(1)
    if u in ('г', 'гр') or u.startswith('грамм'): return 'кг', Decimal('0.001')
    if u == 'мл' or u.startswith('миллилитр'): return 'л', Decimal('0.001')
    return 'л', Decimal(1)

def extract_product(r):
    t = clean_text(r['name']); flags = []
    category = next((c for c, pat in RULES if re.search(pat, t)), None)
    if not category: flags.append('unknown_category')
    brands = [b for b, aliases in BRANDS.items() if any(re.search(r'(?<!\w)' + re.escape(clean_text(a)) + r'(?!\w)', t) for a in aliases)]
    brand = brands[0] if len(brands) == 1 else None
    if not brand: flags.append('unknown_or_ambiguous_brand')
    nonfood = bool(category and category.startswith('Непродовольственные:'))
    # Вес ребенка и объем пакета не являются массой/объемом товара.
    package_text = t
    if category == 'Непродовольственные: подгузники' or re.search(r'^пакеты\b', t):
        package_text = ''
    multi = MULTI.search(package_text) or BAGS.search(package_text)
    count = 1; amount = None; base = None
    if multi:
        count = int(multi[1]); u, factor = unit_info(multi[3])
        amount = number(multi[2].replace(',', '.')) * factor * count; base = u
    else:
        matches = list(MEASURE.finditer(package_text))
        if len(matches) == 1:
            m = matches[0]; base, factor = unit_info(m[2]); amount = number(m[1].replace(',', '.')) * factor
            # A trailing "1 шт" means one retail package; larger counts are ambiguous.
            counts = re.findall(r'(\d+)\s*шт\b', t)
            if any(int(v) != 1 for v in counts): flags.append('ambiguous_pack_count')
        elif len(matches) > 1: flags.append('multiple_measures')
        elif category == 'Яйца' or nonfood:
            m = re.search(r'(\d+)\s*шт\b', t)
            if m: amount = Decimal(m[1]); base = 'шт'
    if amount is None or amount <= 0:
        amount = None; base = None; flags.append('unknown_package')
    # Existing parser fields are checked, never silently treated as truth.
    old = number(r.get('quantity')); oldunit = r.get('unit')
    if old is not None and oldunit in ('г', 'гр', 'кг', 'мл', 'л') and amount is not None:
        ou, factor = unit_info(oldunit)
        if ou != base or old * factor != amount: flags.append('source_quantity_conflict')
    features = {'product_scope': 'nonfood' if nonfood else 'unclassified' if category is None else 'food'}
    if re.search(r'\bвес(?:овой)?\b', t):
        features['sold_by_weight'] = True
    if '…' in t or '...' in t:
        flags.append('truncated_name')
    if nonfood:
        # Для продуктовой модели эти позиции отбираются отдельно.
        features['exclude_from_food_model'] = True
    if category == 'Чай':
        features['tea_type'] = next((v for v in ['зеленый', 'черный'] if v in t), None)
        features['form'] = 'пакетированный' if re.search(r'\d+\s*(пак|пирамид)', t) else ('листовой' if re.search('лист|весовой', t) else None)
    if category == 'Вода':
        features['carbonation'] = 'негазированная' if re.search('негаз|без газа', t) else ('газированная' if re.search(r'газир|\bгаз\b', t) else None)
    if category == 'Макаронные изделия':
        features['shape'] = next((v for v, pat in [('ракушки', 'ракуш|conchiglie'), ('спагетти', 'спагет'), ('рожки', 'рожки'), ('лапша', '^лапша'), ('вермишель', 'вермиш'), ('перья', 'перья')] if re.search(pat, t)), None)
    fat = re.search(r'(\d+(?:\.\d+)?)\s*%', t)
    if fat: features['fat_percent'] = fat[1]
    return dict(product_id=r['id'], store_id=r['store_id'], original_name=r['name'], name_clean=t,
                brand=brand, category=category, total_quantity=amount, normalized_unit=base,
                pack_count=count, attributes=json.dumps(features, ensure_ascii=False),
                needs_review=bool(flags), review_reasons=';'.join(flags), rules_version=VERSION)

def decode_copy(v):
    if v == r'\N': return None
    # PostgreSQL text COPY backslash escaping (not SQL string escaping).
    escapes = {'b':'\b','f':'\f','n':'\n','r':'\r','t':'\t','v':'\v','\\':'\\'}
    return re.sub(r'\\([0-7]{1,3}|x[0-9a-fA-F]{1,2}|.)', lambda m: chr(int(m[1], 16)) if m[1].startswith('x') else (chr(int(m[1],8)) if m[1][0] in '01234567' else escapes.get(m[1], m[1])), v)

def read_dump(path):
    tables = {}; active = None
    with open(path, encoding='utf-8-sig') as f:
        for line in f:
            line = line.rstrip('\r\n')
            if active:
                if line == r'\.': active = None; continue
                values = line.split('\t')
                if len(values) != len(cols): raise ValueError('Неверная строка COPY')
                tables[active].append(dict(zip(cols, map(decode_copy, values))))
            else:
                m = re.fullmatch(r'COPY public\.(\w+) \((.*?)\) FROM stdin;', line)
                if m: active=m[1]; cols=m[2].split(', '); tables[active]=[]
    if active: raise ValueError('Незавершённый COPY')
    for required in ('products','price_history','stores'):
        if required not in tables: raise ValueError('Нет COPY для '+required)
    return tables

def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src=ap.add_mutually_exclusive_group(required=True); src.add_argument('--dump'); src.add_argument('--db', action='store_true')
    ap.add_argument('--apply', action='store_true'); ap.add_argument('--output', default='clean_report')
    ap.add_argument('--host', default='localhost'); ap.add_argument('--port', type=int, default=5432)
    ap.add_argument('--database', default='grocery_store'); ap.add_argument('--user', default='postgres')
    ap.add_argument('--max-price', type=Decimal, default=Decimal('100000'), help='Порог проверки, не исправление цены')
    a=ap.parse_args()
    if a.apply and not a.db: ap.error('--apply разрешён только с --db')
    if not a.max_price.is_finite() or a.max_price<=0: ap.error('Некорректный max-price')
    conn=None
    if a.db:
        import psycopg
        from psycopg.rows import dict_row
        conn=psycopg.connect(host=a.host,port=a.port,dbname=a.database,user=a.user,password=getpass.getpass('Пароль PostgreSQL: '),row_factory=dict_row)
        with conn.cursor() as cur:
            cur.execute('BEGIN ISOLATION LEVEL REPEATABLE READ')
            cur.execute('SET TRANSACTION READ ONLY')
            tables={n:cur.execute('SELECT * FROM public.'+n).fetchall() for n in ('products','price_history','stores')}
        conn.rollback()
    else: tables=read_dump(a.dump)
    stores={str(r['id']):r['name'] for r in tables['stores']}
    products=[extract_product(r) for r in tables['products']]
    for product, source in zip(products, tables['products']):
        product['store_name']=stores.get(str(product['store_id']))
        product['product_url']=source.get('url')
        if product['store_name'] is None:
            raise ValueError(f"Неизвестный store_id: {product['store_id']}")
    byid={str(r['product_id']):r for r in products}
    seen=set(); prices=[]; duplicates=0
    for r in sorted(tables['price_history'], key=lambda r:int(r['id'])):
        # NaN and NULL remain distinct in deduplication: preserve distinct source rows.
        key=tuple(str(r.get(k)) for k in ('product_id','price_som','old_price_som','collected_at','status'))
        if key in seen: duplicates+=1; continue
        seen.add(key); p=number(r['price_som']); old=number(r['old_price_som']); reasons=[]
        if p is None or p<=0: reasons.append('invalid_price')
        elif p>a.max_price: reasons.append('price_above_review_threshold')
        if r['status']!='public_page': reasons.append('unverified_source_status')
        prod=byid.get(str(r['product_id']))
        if not prod: reasons.append('orphan_product')
        q=prod['total_quantity'] if prod else None
        safe_package=prod and q and not any(v in prod['review_reasons'] for v in ['multiple_measures','ambiguous_pack_count','source_quantity_conflict'])
        if q is None: reasons.append('unknown_package')
        elif not safe_package: reasons.append('ambiguous_package')
        eligible=not reasons and bool(safe_package)
        prices.append(dict(history_id=r['id'], product_id=r['product_id'], store_name=prod['store_name'] if prod else None, price_som=p, old_price_som=old,
                           collected_at=r['collected_at'], status=r['status'], price_per_unit=(p/q).quantize(Decimal('.01')) if eligible else None,
                           eligible_price=eligible, review_reasons=';'.join(reasons)))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'products_clean.csv',products,list(products[0]) if products else ['product_id'])
    write_csv(out/'prices_clean.csv',prices,list(prices[0]) if prices else ['history_id'])
    summary={'products':len(products),'source_history_rows':len(tables['price_history']), 'duplicate_rows_excluded':duplicates,
             'clean_history_rows':len(prices),'products_needing_review':sum(r['needs_review'] for r in products),
             'eligible_price_rows':sum(r['eligible_price'] for r in prices), 'rules_version':VERSION,
             'note':'eligible_price проверяет только цену/фасовку; это НЕ подтверждение взаимозаменяемости товаров.'}
    summary['stores']=[]
    for sid, name in stores.items():
        ids={str(p['product_id']) for p in products if str(p['store_id'])==sid}
        sh=[r for r in tables['price_history'] if str(r['product_id']) in ids]
        ch=[r for r in prices if str(r['product_id']) in ids]
        summary['stores'].append(dict(store_id=sid, store_name=name, products=len(ids),
            source_price_rows=len(sh), clean_price_rows=len(ch),
            duplicates_excluded=len(sh)-len(ch),
            eligible_price_rows=sum(r['eligible_price'] for r in ch),
            source_statuses=dict(Counter(r['status'] for r in sh))))
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    if a.apply:
        # Separate versioned output; raw public tables are never updated/deleted.
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute('CREATE SCHEMA IF NOT EXISTS grocery_clean')
                cur.execute('CREATE TABLE IF NOT EXISTS grocery_clean.runs (id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, created_at timestamptz DEFAULT now(), summary jsonb NOT NULL)')
                cur.execute('CREATE TABLE IF NOT EXISTS grocery_clean.records (run_id bigint REFERENCES grocery_clean.runs(id), kind text NOT NULL, source_id bigint NOT NULL, data jsonb NOT NULL, PRIMARY KEY(run_id,kind,source_id))')
                cur.execute('INSERT INTO grocery_clean.runs(summary) VALUES (%s::jsonb) RETURNING id',(json.dumps(summary),)); run_id=cur.fetchone()['id']
                for kind,rows,idcol in [('product',products,'product_id'),('price',prices,'history_id')]:
                    cur.executemany('INSERT INTO grocery_clean.records VALUES (%s,%s,%s,%s::jsonb)',[(run_id,kind,r[idcol],json.dumps(r,ensure_ascii=False,default=str)) for r in rows])
        print('Создан отдельный результат grocery_clean, run_id:',run_id)
    if conn: conn.close()
    print(json.dumps(summary,ensure_ascii=False,indent=2)); print('Отчёты:',out.resolve())

if __name__=='__main__': main()
