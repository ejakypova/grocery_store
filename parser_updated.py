"""Сбор данных о продуктах.. Run: python parser.py"""
import argparse
import json 
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

SOURCES = {
    'Globus': ['https://globus-online.kg/ru-kg/catalog/grocery/category/b2c468e1de8c6a3c22dcfe1c65361cbd',
	    'https://globus-online.kg/ru-kg/catalog/grocery/category/774448e3d08d450aadf655a818663d39',
	    'https://globus-online.kg/ru-kg/catalog/grocery/category/c2ab3033d7d678bb05369d3fee359cde',
	    'https://globus-online.kg/ru-kg/catalog/grocery/category/e4abef89be242c706e5febd0577bb8f4',
	    'https://globus-online.kg/ru-kg/catalog/grocery/category/c9c344c6f2e6b04b6aaad15650894201',
	    'https://globus-online.kg/ru-kg/catalog/grocery/category/31e6bc34c9832d8737ae4512c6e9fc01',
	    'https://globus-online.kg/ru-kg/catalog/grocery/category/aa66e89b3bfecbb8aaaf5b5ef75db193',
	    'https://globus-online.kg/ru-kg/catalog/grocery/category/beverages_deep',
	    'https://globus-online.kg/ru-kg/catalog/grocery/category/256e5b8c85701d630618cf928f16cbb1',
	    'https://globus-online.kg/ru-kg/catalog/grocery/category/e8882715f6df46f998039a3a0b2adc17',
	    'https://globus-online.kg/ru-kg/catalog/grocery/category/National_deep',
	    'https://globus-online.kg/ru-kg/catalog/grocery/category/vegetables_deep',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/fruits_deep',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/a5efea07519a2bca35e8ca0d601dd0e4',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/906767e3aeb842f9d4920611fb84f1e8',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/6e34e245c671fafcb2750d926ca94a17',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/64702c81515df2016f5a2cc40193c360',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/2852b6623b41b97694e17de8e1df9ff9',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/sneki_deep',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/4a45de32a50daafdbff2f3d1d1aa7460',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/14492621173beb74e686cf4703534d4a',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/d64d969d4ab2e8f59096ead5f8f7f971',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/73b6b76346403c2cf47c4f1381315ffa',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/8377e4a6c2353dae316e64dfbe6690e1',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/65e0cdfe6047e97ffb004f14e16788c1',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/8648295218cc61b0922b95759db08fba',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/23c0f601a03cf77800e619d06e1db0d3',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/6578e1c428592a35ae42cd3acb988348',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/ae4c19ddff4d1eaaf8ebf55236d1fc57',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/9f9277640d92304b5c963a9f709a53c9',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/65835fcb5e66034285747db19d2795f3',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/3db8fdff16957cc91c98f392409591c0',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/dcfa8529fdbbae9c29901e2c20356138',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/49d8579b2342c3df7dfc2cd7c3c90d73',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/af22b0454ae7b2c6186c65e8981dfbc4',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/2e138c6c176423e999276fd8d8adc7e7',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/icecream_deep',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/7bfc21487df764259dca10cb81f321dd',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/f860ef056459d32981f467dd05703077',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/8e3faa2363a3b34c0ed3605626ba386c',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/9b68c9323c3b2b0b0db07a9f04db9a27',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/9d22e7097b7782b867069444e16c909a',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/63d7dd51fb640776ab745b9faac51882',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/eebf27d0a26b8e737f0f015301f7d0d5',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/01fbaadb89bd11d53b0f182fe565fa91',
        'https://globus-online.kg/ru-kg/catalog/grocery/category/7d4bb06887c72f81fd0edae94b9046a3'
        ],

    'Dostor': ['https://www.dostor.kg/product?lang=ru'],

    'Optobaza': ['https://optobaza.kg/collection/prodovolstvennye',
	        'https://optobaza.kg/collection/neprodovolstvennye'
        ],

     'Kampa': [
        'https://kampa.kg/vse-kategorii/bakaleya-bishkek/muka-i-ingredienty-dlya-vypechki-bishkek/',
	'https://kampa.kg/vse-kategorii/bakaleya-bishkek/krupy-bishkek/',
	'https://kampa.kg/vse-kategorii/bakaleya-bishkek/makaronnye-izdeliya-bishkek/',
	'https://kampa.kg/vse-kategorii/bakaleya-bishkek/rastitelnoe-maslo-bishkek/',
	'https://kampa.kg/vse-kategorii/bakaleya-bishkek/sousy-ketchup-mayonez-tomat-uksus-bishkek/',
	'https://kampa.kg/vse-kategorii/bakaleya-bishkek/chipsy-suhariki-sneki-popkorn/',
	'https://kampa.kg/vse-kategorii/bakaleya-bishkek/suhofrukty-orehi-semechki-bishkek/',
	'https://kampa.kg/vse-kategorii/bakaleya-bishkek/pripravy-zapravki-i-specii-bishkek/',
	'https://kampa.kg/vse-kategorii/bakaleya-bishkek/sahar-sol-bishkek/',
	'https://kampa.kg/vse-kategorii/gotovye-blyuda-i-kulinariya/blyuda-bystrogo-prigotovleniya/',
	'https://kampa.kg/vse-kategorii/gotovye-blyuda-i-kulinariya/gotovoe-testo/',
	'https://kampa.kg/vse-kategorii/gotovye-blyuda-i-kulinariya/suhie-zavtraki-kashi-hlopya/',
	'https://kampa.kg/vse-kategorii/gotovye-blyuda-i-kulinariya/sendvichi-buterbrody/',
	'https://kampa.kg/vse-kategorii/gotovye-blyuda-i-kulinariya/pelmeni-hinkali-manty-vareniki/',
	'https://kampa.kg/vse-kategorii/molochnye-produkty-syry-yayca/yogurty-kislomolochnye-napitki-i-kiseli/',
	'https://kampa.kg/vse-kategorii/molochnye-produkty-syry-yayca/moloko-slivki/',
	'https://kampa.kg/vse-kategorii/molochnye-produkty-syry-yayca/molochnye-deserty-pudingi-morozhnoe/',
	'https://kampa.kg/vse-kategorii/molochnye-produkty-syry-yayca/slivochnoe-maslo-margarin-spredy/',
	'https://kampa.kg/vse-kategorii/molochnye-produkty-syry-yayca/smetana-tvorog-tvorozhnye-syrki/',
	'https://kampa.kg/vse-kategorii/molochnye-produkty-syry-yayca/syry/',
	'https://kampa.kg/vse-kategorii/molochnye-produkty-syry-yayca/yayca/',
	'https://kampa.kg/vse-kategorii/voda-napitki-soki/mineralnaya-i-pitevaya-voda/',
	'https://kampa.kg/vse-kategorii/voda-napitki-soki/napitki-limonady-kvasy/',
	'https://kampa.kg/vse-kategorii/voda-napitki-soki/nacionalnye-napitki/',
	'https://kampa.kg/vse-kategorii/voda-napitki-soki/soki-nektary-morsy/',
	'https://kampa.kg/vse-kategorii/voda-napitki-soki/holodnyy-chay/',
	'https://kampa.kg/vse-kategorii/chay-kofe-kakao/kakao-goryachiy-shokolad/',
	'https://kampa.kg/vse-kategorii/chay-kofe-kakao/kofe-cikoriy/',
	'https://kampa.kg/vse-kategorii/chay-kofe-kakao/chay/',
	'https://kampa.kg/vse-kategorii/konservirovannye-produkty/med/',
	'https://kampa.kg/vse-kategorii/konservirovannye-produkty/molochnye-konservy/',
	'https://kampa.kg/vse-kategorii/konservirovannye-produkty/myasnye-konservy/',
	'https://kampa.kg/vse-kategorii/konservirovannye-produkty/ovoschnye-konservy/',
	'https://kampa.kg/vse-kategorii/konservirovannye-produkty/rybnye-konservy/',
	'https://kampa.kg/vse-kategorii/konservirovannye-produkty/konservy-fruktovye-varene-dzhem/',
	'https://kampa.kg/konservirovannye-produkty/masla/',
	'https://kampa.kg/vse-kategorii/myasnye-produkty-ptica/',
	'https://kampa.kg/vse-kategorii/ryba-i-moreprodukty/',
	'https://kampa.kg/vse-kategorii/hlebobulochnye-izdeliya/',
	'https://kampa.kg/vse-kategorii/konditerskie-izdeliya/vostochnye-sladosti/',
	'https://kampa.kg/vse-kategorii/konditerskie-izdeliya/zhevatelnaya-rezinka-osvezhayuschie-konfety/',
	'https://kampa.kg/vse-kategorii/konditerskie-izdeliya/konfety-i-shokolady/',
	'https://kampa.kg/vse-kategorii/konditerskie-izdeliya/marmelad-zefir-pastila-sufle/',
	'https://kampa.kg/vse-kategorii/konditerskie-izdeliya/muchnye-konditerskie-izdeliya/',
	'https://kampa.kg/vse-kategorii/konditerskie-izdeliya/torty-pirozhnye-vypechka/',
	'https://kampa.kg/vse-kategorii/ovoschi-frukty-zelen-griby-bishkek/frukty-i-yagody-bishkek/',
	'https://kampa.kg/vse-kategorii/ovoschi-frukty-zelen-griby-bishkek/ovoschi-zelen-salaty-bishkek/',
	'https://kampa.kg/vse-kategorii/dlya-detey/podguzniki/',
	'https://kampa.kg/vse-kategorii/dlya-detey/detskoe-pitanie/',
	'https://kampa.kg/vse-kategorii/bytovaya-himiya/sredstva-dlya-uborki-chistki/',
	'https://kampa.kg/vse-kategorii/bytovaya-himiya/chistyaschie-sredstva/',
	'https://kampa.kg/vse-kategorii/bytovaya-himiya/stiralnye-poroshki-otbelivateli/',
	'https://kampa.kg/vse-kategorii/bytovaya-himiya/dlya-mytya-posudy/',
	'https://kampa.kg/vse-kategorii/bytovaya-himiya/osvezhitel-vozduha/',
	'https://kampa.kg/vse-kategorii/lichnaya-gigiena/shampun-balzam-gel-dlya-dusha/',
	'https://kampa.kg/vse-kategorii/lichnaya-gigiena/uhod-za-polostyu-rta/',
	'https://kampa.kg/vse-kategorii/raznye-produkty/sigarety/',
	'https://kampa.kg/raznye-produkty/salfetki/'
],
}
# Отдельный подраздел молока. При изменении каталога можно передать --category-url.
GLOBUS_MILK_URL = (
    "https://globus-online.kg/ru-kg/catalog/grocery/category/"
    "9f9277640d92304b5c963a9f709a53c9/"
    "66f8ace2f4554af3b8e7c718a780413f000200010000"
)

HEADERS = ['store','name','characteristics','quantity','unit','price_som','old_price_som','url','collected_at','status']
# Пробелы разрешены только между полноценными группами тысяч.
AMOUNT = r"(?:\d{1,3}(?:[\s\u00a0\u202f]\d{3})*|\d+)(?:[.,]\d{1,2})?"
CURRENCY = r"(?:сом(?:ов|а)?|KGS|[сc]\.?)(?!\w)"
PRICE = re.compile(
    r"(?<![\w.,])(" + AMOUNT + r")\s*" + CURRENCY, re.I)
NODE_PRICE = re.compile(r"(" + AMOUNT + r")(?:\s*" + CURRENCY + r")?", re.I)


def price_from_node(node):
    """Принимает только цену целиком, без артикула и других подписей."""
    if node is None:
        return None
    raw = node.get("content") or node.get_text(" ", strip=True)
    match = NODE_PRICE.fullmatch(raw.strip())
    if match is None:
        return None
    value = number(match.group(1))
    return value if value > 0 else None

def first_node_price(card, selectors):
    for selector in selectors:
        for node in card.select(selector):
            value = price_from_node(node)
            if value is not None:
                return value
    return None

SIZE = re.compile(r'(\d+(?:[.,]\d+)?)\s*(кг|гр?|мл|л|шт)\b', re.I)

def number(raw):
    return float(re.sub(r'\s+', '', raw).replace(',', '.'))

def allowed(url):
    p = urlparse(url)
    robots = RobotFileParser()
    robots.set_url(f'{p.scheme}://{p.netloc}/robots.txt')
    try:
        robots.read()
        return robots.can_fetch('GroceryResearchBot', url)
    except Exception as exc:
        print(f'Не удалось проверить robots.txt для {url}: {exc}')
        return False

def clean(s):
    return re.sub(r'\s+', ' ', s or '').strip()

def parse_optobaza(soup, page_url):
    found = []
    cards = soup.select(".product-preview")

    print("Найдено карточек Optobaza:", len(cards))

    for card in cards:
        title = card.select_one(".product-preview__title a")
        if not title:
            continue

        name = clean(title.get_text(" ", strip=True))
        link = urljoin(page_url, title.get("href", ""))

        # Не ищем цену в общем тексте: артикул иногда склеивается с суммой.
       
        current_price = first_node_price(card, (
            ".product-preview__price-cur",
            ".product-preview__price-current",
            "[data-product-price]",
            '[itemprop="price"]',
            ".product-preview__price",
        ))
        if current_price is None:
            continue
        
        old_price = first_node_price(card, (
            ".product-preview__price-old",
            ".product-preview__old-price",
        ))    
        if old_price is not None and old_price <= current_price:
            old_price = None
        

        if name:
            found.append((
                name,
                current_price,
                old_price,
                link
            ))

    return found

def parse_kampa(soup, page_url):
    found = []
    cards = soup.select(".ty-compact-list__item")

    print("Найдено карточек Kampa:", len(cards))

    for card in cards:
        title = card.select_one("a.product-title")

        # Берём именно текущую цену, без кода товара и объёма
        price_node = card.select_one(
            '[id^="sec_discounted_price_"]'
        )

        if title is None or price_node is None:
            continue

        name = clean(title.get_text(" ", strip=True))
        href = title.get("href")

        if not name or not href:
            continue

        try:
            price = number(
                price_node.get_text("", strip=True)
            )
        except ValueError:
            continue

        old_price = None
        old_node = card.select_one(
            '[id^="sec_list_price_"]'
        )

        if old_node is not None:
            try:
                value = number(
                    old_node.get_text("", strip=True)
                )
                if value > price:
                    old_price = value
            except ValueError:
                pass

        found.append((
            name,
            price,
            old_price,
            urljoin(page_url, href)
        ))

    return found


def get_kampa_next_url(soup, current_url):
    next_link = soup.select_one(
        "a.ty-pagination__next[href]"
    )

    if next_link is None:
        return None

    return urljoin(current_url, next_link["href"])

def parse_globus(soup, page_url):
    """Извлекает товары по ссылкам /good/, избегая цен из соседних карточек.

    Селекторы являются резервными: разметка сайта может измениться.
    """
    found = []
    seen = set()
    for a in soup.select('a[href*="/good/"]'):
        href = a.get('href') or ''
        link = urljoin(page_url, href).split('?')[0]
        if link in seen:
            continue
        # Ищем ближайший небольшой контейнер товара, а не весь каталог.
        card = None
        node = a
        for _ in range(5):
            node = node.parent
            if node is None or node.name in ('body', 'html'):
                break
            if len(node.select('a[href*="/good/"]')) > 2:
                break
            if node.select_one('[class*="price"], [class*="Price"], [itemprop="price"]'):
                card = node
                break
        if card is None:
            continue
        title = card.select_one('[class*="title"], [class*="Title"], [class*="name"], [class*="Name"]')
        name = clean(title.get_text(' ', strip=True)) if title else clean(a.get_text(' ', strip=True))
        if len(name) < 5:
            continue
        current = first_node_price(card, (
            '[class*="current"][class*="price"]',
            '[class*="actual"][class*="price"]',
            '[class*="Price"]',
            '[class*="price"]',
            '[itemprop="price"]',
        ))
        if current is None:
            # Текстовое представление вида «141 сом» — только внутри карточки.
            candidates = [number(x) for x in PRICE.findall(clean(card.get_text(' ', strip=True)))]
            if not candidates:
                continue
            current = candidates[0]
        if not (0 < current <= 100000):
            continue
        old = first_node_price(card, ('[class*="old"][class*="price"]', '[class*="Old"][class*="Price"]'))
        if old is not None and old <= current:
            old = None
        found.append((name, current, old, link))
        seen.add(link)
    return found


def scroll_catalog(page, max_scrolls, wait_ms=1100):
    """Прокручивает каталог до стабилизации числа ссылок на товары."""
    stable = 0
    previous = -1
    for _ in range(max_scrolls):
        page.mouse.wheel(0, 1100)
        page.wait_for_timeout(wait_ms)
        count = page.locator('a[href*="/good/"]').count()
        if count == previous:
            stable += 1
        else:
            stable = 0
        previous = count
        if stable >= 4 and count > 0:
            break
    return previous


def get_candidates(soup, store, page_url):
    
    if store == "Kampa":
        return parse_kampa(soup, page_url)

    
    if store == "Optobaza":
        return parse_optobaza(soup, page_url)
    if store == "Globus":
        globus = parse_globus(soup, page_url)
        if globus:
            return globus
    """Use JSON-LD products first, then DOM cards with a title and price."""
    found = []
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(script.string or script.get_text())
        except (ValueError, TypeError):
            continue
        def walk(obj):
            if isinstance(obj, list):
                for x in obj: walk(x)
            elif isinstance(obj, dict):
                typ = obj.get('@type', '')
                if typ == 'Product' or (isinstance(typ, list) and 'Product' in typ):
                    offer = obj.get('offers', {})
                    if isinstance(offer, list): offer = offer[0] if offer else {}
                    if not isinstance(offer, dict): offer = {}
                    try: price = number(str(offer.get('price', '')))
                    except ValueError: price = None
                    if price is not None:
                        found.append((clean(obj.get('name')), price, None, urljoin(page_url, obj.get('url') or page_url)))
                for v in obj.values(): walk(v)
        walk(data)
    if found: return found

    # Generic fallback: intentionally conservative, avoids inventing prices.
    selectors = [
    '[data-testid*="product"]',
    '[class*="product-card"]',
    '[class*="product_card"]',
    '[class*="product-item"]',
    '[class*="product_item"]',
    '[class*="ProductCard"]',
    '[class*="productCard"]',
    'article'
]
    for selector in selectors:
        cards = soup.select(selector)
        if not cards: continue
        for card in cards:
            title_node = card.select_one('h3,h4,[class*="title"],[class*="name"]')
            name = clean(title_node.get_text(' ', strip=True)) if title_node else ''
            if len(name) < 5: continue
            text = clean(card.get_text(' ', strip=True))
            prices = PRICE.findall(text)
            if not prices: continue
            try: vals = [number(v) for v in prices]
            except ValueError: continue
            # Discount labels can duplicate prices. Lowest observed price is current only when explicitly marked.
            current = vals[0]
            old = max(vals) if len(vals) > 1 and max(vals) > current else None
            a = card.select_one('a[href]')
            link = urljoin(page_url, a.get('href')) if a else page_url
            found.append((name, current, old, link))
        if found: break
    return found

def get_optobaza_next_url(soup, current_url):
    """Находит ссылку на следующую страницу Optobaza."""

    for a in soup.select('a[href]'):
        text = clean(a.get_text(" ", strip=True))

        if text.lower() == "следующая":
            href = a.get("href")

            if href:
                return urljoin(current_url, href)

    return None

def run(stores, output, scrolls, pause, globus_milk=False, category_url=None, dry_run=False):
    rows = []
    report = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            locale="ru-RU",
            viewport={"width": 1440, "height": 900}
        )

        try:
            for store in stores:
                if category_url and store == "Globus":
                    urls_to_process = [category_url]
                elif globus_milk and store == "Globus":
                    urls_to_process = [GLOBUS_MILK_URL]
                else:
                    urls_to_process = list(SOURCES[store])
                visited = set()

                for url in urls_to_process:
                    if url in visited:
                        continue

                    visited.add(url)

                    if not allowed(url):
                        report.append({
                            "store": store,
                            "url": url,
                            "status": "robots_not_verified_or_disallowed",
                            "count": 0
                        })
                        continue

                    try:
                        response = page.goto(
                            url,
                            wait_until="domcontentloaded",
                            timeout=60000
                        )

                        page.wait_for_timeout(2500)

                        if store == "Globus":
                            product_links = scroll_catalog(page, scrolls)
                            print(f"Globus: ссылок на товары после прокрутки: {product_links}")
                        else:
                            for _ in range(scrolls):
                                page.mouse.wheel(0, 1300)
                                page.wait_for_timeout(900)

                        soup = BeautifulSoup(
                            page.content(),
                            "html.parser"
                        )

                       
                        next_url = None

                        if store == "Optobaza":
                            next_url = get_optobaza_next_url(
                                soup, page.url
                            )

                        elif store == "Kampa":
                            next_url = get_kampa_next_url(
                                soup, page.url
                            )

                        if (
                            next_url
                            and next_url not in visited
                            and next_url not in urls_to_process
                        ):
                            if (
                                urlparse(next_url).netloc
                                == urlparse(page.url).netloc
                            ):
                                urls_to_process.append(next_url)

                                print(
                                    "Добавлена следующая страница:",
                                    next_url
                                )

                        # Получение товаров
                        candidates = get_candidates(
                            soup, store, page.url
                        )

                        count = 0

                        for name, price, old, link in candidates:
                            if not name:
                                continue

                            size = SIZE.search(name)

                            rows.append({
                                "store": store,
                                "name": name,
                                "characteristics": name,
                                "quantity": (
                                    number(size.group(1))
                                    if size else None
                                ),
                                "unit": (
                                    size.group(2).lower()
                                    if size else None
                                ),
                                "price_som": price,
                                "old_price_som": old,
                                "url": link,
                                "collected_at": datetime.now().isoformat(
                                    timespec="seconds"
                                ),
                                "status": "public_page"
                            })

                            count += 1

                        report.append({
                            "store": store,
                            "url": url,
                            "status": (
                                f"http_{response.status}"
                                if response else "http_unknown"
                            ),
                            "count": count
                        })

                        print(store, count, url)

                    except Exception as exc:
                        report.append({
                            "store": store,
                            "url": url,
                            "status": str(exc)[:200],
                            "count": 0
                        })

                        print("Ошибка:", store, exc)

                    time.sleep(pause)

        finally:
            browser.close()

    # Сохранение после завершения работы браузера
    df = pd.DataFrame(
        rows,
        columns=HEADERS
    ).drop_duplicates(
        subset=["store", "name", "url"]
    )

    # CSV создаётся и при пробном запуске, чтобы проверить названия и цены.
    csv_path = Path("globus_milk_preview.csv" if globus_milk or category_url else "parser_preview.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV с результатами: {csv_path.resolve()}; строк: {len(df)}")
    if dry_run:
        print("Пробный запуск: PostgreSQL не изменён.")
    elif df.empty:
        print("Нет корректно извлечённых товаров. Запись в PostgreSQL отменена.")
    else:
        from creat_database import save_rows
        count = save_rows(df.to_dict(orient="records"))
        print(f"Передано в PostgreSQL: {count} наблюдений")

    print("\nОТЧЁТ:")
    for item in report:
        print(
            item["store"],
            item["count"],
            item["status"],
            item["url"]
        )

    
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stores', nargs='+', choices=list(SOURCES), default=list(SOURCES))
    ap.add_argument('--output', default='unused.xlsx', help='Оставлено для совместимости; данные сохраняются в PostgreSQL')
    ap.add_argument('--scrolls', type=int, default=3)
    ap.add_argument('--pause', type=float, default=2.0)
    ap.add_argument('--globus-milk', action='store_true', help='Только подраздел молока Globus')
    ap.add_argument('--category-url', help='Ссылка на конкретную категорию Globus')
    ap.add_argument('--dry-run', action='store_true', help='Сохранить CSV, но не записывать в БД')
    args = ap.parse_args()
    if (args.globus_milk or args.category_url) and args.stores != ['Globus']:
        ap.error('Для --globus-milk/--category-url укажите --stores Globus')
    run(args.stores, Path(args.output), args.scrolls, args.pause,
        globus_milk=args.globus_milk, category_url=args.category_url,
        dry_run=args.dry_run)

