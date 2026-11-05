"""Pilot scraper for public grocery pages. Run: python parser.py"""
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
    'Globus': [
        'https://globus-online.kg/ru-kg/catalog/technical/category/af22b0454ae7b2c6186c65e8981dfbc4',
        'https://globus-online.kg/ru-kg/catalog/technical/category/vegetables_deep?context_id=_r_73_',
        'https://globus-online.kg/ru-kg/catalog/technical/category/fruits_deep?context_id=_r_cu_',
        'https://globus-online.kg/ru-kg/catalog/technical/category/9f9277640d92304b5c963a9f709a53c9?context_id=_r_j7_',
        'https://globus-online.kg/ru-kg/catalog/technical/category/65835fcb5e66034285747db19d2795f3?context_id=_r_u8_',
        'https://globus-online.kg/ru-kg/catalog/technical/category/3db8fdff16957cc91c98f392409591c0?context_id=_r_14d_',
        'https://globus-online.kg/ru-kg/catalog/technical/category/dcfa8529fdbbae9c29901e2c20356138?context_id=_r_1am_',
        'https://globus-online.kg/ru-kg/catalog/technical/category/23c0f601a03cf77800e619d06e1db0d3?context_id=_r_1gt_',
        'https://globus-online.kg/ru-kg/catalog/technical/category/6578e1c428592a35ae42cd3acb988348?context_id=_r_1n0_',
        'https://globus-online.kg/ru-kg/catalog/technical/category/ae4c19ddff4d1eaaf8ebf55236d1fc57?context_id=_r_1t7_',
        'https://globus-online.kg/ru-kg/catalog/technical/category/8377e4a6c2353dae316e64dfbe6690e1?context_id=_r_2hl_',
        'https://globus-online.kg/ru-kg/catalog/technical/category/b2c468e1de8c6a3c22dcfe1c65361cbd?context_id=_r_2oj_',
        'https://globus-online.kg/ru-kg/catalog/technical/category/774448e3d08d450aadf655a818663d39?context_id=_r_2vv_',
        'https://globus-online.kg/ru-kg/catalog/technical/category/c2ab3033d7d678bb05369d3fee359cde?context_id=_r_375_',
        'https://globus-online.kg/ru-kg/catalog/technical/category/e4abef89be242c706e5febd0577bb8f4?context_id=_r_3ru_',
        'https://globus-online.kg/ru-kg/catalog/technical/category/c9c344c6f2e6b04b6aaad15650894201?context_id=_r_42c_',
        'https://globus-online.kg/ru-kg/catalog/technical/category/63d7dd51fb640776ab745b9faac51882?context_id=_r_4ai_',
        'https://globus-online.kg/ru-kg/catalog/technical/category/9d22e7097b7782b867069444e16c909a?context_id=_r_4hg_',
        'https://globus-online.kg/ru-kg/catalog/technical/category/f860ef056459d32981f467dd05703077?context_id=_r_4os_',
        'https://globus-online.kg/ru-kg/catalog/technical/category/31e6bc34c9832d8737ae4512c6e9fc01?context_id=_r_4vs_',
        'https://globus-online.kg/ru-kg/catalog/technical/category/aa66e89b3bfecbb8aaaf5b5ef75db193?context_id=_r_567_',
        'https://globus-online.kg/ru-kg/catalog/technical/category/beverages_deep?context_id=_r_5ci_',
        'https://globus-online.kg/ru-kg/catalog/technical/category/73b6b76346403c2cf47c4f1381315ffa?context_id=_r_5pb_',
        'https://globus-online.kg/ru-kg/catalog/technical/category/c2ab3033d7d678bb05369d3fee359cde',
        'https://globus-online.kg/ru-kg/catalog/technical/category/31e6bc34c9832d8737ae4512c6e9fc01/e9f88ca196d949779faa5177b09a92b7000200010001',
    ],
    'Dostor': ['https://www.dostor.kg/product?lang=ru'],
    'Optobaza': ['https://optobaza.kg/collection/detskoe-pitanie-smesi',
                 'https://optobaza.kg/collection/kofe-vse-vidy',
                 'https://optobaza.kg/collection/chay-vse-vidy',
                 'https://optobaza.kg/collection/slivki-kakao-tsikoriy',
                 'https://optobaza.kg/collection/shokoladnaya-pasta-varenie-dzhemy',
                 'https://optobaza.kg/collection/pechenie-pryaniki-vafli-konfety',
                 'https://optobaza.kg/collection/makarony-krupy-bystraya-gotovka',
                 'https://optobaza.kg/collection/konservy-pasty-sousy-maslo',
                 'https://optobaza.kg/collection/sneki-semechki-kukuruznye-palochki',
                 'https://optobaza.kg/collection/limonady-energetiki-soki-napitki',
                 'https://optobaza.kg/collection/vlazhnye-salfetki-bumaga',
                 'https://optobaza.kg/collection/sredstva-dlya-stirki-i-uhoda',
                 'https://optobaza.kg/collection/neprodovolstvennye'
                 ],

     'Kampa': [
        'https://kampa.kg/vse-kategorii/molochnye-produkty-syry-yayca/moloko-slivki/',
        'https://kampa.kg/vse-kategorii/voda-napitki-soki/',
        'https://kampa.kg/vse-kategorii/chay-kofe-kakao/',
        'https://kampa.kg/vse-kategorii/myasnye-produkty-ptica/',
        'https://kampa.kg/vse-kategorii/bytovaya-himiya/',
        'https://kampa.kg/vse-kategorii/dlya-detey/'
        ],
}
HEADERS = ['store','name','characteristics','quantity','unit','price_som','old_price_som','url','collected_at','status']
PRICE = re.compile(
    r'(\d[\d\s\u00a0\u202f]*(?:[.,]\d{1,2})?)\s*(?:сом|KGS|с|c.\.?(?!\w))',
    re.I
)

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

        # Проверяем цены внутри конкретной карточки
        prices = PRICE.findall(
            card.get_text(" ", strip=True)
        )

        if not prices:
            continue

        values = [number(x) for x in prices]

        # При скидке на сайте старая цена
        # обычно показана перед новой
        current_price = values[-1]
        old_price = (
            values[0]
            if len(values) > 1 and values[0] > current_price
            else None
        )

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

def get_candidates(soup, store, page_url):
    
    if store == "Kampa":
        return parse_kampa(soup, page_url)

    
    if store == "Optobaza":
        return parse_optobaza(soup, page_url)
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

def run(stores, output, scrolls, pause):
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
                            timeout=45000
                        )

                        page.wait_for_timeout(2500)

                        for _ in range(scrolls):
                            page.mouse.wheel(0, 1300)
                            page.wait_for_timeout(900)

                        soup = BeautifulSoup(
                            page.content(),
                            "html.parser"
                        )

                        # Пагинация только для Optobaza
                        # if store == "Optobaza":
                        #     next_url = get_optobaza_next_url(
                        #         soup, page.url
                        #     )

                        #     if (
                        #         next_url
                        #         and next_url not in visited
                        #         and next_url not in urls_to_process
                        #     ):
                        #         # Не переходим на другие сайты
                        #         if (
                        #             urlparse(next_url).netloc
                        #             == urlparse(url).netloc
                        #             and allowed(next_url)
                        #         ):
                        #             urls_to_process.append(next_url)

                        #             print(
                        #                 "Добавлена следующая страница:",
                        #                 next_url
                        #             )

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
                                "status": (
                                    "demo_catalog"
                                    if store == "Globus"
                                    else "public_page"
                                )
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

    from creat_database import save_rows

    count = save_rows(df.to_dict(orient="records"))

    print(f"Сохранено в PostgreSQL: {count} наблюдений")

    print("\nОТЧЁТ:")
    for item in report:
        print(
            item["store"],
            item["count"],
            item["status"],
            item["url"]
        )


    df = pd.DataFrame(rows, columns=HEADERS).drop_duplicates(subset=['store','name','url'])
    from creat_database import save_rows
    count = save_rows(df.to_dict(orient='records'))
    print(f'Сохранено в PostgreSQL: {count} наблюдений; отчёт: {report}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stores', nargs='+', choices=list(SOURCES), default=list(SOURCES))
    ap.add_argument('--output', default='unused.xlsx', help='Оставлено для совместимости; данные сохраняются в PostgreSQL')
    ap.add_argument('--scrolls', type=int, default=3)
    ap.add_argument('--pause', type=float, default=2.0)
    args = ap.parse_args()
    run(args.stores, Path(args.output), args.scrolls, args.pause)

