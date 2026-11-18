"""Интерфейс проекта разработанный с помощью Streamlit для существующего локального скрипта basket_price_engine_v3.py.
Запуск: python -m streamlit run app.py
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
st.set_page_config(page_title='Продуктовый магазин', page_icon='🛒', layout='wide')
ENGINE_FILES = ('basket_price_engine_v5.py', 'basket_price_engine_v4.py', 'basket_price_engine_v3.py')
REQUIRED_FILES = ('category_model_optimized.joblib', 'product_matching_model.joblib',
                  'product_matching_features_v3.py')


def find_basket_dir() -> Path:
    configured = os.getenv('BASKET_DIR', '').strip().strip('"')
    if configured:
        return Path(configured).expanduser()
    candidates = [ROOT.parent / 'basket', ROOT / 'basket', ROOT.parent,
                  ROOT.parent.parent / 'basket']
    return next((x for x in candidates if x.is_dir() and
                 any((x / n).is_file() for n in ENGINE_FILES)), ROOT.parent / 'basket')


def get_engine_file(basket: Path) -> Path | None:
    return next((basket / n for n in ENGINE_FILES if (basket / n).is_file()), None)


@st.cache_resource(show_spinner='Загружаю модуль basket…')
def import_engine(engine_file_string: str, mtime_ns: int):
    file = Path(engine_file_string)
    directory = str(file.parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    name = 'basket_price_engine_external'
    spec = importlib.util.spec_from_file_location(name, file)
    if spec is None or spec.loader is None:
        raise ImportError(f'Не удалось импортировать {file}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for method in ('load_prices', 'models', 'find', 'clean'):
        if not callable(getattr(module, method, None)):
            raise AttributeError(f'В {file.name} отсутствует функция {method}()')
    return module


@st.cache_resource(show_spinner='Загружаю обученные ML-модели…')
def load_models(engine_file_string: str, engine_mtime: int, category_mtime: int, matching_mtime: int):
    engine = import_engine(engine_file_string, engine_mtime)
    return engine.models()


@st.cache_data(ttl=240, show_spinner='Загружаю цены из PostgreSQL…')
def load_catalog(engine_file_string: str, engine_mtime: int, source_mode: str):
    engine = import_engine(engine_file_string, engine_mtime)
    return engine.load_prices(source_mode)


def label_ru(label):
    return {'same': 'Одинаковый', 'alternative': 'Аналог',
            'different': 'Разный', 'review': 'На проверку', 'found': 'Найден в базе', 'not_evaluated': 'Без ML-оценки'}.get(str(label), str(label))


def predict_category(engine, category_model, query: str) -> dict:
    """One prediction for display AND for decisions about safe automatic matching."""
    normalized = engine.clean(query)
    probabilities = category_model.predict_proba([normalized])[0]
    ranked = sorted(
        [(str(name), float(prob)) for name, prob in zip(category_model.classes_, probabilities)],
        key=lambda pair: pair[1], reverse=True,
    )
    return {
        'category': str(category_model.predict([normalized])[0]),
        'confidence': ranked[0][1] if ranked else 0.0,
        'top': ranked[:5],
    }


def search_by_name_for_review(catalog: pd.DataFrame, query: str, limit: int) -> pd.DataFrame:
    """Only an aid for manual review, NOT a replacement for the trained matching model."""
    if catalog.empty or 'display_name' not in catalog:
        return pd.DataFrame()

    def tokens(value):
        text = str(value).lower().replace('ё', 'е')
        text = re.sub(r'\bвм\b', 'веселый молочник', text)
        text = re.sub(r'\bдвд\b', 'домик в деревне', text)
        # Количество/масса/объём не участвуют в запасном поиске по названию.
        text = re.sub(r'(?<!\w)\d+(?:[.,]\d+)?\s*(?:кг|г|гр|мл|л|шт|штук)\b', ' ', text)
        ignore = {'кг', 'г', 'гр', 'мл', 'л', 'шт', 'штук', 'для', 'упак', 'упаковка', 'грамм', 'литр', 'литра', 'уп', 'или'}
        return {term for term in re.findall(r'[a-zа-я0-9]+', text)
                if len(term) >= 3 and not term.isdigit() and term not in ignore}

    query_tokens = tokens(query)
    if not query_tokens:
        return pd.DataFrame()
    scores = []
    for index, value in enumerate(catalog['display_name'].fillna('')):
        candidate_tokens = tokens(value)
        common = len(query_tokens & candidate_tokens)
        if common:
            # Rank full-name matches first, partial name matches only for manual review.
            score = (common / len(query_tokens), common / max(len(candidate_tokens), 1))
            scores.append((score, index))
    if not scores:
        return pd.DataFrame()
    scores.sort(key=lambda item: item[0], reverse=True)
    selected_positions = [pos for _, pos in scores[:limit]]
    found = catalog.iloc[selected_positions].copy()
    found['safe_label'] = 'review'
    found['same_probability'] = float('nan')
    found['причина_проверки'] = 'Найдено по названию: совпадение ML не подтверждено'
    return found


def find_with_diagnostics(engine, catalog, query, category_model, matching_model,
                          *, limit: int, include_uncategorized: bool,
                          category_threshold: float, same_threshold: float):
    """Search even with uncertain category; never silently approve such candidates."""
    diagnosis = predict_category(engine, category_model, query)
    low_confidence = diagnosis['confidence'] < category_threshold
    # In low-confidence cases remove only the category GATE in the engine,
    # not the ML same-probability threshold. All results still require review.
    effective_category_threshold = 0.0 if low_confidence else category_threshold
    found = engine.find(
        catalog, query, category_model, matching_model,
        limit=limit, include_uncategorized=include_uncategorized,
        category_threshold=effective_category_threshold,
        same_threshold=same_threshold,
    )
    if not isinstance(found, pd.DataFrame):
        raise TypeError('engine.find() должен возвращать pandas.DataFrame')
    found = found.copy()
    source = 'ML'
    if found.empty:
        found = search_by_name_for_review(catalog, query, limit)
        source = 'Поиск по названию (только проверка)' if not found.empty else 'Не найдено'
    elif 'safe_label' not in found:
        found['safe_label'] = 'review'

    if not found.empty and low_confidence:
        # Preserve model estimates for diagnostics, but forbid automatic cheapest-choice.
        if source.startswith('ML') and 'safe_label' in found:
            found['ml_label_original'] = found['safe_label']
        found['safe_label'] = 'review'
        reason = f'Категория не подтверждена: уверенность {diagnosis["confidence"]:.1%}'
        previous = found['причина_проверки'].fillna('').astype(str) if 'причина_проверки' in found else pd.Series('', index=found.index)
        found['причина_проверки'] = previous.map(lambda s: f'{s}; {reason}' if s else reason)
    return found, diagnosis, source


def view_candidates(found: pd.DataFrame):
    cols = ['store',
            'display_name',
            'brand', 'category',
            'price_som',
            'safe_label',
            'ml_label_original',
            'same_probability',
#           'match_source',
            'collected_at',
            'source_status',
            'причина_проверки'
            ]
    display = found[[c for c in cols if c in found.columns]].copy()
    if 'safe_label' in display:
        display['safe_label'] = display['safe_label'].map(label_ru)
    for col in ['price_som', 'same_probability']:
        if col in display:
            display[col] = pd.to_numeric(display[col], errors='coerce').round(2)
    st.dataframe(display, width='stretch', hide_index=True,
                 column_config={
                    'store': 'Магазин',
                    'display_name': 'Товар',
                    'brand': 'Бренд',
                    'category': 'Категория',
                    'price_som': 'Цена за упаковку, сом',
                    'safe_label': 'Статус проверки',
                    'ml_label_original': 'ML-класс до проверки',
                    'same_probability': 'P(same)',
 #                  'match_source': 'Способ поиска',
                    'collected_at': 'Дата цены',
                    'source_status': 'Источник',
                    'причина_проверки': 'Ограничение'
                })




def parse_product_list(raw: str) -> list[str]:
    """Список товаров через запятую; дробная часть 3,2% остаётся внутри названия."""
    return [part.strip() for part in re.split(r'(?<!\d),|,(?!\d)', raw) if part.strip()]


def money(value) -> str:
    if value is None or pd.isna(value):
        return '—'
    return f'{float(value):,.2f}'.replace(',', ' ').replace('.', ',') + ' сом'


def valid_offers(frame: pd.DataFrame, number: int, query: str) -> pd.DataFrame:
    """Keep only rows with actual prices; do not convert sizes/units."""
    result = frame.copy()
    if 'price_som' not in result:
        result['price_som'] = float('nan')
    result['price_som'] = pd.to_numeric(result['price_som'], errors='coerce')
    result = result[result['price_som'].notna() & (result['price_som'] > 0)].copy()
    if 'safe_label' not in result:
        result['safe_label'] = 'review'
    result['safe_label'] = result['safe_label'].fillna('review').astype(str).str.lower()
    if 'ml_label_original' not in result:
        result['ml_label_original'] = ''
    if 'store' not in result:
        result['store'] = 'Не указан'
    if 'display_name' not in result:
        result['display_name'] = query
    result['позиция'] = number
    result['запрос'] = query
    return result


def eligible_by_kind(offers: pd.DataFrame, kind: str, preliminary: bool) -> pd.DataFrame:
    """Never consider plain name-search fallback a verified ML match."""
    if offers.empty:
        return offers
    mask = offers['safe_label'].eq(kind)
    if preliminary:
        mask = mask | (offers['safe_label'].eq('review') &
                       offers['ml_label_original'].fillna('').astype(str).eq(kind))
    return offers.loc[mask].sort_values('price_som').copy()


def basket_report(items, engine, catalog, category_model, matching_model,
                  category_threshold, same_threshold, include_uncategorized, limit):
    """For each position pick cheapest ML same and cheaper ML alternatives."""
    entries = []
    offers_by_item = {}
    cheaper_by_item = {}
    errors = []
    for position, query in enumerate(items, 1):
        try:
            with st.spinner(f'Ищу предложения для «{query}» ({position}/{len(items)})…'):
                candidates, diagnosis, source = find_with_diagnostics(
                    engine, catalog, query, category_model, matching_model,
                    limit=int(limit), include_uncategorized=include_uncategorized,
                    category_threshold=category_threshold, same_threshold=same_threshold,
                )
            offers = valid_offers(candidates, position, query)
        except Exception as exc:
            errors.append(f'{position}. {query}: {exc}')
            entries.append({'№': position, 'Запрос': query, 'Категория ML': '—',
                            'Уверенность': '—', 'Статус': 'Ошибка поиска',
                            'Найден в базе': '—', 'Магазин в базе': '—', 'Цена в базе, сом': None,
                            'Самый дешёвый same': '—', 'Магазин': '—',
                            'Цена same, сом': None, 'Дешёвый аналог': '—',
                            'Магазин аналога': '—', 'Цена аналога, сом': None,
                            'Экономия, сом': None, 'Выбранный товар': '—',
                            'Выбранный магазин': '—', 'Итог, сом': None,
                            'Поиск': 'Ошибка'})
            continue
        offers_by_item[position] = offers
        low_conf = diagnosis['confidence'] < category_threshold
        # Under low confidence we may show the original ML predictions as preliminary.
        # Never promote string-only fallback to a trustworthy same/alternative label.
        pre = bool(low_conf and source.startswith('ML'))
        same = eligible_by_kind(offers, 'same', pre)
        alternatives = eligible_by_kind(offers, 'alternative', pre)
        base = same.iloc[0] if not same.empty else None
        # Отдельно показываем цену найденного в PostgreSQL товара даже без ML same.
        # Не выдаём это предложение за подтверждённый ML-match.
        database_found = None
        if not offers.empty:
            direct = offers.copy()
            if 'direct_found' in direct:
                direct = direct[direct['direct_found'].fillna(False)]
            else:
                direct = direct.iloc[0:0]
            if not direct.empty:
                if 'query_match_score' in direct:
                    direct = direct.sort_values(['query_match_score', 'price_som'], ascending=[False, True])
                database_found = direct.iloc[0]
            elif source == 'Поиск по названию (только проверка)':
                database_found = offers.iloc[0]
        cheaper = (alternatives[alternatives['price_som'] < float(base['price_som'])]
                   if base is not None else alternatives.iloc[0:0])
        cheaper_by_item[position] = cheaper.head(20)
        alternate = cheaper.iloc[0] if not cheaper.empty else None
        if base is None:
            status = ('Найден в базе — ML не подтвердил' if database_found is not None
                      else 'Кандидаты на проверку' if not offers.empty else 'Нет предложений')
        elif pre:
            status = ' ML (категория не подтверждена)'
        else:
            status = 'ML same'
        entries.append({
            '№': position, 'Запрос': query, 'Категория ML': diagnosis['category'],
            'Найден в базе': str(database_found['display_name']) if database_found is not None else '—',
            'Магазин в базе': str(database_found['store']) if database_found is not None else '—',
            'Цена в базе, сом': float(database_found['price_som']) if database_found is not None else None,
            'Уверенность': f'{diagnosis["confidence"]:.1%}', 'Статус': status,
            'Самый дешёвый same': str(base['display_name']) if base is not None else '—',
            'Магазин': str(base['store']) if base is not None else '—',
            'Цена same, сом': float(base['price_som']) if base is not None else None,
            'Дешёвый аналог': str(alternate['display_name']) if alternate is not None else '—',
            'Магазин аналога': str(alternate['store']) if alternate is not None else '—',
            'Цена аналога, сом': float(alternate['price_som']) if alternate is not None else None,
            'Экономия, сом': round(float(base['price_som']) - float(alternate['price_som']), 2)
                           if alternate is not None and base is not None else None,
            'Выбранный товар': str(base['display_name']) if base is not None else '—',
            'Выбранный магазин': str(base['store']) if base is not None else '—',
            'Итог, сом': float(base['price_som']) if base is not None else None,
            'Поиск': source,
        })
    return {'items': list(items), 'entries': entries, 'offers': offers_by_item,
            'cheaper': cheaper_by_item, 'errors': errors}


def render_report(report: dict, apply_alternatives: bool, show_all: bool):
    summary = pd.DataFrame(report['entries']).copy()
    if summary.empty:
        st.warning('Список товаров пуст. Введите названия через запятую.')
        return
    for i, r in summary.iterrows():
        # An alternative can replace only an already-found same, and only if cheaper.
        if apply_alternatives and pd.notna(r['Цена аналога, сом']):
            summary.at[i, 'Выбранный товар'] = r['Дешёвый аналог']
            summary.at[i, 'Выбранный магазин'] = r['Магазин аналога']
            summary.at[i, 'Итог, сом'] = r['Цена аналога, сом']
    matched = int(summary['Цена same, сом'].notna().sum())
    count = len(summary)
    provisional = bool(summary['Статус'].astype(str).str.contains('ML (категория', regex=False).any())
    st.subheader('Итоговая корзина')
    # A list that is not fully matched must not have an allegedly complete "best basket" total.
    base_sum = float(summary['Цена same, сом'].dropna().sum())
    chosen_sum = float(summary['Итог, сом'].dropna().sum())
    potential_saving = max(base_sum - chosen_sum, 0.0)
    a, b, c, d = st.columns(4)
    found_in_db = int((summary['Цена в базе, сом'].notna() | summary['Цена same, сом'].notna()).sum())
    a.metric('Найдено в базе или ML', f'{found_in_db} из {count}')
    a_caption = 'Полная корзина' if matched == count else 'Только найденные позиции'
    b.metric('Самые дешёвые одинаковые', money(base_sum) if matched else '—')
    c.metric('С дешёвыми аналогами' if apply_alternatives else 'Выбранная корзина',
             money(chosen_sum) if matched else '—')
    d.metric('Возможная экономия', money(potential_saving) if matched and apply_alternatives else '—')
    if matched != count:
      pass
 #    st.warning(f'Расчёт неполный: {matched} из {count} позиций имеют найденный ML same. '
 #                f'Показанные суммы относятся только к найденным товарам. Полная стоимость корзины неизвестна.')
    if provisional:
        st.warning('Это демонстрационная оценка — требуется проверить товары перед покупкой.')
                   
    st.dataframe(
        summary[['№',
                 'Запрос',
  #             'Статус',
  #              'Найден в базе',
  #              'Магазин в базе',
  #              'Цена в базе, сом',
                 'Самый дешёвый same',
                 'Магазин',
                 'Цена same, сом',
                 'Дешёвый аналог',
                 'Магазин аналога',
                 'Цена аналога, сом',
                 'Экономия, сом',
  #              'Выбранный товар',
  #              'Выбранный магазин',
  #              'Итог, сом'
                 ]],
        width='stretch', hide_index=True,
        column_config={
            'Цена в базе, сом': st.column_config.NumberColumn(format='%.2f'),
            'Цена same, сом': st.column_config.NumberColumn(format='%.2f'),
            'Цена аналога, сом': st.column_config.NumberColumn(format='%.2f'),
            'Экономия, сом': st.column_config.NumberColumn(format='%.2f'),
            'Итог, сом': st.column_config.NumberColumn(format='%.2f'),
        },
    )
    if (summary['Цена в базе, сом'].notna() & summary['Цена same, сом'].isna()).any():
        pass
#       st.info('Товары найдены в базе, но не подтверждены ML как одинаковые. Их цены показаны в отдельных столбцах и не включены в расчёт итоговой корзины.')
    st.subheader('Более дешёвые аналоги')
    any_cheaper = False
    for r in summary.to_dict('records'):
        position = int(r['№'])
        cheaper = report['cheaper'].get(position, pd.DataFrame())
        if cheaper.empty:
            continue
        any_cheaper = True
        with st.expander(f'{r["Запрос"]} — найдено {len(cheaper)} вариантов дешевле', expanded=True):
            baseline = float(r['Цена same, сом'])
            display = cheaper.copy()
            display['экономия_сом'] = (baseline - display['price_som']).round(2)
            wanted = ['display_name', 'store', 'price_som', 'экономия_сом',
                      'safe_label', 'ml_label_original', 'collected_at', 'source_status']
            st.dataframe(display[[key for key in wanted if key in display]],
                         hide_index=True, width='stretch',
                         column_config={'display_name': 'Аналог', 'store': 'Магазин',
                                        'price_som': 'Цена упаковки, сом',
                                        'экономия_сом': 'Экономия, сом',
                                        'safe_label': 'Статус ML',
                                        'ml_label_original': 'Исходный ML-класс',
                                        'collected_at': 'Дата цены',
                                        'source_status': 'Источник'})
    if not any_cheaper:
        st.info('Более дешёвых ML-аналогов относительно самого дешёвого совпадения не найдено. '
                'Либо у исходных товаров нет подходящих кандидатов.')
    if show_all:
        st.subheader('Все найденные предложения')
        for r in summary.to_dict('records'):
            position = int(r['№'])
            with st.expander(f'{r["Запрос"]} · {r["Статус"]}'):
                offers = report['offers'].get(position, pd.DataFrame())
                if offers.empty:
                    st.info('Нет предложений с ценами.')
                else:
                    view_candidates(offers.sort_values('price_som'))
    if report['errors']:
        for err in report['errors']:
            st.error(err)
  #st.download_button('Скачать корзину CSV', summary.to_csv(index=False).encode('utf-8-sig'),
  #                    'produktovy_magazin_korzina.csv', 'text/csv', width='stretch')
  # detailed = [frame for frame in report['offers'].values() if not frame.empty]
  # if detailed:
  #     st.download_button('Скачать все найденные предложения CSV',
 #                         pd.concat(detailed, ignore_index=True).to_csv(index=False).encode('utf-8-sig'),
 #                         'produktovy_magazin_predlozheniya.csv', 'text/csv', width='stretch')
    
st.title('🛒 Продуктовый магазин')
st.caption('Магазины Globus, Optobaza и Kampa · поиск одинаковых товаров и более дешёвых аналогов.')

with st.sidebar:
    st.header('Подключение к проекту')
    basket_path = st.text_input('Папка basket на этом компьютере', str(find_basket_dir()))
    basket = Path(basket_path.strip().strip('"')).expanduser()
    load_dotenv(basket / '.env')  # existing project DB settings; never display passwords
    engine_file = get_engine_file(basket)
    st.caption('Приоритет: движок v4, затем v5/v3. Нужны модели .joblib и product_matching_features_v3.py.')
    st.divider()
    source_options = {'Реальные (public_page)': 'public',
                      'Демонстрационный каталог': 'demo',
                      'Все источники (для диагностики)': 'all'}
    selected_mode = st.selectbox('Какие цены использовать', list(source_options))
    mode = source_options[selected_mode]
    if mode != 'public':
        st.warning('Выбран не только публичный источник. Не выдавайте демонстрационные цены за реальные.')
    st.divider()
    category_threshold = st.slider('Порог уверенности категории', 0.0, 1.0, 0.45, 0.05)
    same_threshold = st.slider('Порог P(same)', 0.0, 1.0, 0.80, 0.05)
    include_uncategorized = st.checkbox('Дополнительно проверять товары без категории', False)
    limit = st.number_input('Максимум кандидатов на товар', 5, 500, 100, step=5)
    if st.button('Обновить цены и модели'):
        load_catalog.clear()
        load_models.clear()
        st.rerun()

if not basket.is_dir():
    st.error('Папка basket не найдена. Укажите её полный путь слева или в BASKET_DIR файла .env.')
    st.code(r'BASKET_DIR=C:\Users\<имя>\Desktop\Проект DS\basket', language='dotenv')
    st.stop()

if engine_file is None:
    st.error('Не найден basket_price_engine_v4.py или basket_price_engine_v3.py в папке basket.')
    st.info('Положите демо-приложение рядом с папкой basket или укажите путь к папке, где лежит файл движка.')
    st.stop()

missing = [name for name in REQUIRED_FILES if not (basket / name).is_file()]
if missing:
    st.error('Для работы модели не найдены файлы: ' + ', '.join(missing))
    st.caption('Нужны именно обученные модели и файл признаков из вашей рабочей папки basket.')
    st.stop()

engine_mtime = engine_file.stat().st_mtime_ns
try:
    engine = import_engine(str(engine_file), engine_mtime)
    category_model, matching_model = load_models(
        str(engine_file), engine_mtime,
        (basket / REQUIRED_FILES[0]).stat().st_mtime_ns,
        (basket / REQUIRED_FILES[1]).stat().st_mtime_ns,
    )
except Exception as e:
    st.error(f'Не удалось загрузить модели из basket: {e}')
    st.caption('Проверьте версии scikit-learn, путь к product_matching_features_v3.py и модели .joblib.')
    st.stop()

try:
    catalog = load_catalog(str(engine_file), engine_mtime, mode)
except Exception as e:
    st.error(f'Ошибка загрузки каталога через {engine_file.name}: {e}')
    st.caption('Проверьте запущенный PostgreSQL 17, существование grocery_store, доступ и .env в basket.')
    st.stop()

if catalog.empty:
    st.warning('В выбранном источнике нет товаров с подходящими ценами. Измените источник данных слева.')

with st.sidebar:
    st.success(f'Подключено: {basket.name}')
    st.caption(f'Каталог: {len(catalog):,} актуальных записей')

basket_tab, catalog_tab, ml_tab = st.tabs(
    ['🛒 Магазин',
     '🔎 Каталог',
     '🤖 ML',
#   '📊 Аналитика',
#    '⚙️ Проверка файлов'
     ])
# , stats_tab, diagnostic_tab (если хочу вывести эти разделы необходимо включить в st.tab строка 492)

with basket_tab:
    st.subheader('Найдём выгодные цены на вашу корзину')
    st.write('Введите названия продуктов **через запятую**. Приложение сравнит цены на каждый товар '
             'в магазинах и отдельно предложит более дешёвые аналоги.')
    basket_input = st.text_input(
        'Список продуктов через запятую',
        value='Молоко Белая Река, Сок Rich яблоко, Атакан котлеты',
     #   placeholder='Например: молоко, хлеб, сахар, сок Rich',
     #  help='Вводите только наименования. Например: Молоко Белая Река, Сок Rich, Атакан котлеты',
    )
    first, second = st.columns(2)
    with first:
        apply_alternatives = st.checkbox('Учитывать более дешёвые аналоги в итоговой корзине', True)
    with second:
        show_all = st.checkbox('Показывать все найденные предложения', False)
    items = parse_product_list(basket_input)
    if st.button('🛒 Найти выгодную корзину', type='primary', width='stretch'):
        if not items:
            st.warning('Введите хотя бы один товар через запятую.')
            st.session_state.pop('latest_basket_report', None)
        elif catalog.empty:
           st.warning('Каталог пуст. Проверьте PostgreSQL и источник цен в боковом меню.')
           st.session_state.pop('latest_basket_report', None)
        else:
            st.session_state['latest_basket_report'] = basket_report(
                items, engine, catalog, category_model, matching_model,
                category_threshold=category_threshold, same_threshold=same_threshold,
                include_uncategorized=include_uncategorized, limit=int(limit),
            )
            st.session_state['latest_basket_settings'] = (mode, category_threshold,
                same_threshold, include_uncategorized, int(limit), str(engine_file))
    report = st.session_state.get('latest_basket_report')
    if report:
        settings = (mode, category_threshold, same_threshold, include_uncategorized,
                    int(limit), str(engine_file))
        if report['items'] != items or st.session_state.get('latest_basket_settings') != settings:
            st.info('Список товаров или настройки изменились. Нажмите «Найти выгодную корзину» для обновления расчёта.')
        else:
            render_report(report, apply_alternatives=apply_alternatives, show_all=show_all)

with catalog_tab:
    if catalog.empty:
        st.info('Каталог пуст.')
    else:
        shops = sorted(catalog.store.dropna().astype(str).unique().tolist())
        chosen_shops = st.multiselect('Магазины', shops, default=shops)
        query_name = st.text_input('Название содержит', '')
        filtered = catalog[catalog.store.astype(str).isin(chosen_shops)]
        if query_name:
            filtered = filtered[filtered.display_name.fillna('').str.contains(query_name, case=False, regex=False)]
        st.write('Найдено:', len(filtered))
        st.dataframe(filtered.head(2000), width='stretch', hide_index=True)

with ml_tab:
    st.subheader('Определение категории и сопоставление товаров')
 #  st.caption('Результаты показывают цену за упаковку без пересчёта граммов, килограммов или литров.')
    test_name = st.text_input('Введите новый товар', 'Сок Rich яблоко 1 л')
 #  st.caption('Кнопка запускает оба шага: ML-категория → поиск кандидатов → ML-сопоставление, если кандидаты доступны.')
    if st.button('Определить категорию и найти товары', type='primary'):
        if not test_name.strip():
            st.warning('Введите название товара.')
        else:
            try:
                results, diagnosis, source = find_with_diagnostics(
                    engine, catalog, test_name.strip(), category_model, matching_model,
                    limit=int(limit), include_uncategorized=include_uncategorized,
                    category_threshold=category_threshold, same_threshold=same_threshold,
                )
  #             st.metric('Категория', diagnosis['category'])
                st.write(f'Уверенность модели: **{diagnosis["confidence"]:.1%}**')
                st.caption('Пять наиболее вероятных категорий:')
                st.dataframe(pd.DataFrame([
                    {'Категория': name, 'Вероятность': f'{prob:.1%}'}
                    for name, prob in diagnosis['top']
                ]), hide_index=True, width='stretch')
                if diagnosis['confidence'] < category_threshold:
 #                  st.warning('Уверенность категории ниже порога. Совпадения можно просмотреть, но автоматическое подтверждение отключено.')
                 st.subheader('Сопоставление товаров')
               #st.caption(f'Движок: {engine_file.name} · Модель: {type(matching_model).__name__} · Поиск: {source}')
                if results.empty:
                    st.info('Не найдены кандидаты в выбранном каталоге. Проверьте вкладку «Каталог» или поменяйте источник цен.')
                else:
                    view_candidates(results)
                    if source != 'ML':
                        st.info('Это предварительные совпадения по названию. Модель сопоставления не подтверждала эти пары.')
                    elif diagnosis['confidence'] >= category_threshold:
                        pass
   #                    st.caption('Метки same / alternative / different присваиваются относительно опорного товара в движке, а не напрямую относительно запроса.')
            except Exception as e:
                st.error(f'Ошибка поиска или ML-сопоставления: {e}')
   #st.caption('Сопоставление остаётся предварительным; размеры упаковок в приложении не пересчитываются. '
   #           'При низкой уверенности категории товары требуют проверки.')

# with stats_tab:
#     a, b, c = st.columns(3)
#     a.metric('Актуальных ценовых записей', len(catalog))
#     b.metric('Магазинов', catalog.store.nunique() if not catalog.empty else 0)
#     c.metric('Категорий', catalog.category.nunique() if not catalog.empty else 0)
#     if not catalog.empty:
#         st.subheader('Самые частые категории')
#         st.bar_chart(catalog.category.fillna('Без категории').value_counts().head(15))
#         st.subheader('Цены по магазинам: число предложений')
#         st.bar_chart(catalog.store.value_counts())
   
# with diagnostic_tab:
#     st.write('Используемый путь:', str(basket.resolve()))
#     st.write('Файл поискового движка:', engine_file.name)
#     st.dataframe(pd.DataFrame([
#         {'Файл': file, 'Найден': (basket / file).is_file()}
#         for file in (engine_file.name,) + REQUIRED_FILES + ('.env',)
#     ]), hide_index=True)
   
