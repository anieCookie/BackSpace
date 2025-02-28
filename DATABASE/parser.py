import requests
from bs4 import BeautifulSoup
import sqlite3
import os
import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sberbank-ai/sbert_large_nlu_ru')

# Список стартовых страниц
start_urls = [
    'https://ru.kinorium.com/collections/critics/131/?order=sequence&page=1&perpage=200&show_viewed=1',
    'https://ru.kinorium.com/collections/critics/131/?order=sequence&page=2&perpage=200&show_viewed=1',
    'https://ru.kinorium.com/collections/critics/131/?order=sequence&page=3&perpage=200&show_viewed=1',
    'https://ru.kinorium.com/collections/critics/131/?order=sequence&page=4&perpage=200&show_viewed=1',
    'https://ru.kinorium.com/collections/critics/131/?order=sequence&page=5&perpage=200&show_viewed=1'
]

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
    'Accept-Encoding': 'gzip, deflate, br',
    'Referer': 'https://www.google.com/',
    'Connection': 'keep-alive',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'same-origin',
    'DNT': '1'
}

listf = []

def create_database():
    conn = sqlite3.connect('movies.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS movies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            rating TEXT,
            country TEXT,
            genre TEXT,
            year TEXT,
            duration TEXT,
            director TEXT,
            actors TEXT,
            tags TEXT,
            about TEXT,
            embedding_str1 TEXT,
            embedding_str2 TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_movie(data):
    conn = sqlite3.connect('movies.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO movies (title, rating, country, genre, year, duration, director, actors, tags, about, embedding_str1, embedding_str2)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', data)
    conn.commit()
    conn.close()

def extract_table_data(soup, target_legend):
    """Извлекает данные из таблицы по указанной легенде"""
    table = soup.find('table', class_='infotable')
    if not table:
        return None

    for row in table.find_all('tr'):
        legend_cell = row.find('td', class_='legend')
        data_cell = row.find('td', class_='data')

        if legend_cell and data_cell:
            legend_text = legend_cell.get_text(strip=True)
            if legend_text.lower() == target_legend.lower():
                return data_cell.get_text(' ', strip=True).replace('\xa0', ' ')

def get_tags(soup):
    tags = set()
    containers = soup.select('div[class^="film-page__adjective-list"]')

    for container in containers:
        spans = container.find_all('span')
        for span in spans:
            tag_link = span.find('a')
            if tag_link:
                tag_text = tag_link.get_text(strip=True)
                if tag_text:
                    tags.add(tag_text)

    return ', '.join(tags) if tags else 'нет данных'

def get_actors(soup):
    actors = []
    actor_divs = soup.find_all('div', itemprop='actor')

    for div in actor_divs:
        name_span = div.find('span', class_='cast__name-wrap cast__name-wrap_cut')
        if name_span:
            actor_name = name_span.get_text(strip=True)
            actors.append(actor_name)

    return ', '.join(actors) if actors else 'нет данных'

# Создаем базу данных и таблицу
create_database()

# Собираем ссылки, сохраняя порядок
for url in start_urls:
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            movies = soup.find_all('div', class_="item headlines_type-")
            for movie in movies:
                if tag := movie.find("a", class_="filmList__item-title item__title filmList__item-title-link"):
                    link = f'https://ru.kinorium.com{tag["href"]}'
                    if link not in listf:  # Проверка на дубликаты
                        listf.append(link)
    except Exception as e:
        print(f'Ошибка при обработке {url}: {str(e)}')

# Обрабатываем каждую ссылку в исходном порядке
for n, link in enumerate(listf, 1):
    try:
        response = requests.get(link, headers=headers, timeout=10)

        soup = BeautifulSoup(response.text, 'html.parser')

        # Извлекаем легкодоступные данные
        title_tag = soup.find('h1', class_="film-page__title-text film-page__itemprop").get_text(strip=True)
        teg = soup.find('li', attrs={"itemprop": "genre"}).get('content')
        year = soup.find('span', class_="data film-page__date").get_text(strip=True)
        rating = soup.find("a", class_="noLink ratingsBlockIMDb").find('span').get_text(strip=True)
        directors = soup.find("span", class_="cast__name-wrap").get_text(strip=True)
        about_text = soup.find('section', class_='text film-page__text', attrs={'itemprop': 'description'}).get_text(strip=True)

        # Извлекаем данные из td
        duration = extract_table_data(soup, 'длительность')
        country = extract_table_data(soup, 'страна')
        budget = extract_table_data(soup, 'бюджет')
        other_title = extract_table_data(soup, 'другие названия')
        actors = get_actors(soup)
        tags = get_tags(soup)

        # Вычисляем эмбеддинг для описания и тэга
        embedding_about = model.encode(about_text)
        embedding_tag = model.encode(teg)
        # Преобразуем эмбеддинг в строку (JSON)
        embedding_str = json.dumps(embedding_about.tolist())
        embedding_str2 = json.dumps(embedding_tag.tolist())

        # Подготовка данных для вставки в БД
        movie_data = (
            title_tag,
            rating,
            country or 'нет данных',
            teg or 'нет данных',
            year[1:-1] or 'нет данных',
            duration or 'нет данных',
            directors or 'нет данных',
            actors or 'нет данных',
            tags or 'нет данных',
            about_text[8:] or 'нет данных',
            embedding_str,
            embedding_str2
        )

        # Вставка данных в БД
        insert_movie(movie_data)

        # Вывод
        print(f"{n}. {title_tag}. {rating}★")
        print(f"├── Страна: {country or 'нет данных'}")
        print(f"├── Жанр: {teg or 'нет данных'}")
        print(f"├── Год: {year[1:-1] or 'нет данных'}")
        print(f"├── Длительность: {duration or 'нет данных'}")
        print(f"├── Бюджет: {budget or 'нет данных'}")
        print(f"├── Режиссер: {directors or 'нет данных'}")
        print(f"├── Главные герои: {actors or 'нет данных'}")
        print(f"├── Тэг: {tags or 'нет данных'}")
        print(f"├── О фильме: {about_text[8:]or 'нет данных'}")
        print(f"└── Другие названия: {other_title or 'нет данных'}")
        print('-' * 60)

    except Exception as e:
        print(f'Ошибка: {link} - {str(e)}')
