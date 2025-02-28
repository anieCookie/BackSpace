from aiogram import Router, F
from aiogram import types
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
from aiogram.enums import ChatAction
from Keyboards import main_inlines_kb, st, recommend, first_button_kb
from aiogram.fsm.state import State, StatesGroup
from sentence_transformers import SentenceTransformer
import sqlite3
import os
import json
import aiosqlite
import numpy as np
from aiogram.utils.keyboard import InlineKeyboardBuilder

DB_PATH = "D:\kinobot\pythonProject\DB\movies.db"
USER_PATH = "user.db"
REC_PATH = "rec.db"

global user_id
q = " "

semantic_model = SentenceTransformer("sberbank-ai/sbert_large_nlu_ru")
user_router = Router()


def create_db():
    conn = sqlite3.connect(USER_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_movies (
        id INTEGER PRIMARY KEY AUTOINCREMENT, 
        user_id INTEGER NOT NULL, 
        title TEXT NOT NULL, 
        rating TEXT NOT NULL,
        genre TEXT NOT NULL,
        watched INTEGER DEFAULT 0, 
        mark INTEGER DEFAULT 0,
        UNIQUE (user_id, title)
    );
    """)

    conn.commit()
    conn.close()

create_db()

def do_db():
    conn = sqlite3.connect(REC_PATH)
    cursor = conn.cursor()

    cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                user_id TEXT NOT NULL, 
                q TEXT NOT NULL, 
                UNIQUE (user_id)
            );
            """)

    conn.commit()
    conn.close()

do_db()


async def set_processing(state: FSMContext, is_processing: bool):
    await state.update_data(is_processing=is_processing)

async def is_processing(state: FSMContext):
    data = await state.get_data()
    return data.get("is_processing", False)

class Work(StatesGroup):
    wait = State()
    process = State()
    mark = State()



@user_router.message(CommandStart())
async def cmd_start(message: types.Message):
    await message.answer(
        f"Привет, <b>{message.from_user.first_name}</b>!\nВы попали на бота команды <b>BackSpace</b> 🌌\n Я уверен, вам очень понравится)",
        reply_markup=first_button_kb, parse_mode='HTML')

@user_router.message(F.text == "Меню" )
async def cmd_next(message: types.Message):
    await message.answer("Вот, что я знаю про фильмы",
                         reply_markup=main_inlines_kb)

@user_router.message(Command("menu"))
async def cmd_next(message: types.Message):
    await message.answer("Вот, что я знаю про фильмы",
                         reply_markup=main_inlines_kb)

@user_router.message(Command("des"))
async def cmd_search_movie(message: types.Message, state: FSMContext):
    await message.answer("Опишите фильм, который хотите найти ")
    await state.set_state(Work.wait)

@user_router.callback_query(F.data == "recommend")
async def cmd_o(callback: types.CallbackQuery):
    await callback.message.answer("Вот, что я могу вам предложить",
                         reply_markup=recommend)

@user_router.callback_query(F.data == "stat")
async def cmd_o(callback: types.CallbackQuery):
    await callback.message.answer("Вот, что я могу вам показать", reply_markup=st)

import aiosqlite

@user_router.callback_query(F.data == "pr")
async def show_user_movies(callback: types.CallbackQuery):
    user_id = callback.from_user.id  # Получаем ID пользователя

    async with aiosqlite.connect(USER_PATH) as db:
        query = """
        SELECT title, rating, genre FROM user_movies 
        WHERE user_id = ?
        """  # Добавили фильтр по user_id
        async with db.execute(query, (user_id,)) as cursor:
            movies = await cursor.fetchall()

    if movies:
        result = "\n\n".join(
            f"🎬 <b>{title}</b>\n⭐ {rating}\n📌 {genre}"
            for title, rating, genre in movies
        )
    else:
        result = "В базе данных пока нет оценённых фильмов."

    keyboard = InlineKeyboardBuilder()
    keyboard.button(text="Вернуться обратно", callback_data="back52")
    keyboard.adjust(1)

    sent_message = await callback.message.answer(result, parse_mode="HTML", reply_markup=keyboard.as_markup())

    @user_router.callback_query(F.data == "back52")
    async def cmd_back(callback: types.CallbackQuery):
        await callback.bot.delete_message(chat_id=callback.from_user.id, message_id=sent_message.message_id)

@user_router.callback_query(F.data == "oc")
async def cmd_o(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    async with aiosqlite.connect(USER_PATH) as db:
        query = """
                SELECT title, rating, genre FROM user_movies 
                WHERE mark != 0 AND user_id = ?
                """
        async with db.execute(query, (user_id,)) as cursor:
            movies = await cursor.fetchall()

    if movies:
        result = "\n\n".join(
            f"🎬 <b>{title}</b>\n⭐ {mark} - Ваша оценка\n📌 {genre}"
            for title, mark, genre in movies
        )
    else:
        result = "В базе данных пока нет фильмов."

    keyboard = InlineKeyboardBuilder()
    keyboard.button(text="Вернуться обратно", callback_data="back42")
    keyboard.adjust(1)

    sent_message = await callback.message.answer(result, parse_mode="HTML", reply_markup=keyboard.as_markup())

    @user_router.callback_query(F.data == "back42")
    async def cmd_back(callback: types.CallbackQuery):
        await callback.bot.delete_message(chat_id=callback.from_user.id, message_id=sent_message.message_id)

@user_router.callback_query(F.data == "movie")
async def cmd_search_movie(callback: types.CallbackQuery, state: FSMContext):
    await callback.message.answer("Опишите фильм, который хотите найти ")
    await state.set_state(Work.wait)

@user_router.callback_query(F.data == "add_tag")
async def cmd_search_movie(callback: types.CallbackQuery, state: FSMContext):
    await callback.message.answer("Напишите теги, по которым вы хотите найти фильм")
    await state.set_state(Work.wait)

@user_router.callback_query(F.data == "rec1")
async def show_recommendations(callback: types.CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id

    conn = sqlite3.connect(REC_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT q FROM recommendations WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()

    conn.close()

    if not (result and result[0].strip()):
        await callback.message.answer(
            "У вас пока нет сохранённых предпочтений. Воспользуйтесь функцией «Задать рекомендации ✍️»",
            reply_markup=first_button_kb
        )
        return

    results = description(result)
    # Фильм 1
    title1, rating1, country1, genre1, year1, duration1, director1, actors1, tags1, about1 = results[0]
    response1 = f"🎬 <b>{title1}</b>\n⭐ {rating1}\n📌 {genre1}\n📖 {about1}"
    keyboard1 = InlineKeyboardBuilder()
    keyboard1.button(text="➕ В рекомендации", callback_data="ar1")
    keyboard1.button(text="✅ Просмотрено", callback_data="w1")
    keyboard1.button(text="⭐ Оценить", callback_data="rate1")
    keyboard1.button(text="📽️ Подробнее о фильме", callback_data="info01")
    keyboard1.adjust(1)
    await callback.message.answer(response1, parse_mode="HTML", reply_markup=keyboard1.as_markup())

    @user_router.callback_query(F.data == "ar1")
    async def add_to_recommendations(callback: types.CallbackQuery, state: FSMContext):
        user_id = callback.from_user.id
        tags = tags1

        conn = sqlite3.connect(REC_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT q FROM recommendations WHERE user_id = ?", (user_id,))
        existing_entry = cursor.fetchone()

        if existing_entry:
            updated_rec = existing_entry[0] + ", " + tags
            cursor.execute("UPDATE recommendations SET q = ? WHERE user_id = ?", (updated_rec, user_id))
        else:
            cursor.execute("INSERT INTO recommendations (user_id, q) VALUES (?, ?)", (user_id, tags))

        conn.commit()
        conn.close()

        await callback.answer("Фильм добавлен в список рекомендаций")

    @user_router.callback_query(F.data.startswith("rate1"))
    async def rate_callback(callback: types.CallbackQuery, state: FSMContext):
        await state.update_data(rating_title=title1, rating_g=genre1, rating_r=rating1)

        await callback.message.answer(f"Оцените фильм «{title1}» от 1 до 10")
        await state.set_state(Work.mark)

    @user_router.callback_query(F.data == "w1")
    async def handle_callback(callback: types.CallbackQuery):
        user_id = callback.from_user.id

        conn = sqlite3.connect(USER_PATH)
        cursor = conn.cursor()

        try:
            # Добавляем фильм (если такого ещё нет)
            cursor.execute("""
                    INSERT OR IGNORE INTO user_movies (user_id, title, watched, mark, rating, genre)
                    VALUES (?, ?, 1, 0, ?, ?)
                """, (user_id, title1, rating1, genre1))

            conn.commit()

            # Выбираем все записи из таблицы user_movies
            cursor.execute("SELECT * FROM user_movies")
            rows = cursor.fetchall()

            # Печатаем все строки
            for row in rows:
                print(row)

        except sqlite3.Error as e:
            print(f"Ошибка при добавлении фильма: {e}")

        finally:
            conn.close()

        await callback.answer("Фильм сохранён в список просмотренных! 🎬")

    @user_router.callback_query(F.data == "info1")
    async def cmd_start_ai(callback: types.CallbackQuery):
        response1 = f"🎬 <b>{title1}</b>\n⭐ {rating1}\n\n📌 {genre1}, {country1}, {year1} год\n\nℹ️ {duration1}\n\n🧑‍ {director1}\n🎭 {actors1}\n\n📖 {about1}"

        keyboardr1 = InlineKeyboardBuilder()
        keyboardr1.button(text="Вернуться обратно", callback_data="backr1")
        keyboardr1.adjust(1)
        sent_message = await callback.message.answer(response1, parse_mode="HTML", reply_markup=keyboardr1.as_markup())

        @user_router.callback_query(F.data == "backr1")
        async def cmd_back(callback: types.CallbackQuery):
            await callback.bot.delete_message(chat_id=callback.from_user.id, message_id=sent_message.message_id)

    # Фильм 2
    title2, rating2, country2, genre2, year2, duration2, director2, actors2, tags2, about2 = results[1]
    response2 = f"🎬 <b>{title2}</b>\n⭐ {rating2}\n📌 {genre2}\n📖 {about2}"
    keyboard2 = InlineKeyboardBuilder()
    keyboard2.button(text="➕ В рекомендации", callback_data=f"ar2")
    keyboard2.button(text="✅ Просмотрено", callback_data=f"w2")
    keyboard2.button(text="⭐ Оценить", callback_data=f"rate2")
    keyboard2.button(text="📽️ Подробнее о фильме", callback_data="info2")
    keyboard2.adjust(1)
    await callback.message.answer(response2, parse_mode="HTML", reply_markup=keyboard2.as_markup())

    @user_router.callback_query(F.data == "ar2")
    async def add_to_recommendations(callback: types.CallbackQuery, state: FSMContext):
        user_id = callback.from_user.id
        tags = tags2

        conn = sqlite3.connect(REC_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT q FROM recommendations WHERE user_id = ?", (user_id,))
        existing_entry = cursor.fetchone()

        if existing_entry:
            updated_rec = existing_entry[0] + ", " + tags
            cursor.execute("UPDATE recommendations SET q = ? WHERE user_id = ?", (updated_rec, user_id))
        else:
            cursor.execute("INSERT INTO recommendations (user_id, q) VALUES (?, ?)", (user_id, tags))

        conn.commit()
        conn.close()

        await callback.answer("Фильм добавлен в список рекомендаций")


    @user_router.callback_query(F.data.startswith("rate2"))
    async def rate_callback(callback: types.CallbackQuery, state: FSMContext):
        await state.update_data(rating_title=title2, rating_g=genre2, rating_r=rating2)

        await callback.message.answer(f"Оцените фильм «{title2}» от 1 до 10")
        await state.set_state(Work.mark)

    @user_router.callback_query(F.data == "w2")
    async def handle_callback(callback: types.CallbackQuery):
        user_id = callback.from_user.id

        conn = sqlite3.connect(USER_PATH)
        cursor = conn.cursor()

        try:
            # Добавляем фильм (если такого ещё нет)
            cursor.execute("""
                    INSERT OR IGNORE INTO user_movies (user_id, title, watched, mark, rating, genre)
                    VALUES (?, ?, 1, 0, ?, ?)
                """, (user_id, title2, rating2, genre2))

            conn.commit()

            # Выбираем все записи из таблицы user_movies
            cursor.execute("SELECT * FROM user_movies")
            rows = cursor.fetchall()

            # Печатаем все строки
            for row in rows:
                print(row)  # Каждая строка будет кортежем

        except sqlite3.Error as e:
            print(f"Ошибка при добавлении фильма: {e}")

        finally:
            conn.close()

        await callback.answer("Фильм сохранён в список просмотренных! 🎬")

    @user_router.callback_query(F.data == "info2")
    async def cmd_start_ai(callback: types.CallbackQuery):
        response2 = f"🎬 <b>{title2}</b>\n⭐ {rating2}\n\n📌 {genre2}, {country2}, {year2} год\n\nℹ️ {duration2}\n\n🧑‍ {director2}\n🎭 {actors2}\n\n📖 {about2}"

        keyboardr2 = InlineKeyboardBuilder()
        keyboardr2.button(text="Вернуться обратно", callback_data="backr2")
        keyboardr2.adjust(1)
        sent_message = await callback.message.answer(response2, parse_mode="HTML", reply_markup=keyboardr2.as_markup())

        @user_router.callback_query(F.data == "backr2")
        async def cmd_back(callback: types.CallbackQuery):
            await callback.bot.delete_message(chat_id=callback.from_user.id, message_id=sent_message.message_id)

    # Фильм 3
    title3, rating3, country3, genre3, year3, duration3, director3, actors3, tags3, about3 = results[2]
    response3 = f"🎬 <b>{title3}</b>\n⭐ {rating3}\n📌 {genre3}\n📖 {about3}"
    keyboard3 = InlineKeyboardBuilder()
    keyboard3.button(text="➕ В рекомендации", callback_data=f"ar3")
    keyboard3.button(text="✅ Просмотрено", callback_data=f"w3")
    keyboard3.button(text="⭐ Оценить", callback_data=f"rate3")
    keyboard3.button(text="📽️ Подробнее о фильме", callback_data="info3")
    keyboard3.adjust(1)
    await callback.message.answer(response3, parse_mode="HTML", reply_markup=keyboard3.as_markup())

    @user_router.callback_query(F.data == "ar3")
    async def add_to_recommendations(callback: types.CallbackQuery, state: FSMContext):
        user_id = callback.from_user.id
        tags = tags3

        conn = sqlite3.connect(REC_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT q FROM recommendations WHERE user_id = ?", (user_id,))
        existing_entry = cursor.fetchone()

        if existing_entry:
            updated_rec = existing_entry[0] + ", " + tags
            cursor.execute("UPDATE recommendations SET q = ? WHERE user_id = ?", (updated_rec, user_id))
        else:
            cursor.execute("INSERT INTO recommendations (user_id, q) VALUES (?, ?)", (user_id, tags))

        conn.commit()
        conn.close()

        await callback.answer("Фильм добавлен в список рекомендаций")

    @user_router.callback_query(F.data.startswith("rate3"))
    async def rate_callback(callback: types.CallbackQuery, state: FSMContext):
        await state.update_data(rating_title=title3, rating_g=genre3, rating_r=rating3)
        await callback.message.answer(f"Оцените фильм «{title3}» от 1 до 10")
        await state.set_state(Work.mark)

    @user_router.callback_query(F.data == "w3")
    async def handle_callback(callback: types.CallbackQuery):
        user_id = callback.from_user.id

        conn = sqlite3.connect(USER_PATH)
        cursor = conn.cursor()

        try:
            # Добавляем фильм (если такого ещё нет)
            cursor.execute("""
                    INSERT OR IGNORE INTO user_movies (user_id, title, watched, mark, rating, genre)
                    VALUES (?, ?, 1, 0, ?, ?)
                """, (user_id, title3, rating3, genre3))

            conn.commit()

            # Выбираем все записи из таблицы user_movies
            cursor.execute("SELECT * FROM user_movies")
            rows = cursor.fetchall()

            # Печатаем все строки
            for row in rows:
                print(row)  # Каждая строка будет кортежем

        except sqlite3.Error as e:
            print(f"Ошибка при добавлении фильма: {e}")

        finally:
            conn.close()

        await callback.answer("Фильм сохранён в список просмотренных! 🎬")

    @user_router.callback_query(F.data == "info3")
    async def cmd_start_ai(callback: types.CallbackQuery):
        response3 = f"🎬 <b>{title3}</b>\n⭐ {rating3}\n\n📌 {genre3}, {country3}, {year3} год\n\nℹ️ {duration3}\n\n🧑‍ {director3}\n🎭 {actors3}\n\n📖 {about3}"

        keyboardr3 = InlineKeyboardBuilder()
        keyboardr3.button(text="Вернуться обратно", callback_data="backr3")
        keyboardr3.adjust(1)
        sent_message = await callback.message.answer(response3, parse_mode="HTML", reply_markup=keyboardr3.as_markup())

        @user_router.callback_query(F.data == "backr3")
        async def cmd_back(callback: types.CallbackQuery):
            await callback.bot.delete_message(chat_id=callback.from_user.id, message_id=sent_message.message_id)

    await state.clear()



    await callback.message.answer("<i>Если подборка не устраивает вас, уточните свои предпочтения или сбросьте их</i>",
                                  reply_markup=first_button_kb, parse_mode='HTML')

@user_router.callback_query(F.data == "rec2")
async def cmd_start_ai(callback: types.CallbackQuery, state: FSMContext):
    await callback.message.answer("Какие фильмы вам нравятся?")
    await state.set_state(Work.process)


@user_router.callback_query(F.data == "rec3")
async def reset_recommendations(callback: types.CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id

    conn = sqlite3.connect(REC_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT q FROM recommendations WHERE user_id = ?", (user_id,))
    existing_entry = cursor.fetchone()

    if existing_entry:

        cursor.execute("UPDATE recommendations SET q = '' WHERE user_id = ?", (user_id,))
    else:
        cursor.execute("INSERT INTO recommendations (user_id, q) VALUES (?, '')", (user_id,))

    conn.commit()
    conn.close()

    await callback.answer("Рекомендации успешно сброшены")




def description(query):
    """
    Выполняет семантический поиск фильмов.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT title, rating, country, genre, year, duration, director, actors, tags, about, embedding_str1 FROM movies")
    rows = cursor.fetchall()
    conn.close()

    query_embedding = semantic_model.encode(query)
    results = []

    for title, rating, country, genre, year, duration, director, actors, tags, about, emb_str in rows:
        try:
            movie_embedding = np.array(json.loads(emb_str))
        except Exception:
            continue
        norm_q = np.linalg.norm(query_embedding)
        norm_m = np.linalg.norm(movie_embedding)

        similarity = np.dot(query_embedding, movie_embedding) / (norm_q * norm_m) if norm_q and norm_m else 0

        results.append((title, rating, country, genre, year, duration, director, actors, tags, about, similarity))

    results.sort(key=lambda x: x[10], reverse=True)
    return [(title, rating, country, genre, year, duration, director, actors, tags, about) for title, rating, country, genre, year, duration, director, actors, tags, about, _ in results[:3]]

def tag(query):
    """
        Выполняет семантический поиск фильмов.
        """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT title, rating, country, genre, year, duration, director, actors, tags, about, embedding_str2 FROM movies")
    rows = cursor.fetchall()
    conn.close()

    query_embedding = semantic_model.encode(query)
    results = []

    for title, rating, country, genre, year, duration, director, actors, tags, about, emb_str in rows:
        try:
            movie_embedding = np.array(json.loads(emb_str))
        except Exception:
            continue
        norm_q = np.linalg.norm(query_embedding)
        norm_m = np.linalg.norm(movie_embedding)

        similarity = np.dot(query_embedding, movie_embedding) / (norm_q * norm_m) if norm_q and norm_m else 0

        results.append((title, rating, country, genre, year, duration, director, actors, tags, about, similarity))

    results.sort(key=lambda x: x[10], reverse=True)
    return [(title, rating, country, genre, year, duration, director, actors, tags, about) for
            title, rating, country, genre, year, duration, director, actors, tags, about, _ in results[:3]]



@user_router.message(Work.wait)
async def process_movie(message: types.Message, state: FSMContext):

    query = message.text.strip()
    results = description(query)

    if not results or len(results) < 3:
        await message.answer("Фильмы не найдены")
        await state.clear()
        return

    # Фильм 1
    title1, rating1, country1, genre1, year1, duration1, director1, actors1, tags1, about1 = results[0]
    response1 = f"🎬 <b>{title1}</b>\n⭐ {rating1}\n📌 {genre1}\n📖 {about1}"
    keyboard1 = InlineKeyboardBuilder()
    keyboard1.button(text="➕ В рекомендации", callback_data="ar1")
    keyboard1.button(text="✅ Просмотрено", callback_data=f"w1")
    keyboard1.button(text="⭐ Оценить", callback_data="rate1")
    keyboard1.button(text="📽️ Подробнее о фильме", callback_data="info01")
    keyboard1.adjust(1)
    await message.answer(response1, parse_mode="HTML", reply_markup=keyboard1.as_markup())

    @user_router.callback_query(F.data == "ar1")
    async def add_to_recommendations(callback: types.CallbackQuery, state: FSMContext):
        user_id = callback.from_user.id
        tags = tags1

        conn = sqlite3.connect(REC_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT q FROM recommendations WHERE user_id = ?", (user_id,))
        existing_entry = cursor.fetchone()

        if existing_entry:
            updated_rec = existing_entry[0] + ", " + tags
            cursor.execute("UPDATE recommendations SET q = ? WHERE user_id = ?", (updated_rec, user_id))
        else:
            cursor.execute("INSERT INTO recommendations (user_id, q) VALUES (?, ?)", (user_id, tags))

        conn.commit()
        conn.close()

        await callback.answer("Фильм добавлен в список рекомендаций")

    @user_router.callback_query(F.data.startswith("rate1"))
    async def rate_callback(callback: types.CallbackQuery, state: FSMContext):
        await state.update_data(rating_title=title1, rating_g=genre1, rating_r=rating1)

        await callback.message.answer(f"Оцените фильм «{title1}» от 1 до 10")
        await state.set_state(Work.mark)

    @user_router.callback_query(F.data == "w1")
    async def handle_callback(callback: types.CallbackQuery):
        user_id = callback.from_user.id

        conn = sqlite3.connect(USER_PATH)
        cursor = conn.cursor()

        try:
            # Добавляем фильм (если такого ещё нет)
            cursor.execute("""
                INSERT OR IGNORE INTO user_movies (user_id, title, watched, mark, rating, genre)
                    VALUES (?, ?, 1, 0, ?, ?)
                """, (user_id, title1, rating1, genre1))

            conn.commit()

            # Выбираем все записи из таблицы user_movies
            cursor.execute("SELECT * FROM user_movies")
            rows = cursor.fetchall()

            # Печатаем все строки
            for row in rows:
                print(row)  # Каждая строка будет кортежем

        except sqlite3.Error as e:
            print(f"Ошибка при добавлении фильма: {e}")

        finally:
            conn.close()

        await callback.answer("Фильм сохранён в список просмотренных! 🎬")

    @user_router.callback_query(F.data == "info01")
    async def cmd_start_ai(callback: types.CallbackQuery):
        response1 = f"🎬 <b>{title1}</b>\n⭐ {rating1}\n\n📌 {genre1}, {country1}, {year1} год\n\nℹ️ {duration1}\n\n🧑‍ {director1}\n🎭 {actors1}\n\n📖 {about1}"

        keyboard01 = InlineKeyboardBuilder()
        keyboard01.button(text="Вернуться обратно", callback_data="back1")
        keyboard01.adjust(1)
        sent_message = await callback.message.answer(response1, parse_mode="HTML", reply_markup=keyboard01.as_markup())

        @user_router.callback_query(F.data == "back1")
        async def cmd_back(callback: types.CallbackQuery):
            await callback.bot.delete_message(chat_id=callback.from_user.id, message_id=sent_message.message_id)



    # Фильм 2
    title2, rating2, country2, genre2, year2, duration2, director2, actors2, tags2, about2 = results[1]
    response2 = f"🎬 <b>{title2}</b>\n⭐ {rating2}\n📌 {genre2}\n📖 {about2}"
    keyboard2 = InlineKeyboardBuilder()
    keyboard2.button(text="➕ В рекомендации", callback_data=f"ar2")
    keyboard2.button(text="✅ Просмотрено", callback_data=f"w2")
    keyboard2.button(text="⭐ Оценить", callback_data=f"rate2")
    keyboard2.button(text="📽️ Подробнее о фильме", callback_data="info02")
    keyboard2.adjust(1)
    await message.answer(response2, parse_mode="HTML", reply_markup=keyboard2.as_markup())

    @user_router.callback_query(F.data == "ar2")
    async def add_to_recommendations(callback: types.CallbackQuery, state: FSMContext):
        user_id = callback.from_user.id
        tags = tags2

        conn = sqlite3.connect(REC_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT q FROM recommendations WHERE user_id = ?", (user_id,))
        existing_entry = cursor.fetchone()

        if existing_entry:
            updated_rec = existing_entry[0] + ", " + tags
            cursor.execute("UPDATE recommendations SET q = ? WHERE user_id = ?", (updated_rec, user_id))
        else:
            cursor.execute("INSERT INTO recommendations (user_id, q) VALUES (?, ?)", (user_id, tags))

        conn.commit()
        conn.close()

        await callback.answer("Фильм добавлен в список рекомендаций")


    @user_router.callback_query(F.data.startswith("rate2"))
    async def rate_callback(callback: types.CallbackQuery, state: FSMContext):
        await state.update_data(rating_title=title2, rating_g=genre2, rating_r=rating2)

        await callback.message.answer(f"Оцените фильм «{title2}» от 1 до 10")
        await state.set_state(Work.mark)

    @user_router.callback_query(F.data == "w2")
    async def handle_callback(callback: types.CallbackQuery):
        user_id = callback.from_user.id

        conn = sqlite3.connect(USER_PATH)
        cursor = conn.cursor()

        try:
            # Добавляем фильм (если такого ещё нет)
            cursor.execute("""
                    INSERT OR IGNORE INTO user_movies (user_id, title, watched, mark, rating, genre)
                    VALUES (?, ?, 1, 0, ?, ?)
                """, (user_id, title2, rating2, genre2))

            conn.commit()

            # Выбираем все записи из таблицы user_movies
            cursor.execute("SELECT * FROM user_movies")
            rows = cursor.fetchall()

            # Печатаем все строки
            for row in rows:
                print(row)  # Каждая строка будет кортежем

        except sqlite3.Error as e:
            print(f"Ошибка при добавлении фильма: {e}")

        finally:
            conn.close()

        await callback.answer("Фильм сохранён в список просмотренных! 🎬")

    @user_router.callback_query(F.data == "info02")
    async def cmd_start_ai(callback: types.CallbackQuery):
        response02 = f"🎬 <b>{title2}</b>\n⭐ {rating2}\n\n📌 {genre2}, {country2}, {year2} год\n\nℹ️ {duration2}\n\n🧑‍ {director2}\n🎭 {actors2}\n\n📖 {about2}"
        keyboard02 = InlineKeyboardBuilder()
        keyboard02.button(text="Вернуться обратно", callback_data="back2")
        keyboard02.adjust(1)
        sent_message = await callback.message.answer(response02, parse_mode="HTML", reply_markup=keyboard02.as_markup())

        @user_router.callback_query(F.data == "back2")
        async def cmd_back(callback: types.CallbackQuery):
            await callback.bot.delete_message(chat_id=callback.from_user.id, message_id=sent_message.message_id)



    # Фильм 3
    title3, rating3, country3, genre3, year3, duration3, director3, actors3, tags3, about3 = results[2]
    response3 = f"🎬 <b>{title3}</b>\n⭐ {rating3}\n📌 {genre3}\n📖 {about3}"
    keyboard3 = InlineKeyboardBuilder()
    keyboard3.button(text="➕ В рекомендации", callback_data=f"ar3")
    keyboard3.button(text="✅ Просмотрено", callback_data=f"w3")
    keyboard3.button(text="⭐ Оценить", callback_data=f"rate3")
    keyboard3.button(text="📽️ Подробнее о фильме", callback_data="info03")
    keyboard3.adjust(1)
    await message.answer(response3, parse_mode="HTML", reply_markup=keyboard3.as_markup())

    @user_router.callback_query(F.data == "ar3")
    async def add_to_recommendations(callback: types.CallbackQuery, state: FSMContext):
        user_id = callback.from_user.id
        tags = tags3

        conn = sqlite3.connect(REC_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT q FROM recommendations WHERE user_id = ?", (user_id,))
        existing_entry = cursor.fetchone()

        if existing_entry:
            updated_rec = existing_entry[0] + ", " + tags
            cursor.execute("UPDATE recommendations SET q = ? WHERE user_id = ?", (updated_rec, user_id))
        else:
            cursor.execute("INSERT INTO recommendations (user_id, q) VALUES (?, ?)", (user_id, tags))

        conn.commit()
        conn.close()

        await callback.answer("Фильм добавлен в список рекомендаций")


    @user_router.callback_query(F.data.startswith("rate3"))
    async def rate_callback(callback: types.CallbackQuery, state: FSMContext):
        await state.update_data(rating_title=title3, rating_g=genre3, rating_r=rating3)
        await callback.message.answer(f"Оцените фильм «{title3}» от 1 до 10")
        await state.set_state(Work.mark)

    @user_router.callback_query(F.data == "w3")
    async def handle_callback(callback: types.CallbackQuery):
        user_id = callback.from_user.id

        conn = sqlite3.connect(USER_PATH)
        cursor = conn.cursor()

        try:
            # Добавляем фильм (если такого ещё нет)
            cursor.execute("""
                    INSERT OR IGNORE INTO user_movies (user_id, title, watched, mark, rating, genre)
                    VALUES (?, ?, 1, 0, ?, ?)
                """, (user_id, title3, rating3, genre3))

            conn.commit()

            # Выбираем все записи из таблицы user_movies
            cursor.execute("SELECT * FROM user_movies")
            rows = cursor.fetchall()

            # Печатаем все строки
            for row in rows:
                print(row)  # Каждая строка будет кортежем

        except sqlite3.Error as e:
            print(f"Ошибка при добавлении фильма: {e}")

        finally:
            conn.close()

        await callback.answer("Фильм сохранён в список просмотренных! 🎬")

    @user_router.callback_query(F.data == "info03")
    async def cmd_start_ai(callback: types.CallbackQuery):
        response03 = f"🎬 <b>{title3}</b>\n⭐ {rating3}\n\n📌 {genre3}, {country3}, {year3} год\n\nℹ️ {duration3}\n\n🧑‍ {director3}\n🎭 {actors3}\n\n📖 {about3}"

        keyboard03 = InlineKeyboardBuilder()
        keyboard03.button(text="Вернуться обратно", callback_data="back3")
        keyboard03.adjust(1)
        sent_message = await callback.message.answer(response03, parse_mode="HTML", reply_markup=keyboard03.as_markup())

        @user_router.callback_query(F.data == "back3")
        async def cmd_back(callback: types.CallbackQuery):
            await callback.bot.delete_message(chat_id=callback.from_user.id, message_id=sent_message.message_id)

    await state.clear()
    await message.answer("<i>Если подборка не устраивает Вас, опишите фильм подробнее и попробуйте ещё раз</i>",
                                  reply_markup=first_button_kb, parse_mode='HTML')

@user_router.message(Work.mark)
async def save_rating(message: types.Message, state: FSMContext):
    user_id = message.from_user.id
    data = await state.get_data()
    title = data.get("rating_title")
    rating = data.get("rating_r")
    genre = data.get("rating_g")
    print(title)

    if not title:
        await message.answer("Ошибка: не найдено название фильма. Попробуйте снова.")
        await state.clear()
        return

    try:
        rating0 = int(message.text)
        if not (1 <= rating0 <= 10):
            await message.answer("Введите число от 1 до 10")
            return

        conn = sqlite3.connect(USER_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM user_movies WHERE user_id = ? AND title = ?", (user_id, title))
        existing_entry = cursor.fetchone()

        if existing_entry:
            cursor.execute("UPDATE user_movies SET mark = ? WHERE user_id = ? AND title = ? AND rating = ? AND genre = ?", (rating0, user_id, title, rating, genre))
        else:
            cursor.execute("INSERT INTO user_movies (user_id, title, watched, mark, rating, genre) VALUES (?, ?, 1, ?, ?, ?)",
                           (user_id, title, rating0, rating, genre))

        conn.commit()

        # Выбираем все записи из таблицы user_movies и выводим в консоль
        cursor.execute("SELECT * FROM user_movies")
        rows = cursor.fetchall()
        for row in rows:
            print(row)

        await message.answer(f"Вы поставили фильму «{title}» оценку {rating0}/10",reply_markup=first_button_kb)

    except ValueError:
        await message.answer("Введите число от 1 до 10")
    except sqlite3.Error as e:
        print(f"Ошибка при работе с базой данных: {e}")
        await message.answer("Произошла ошибка при сохранении оценки. Попробуйте позже.")
    finally:
        conn.close()
        await state.clear()

@user_router.message(Work.process)
async def cmd_ai_process(message: types.Message, state: FSMContext):
    user_id = message.from_user.id
    user_text = message.text.strip() if message.text else ""

    if not user_text:
        await message.answer("Пожалуйста, введите текст")
        return

    conn = sqlite3.connect(REC_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT q FROM recommendations WHERE user_id = ?", (user_id,))
    existing_entry = cursor.fetchone()

    if existing_entry:
        # Если запись есть, обновляем поле rec, добавляя новый текст
        updated_rec = existing_entry[0] + ", " + user_text
        cursor.execute("UPDATE recommendations SET q = ? WHERE user_id = ?", (updated_rec, user_id))
    else:
        # Если записи нет, создаем новую
        cursor.execute("INSERT INTO recommendations (user_id, q) VALUES (?, ?)", (user_id, user_text))

    conn.commit()
    conn.close()

    await message.answer("Изменения успешно сохранены", reply_markup=first_button_kb)
    await state.clear()  # Очищаем состояние



@user_router.message()
async def echo_messages(message: types.Message):
    await message.bot.send_chat_action(chat_id=message.from_user.id, action=ChatAction.TYPING)
    if message.text:
        await message.answer(text="Подождите...\nУ меня нет команды на данное сообщение.\nВот мой весь список команд:\n"                                  "/start - Запуск бота\n")
    else:
        try:
            await message.copy_to(chat_id=message.chat.id)
        except TypeError:
            await message.reply(text="💥🫣 Ничего непонятно...")
