import asyncio
import logging

from aiogram import Bot, Dispatcher
from user_handlers import user_router

from config import BOT_TOKEN


async def main():

    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(user_router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    try:
        asyncio.run(main())
    except:
        print("BOT OFF")