import logging                      # Стандартный модуль логирования событий
import os                            # Работа с переменными окружения и файловой системой
import asyncio                       # Асинхронное выполнение задач
import concurrent.futures            # Пул потоков для запуска блокирующего кода (LLM, БД) без блокировки бота
import signal                        # Обработка системных сигналов для корректного завершения работы

from telegram import Update          # Объект обновления (входящее сообщение, команда и т.д.)
from telegram.ext import (
    ApplicationBuilder,              # Строитель объекта приложения бота
    CommandHandler,                  # Обработчик команд (например, /start)
    ContextTypes,                    # Типизация контекста для обработчиков
    MessageHandler,                  # Обработчик текстовых сообщений
    ConversationHandler,             # Обработчик многошаговых диалогов (FSM)
    filters,                         # Фильтры для отбора нужных сообщений
)
from dotenv import load_dotenv       # Загрузка переменных окружения из файла .env

from db import DBClient              # Наш клиент для работы с ChromaDB
from llm import LLMClient, GenerationConfig  # Клиент языковой модели и конфиг генерации

# Загружаем переменные окружения из .env (TELEGRAM_TOKEN, WEBHOOK_URL и т.д.)
load_dotenv()

# Настраиваем формат и уровень логов; все события будут выводиться в консоль
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Токен бота из BotFather
TELEGRAM_TOKEN: str = os.environ.get("TELEGRAM_TOKEN", "")

# URL для webhook-режима; если пуст — бот запускается в режиме polling
WEBHOOK_URL: str = os.environ.get("WEBHOOK_URL", "")

# Порт, на котором слушает webhook-сервер
PORT: int = int(os.environ.get("PORT", "8080"))

# Директория и название коллекции для ChromaDB
DB_DIR: str = "./chromadb"
DB_COLLECTION: str = "my_collection"

# Путь к локальной директории с файлами языковой модели
MODEL_PATH: str = "./Vikhr-Llama-3.2-1B-Instruct"

# ── Состояния ConversationHandler ─────────────────────────────────────────────
WAITING_GENERATE = 1  # Ожидаем запрос для прямой генерации
WAITING_RAG = 2       # Ожидаем запрос для RAG-поиска

# ── Глобальные объекты ────────────────────────────────────────────────────────
db_client: DBClient | None = None   # Клиент базы данных (инициализируется в main)
llm_client: LLMClient | None = None  # Клиент языковой модели (инициализируется в main)

# Единый пул потоков для всего приложения.
# LLM и ChromaDB — синхронные (блокирующие) операции, их нельзя вызывать напрямую
# из async-кода: они заморозят весь event loop. Поэтому запускаем их в отдельных потоках.
# Создаём один общий пул, а не новый на каждый запрос — это эффективнее.
_executor: concurrent.futures.ThreadPoolExecutor | None = None


# ── Вспомогательные функции ───────────────────────────────────────────────────
def _build_rag_prompt(user_query: str, context: str) -> str:
    """
    Формирует итоговый промпт для языковой модели в режиме RAG
    (Retrieval-Augmented Generation — генерация с опорой на найденный контекст).

    Если релевантный контекст найден в базе — включаем его в промпт,
    чтобы модель могла опираться на конкретные факты при ответе.

    :param user_query: Вопрос пользователя.
    :param context:    Текст, найденный в базе знаний (или пустая строка).
    :return:           Готовый промпт для передачи в LLM.
    """
    if context:
        return (
            f"Контекст из базы знаний:\n{context}\n\n"
            f"Вопрос пользователя: {user_query}"
        )
    # Если ничего не найдено — сообщаем об этом модели, но всё равно отвечаем
    return (
        f"Релевантный контекст не найден.\n\n"
        f"Вопрос пользователя: {user_query}"
    )


# ── Обработчики команд ────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обработчик команды /start и /help.
    Отправляет пользователю список доступных команд.
    """
    text = (
        "Привет! Я бот с языковой моделью.\n\n"
        "Команды:\n"
        "  /generate — задать вопрос напрямую модели\n"
        "  /rag      — задать вопрос с поиском по базе знаний\n"
        "  /add      — добавить текст в базу знаний\n"
        "  /db       — показать содержимое базы знаний\n"
        "  /cancel   — отменить текущую операцию"
    )
    await update.message.reply_text(text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /help — перенаправляет на /start."""
    await start(update, context)


# /generate: прямая генерация без RAG
async def generate_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Точка входа в диалог /generate.
    Просит пользователя ввести запрос и переводит FSM в состояние WAITING_GENERATE.
    """
    await update.message.reply_text("Введите ваш запрос для генерации:")
    return WAITING_GENERATE  # Переходим в состояние ожидания запроса


async def generate_handle(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обрабатывает текстовый запрос пользователя в состоянии WAITING_GENERATE.
    Передаёт запрос напрямую в LLM (без поиска по базе знаний).
    Завершает диалог после получения ответа.
    """
    user_text = update.message.text.strip()

    # Защита от пустого ввода — остаёмся в том же состоянии
    if not user_text:
        await update.message.reply_text("Запрос не может быть пустым. Попробуйте ещё раз.")
        return WAITING_GENERATE

    # Отправляем предварительное сообщение, которое потом заменим ответом модели
    msg = await update.message.reply_text("⏳ Генерирую ответ... (до 2 минут)")

    try:
        # get_running_loop() — правильный способ получить текущий event loop в async-контексте
        loop = asyncio.get_running_loop()

        # run_in_executor запускает синхронный llm_client.ask в отдельном потоке,
        # не блокируя event loop и позволяя боту обрабатывать другие обновления
        answer = await asyncio.wait_for(
            loop.run_in_executor(_executor, llm_client.ask, user_text),
            timeout=110.0  # Максимальное время ожидания ответа от модели
        )

        # Редактируем ранее отправленное сообщение — заменяем "⏳" на готовый ответ
        await msg.edit_text(answer)

    except asyncio.TimeoutError:
        # Модель не успела ответить за отведённое время
        await msg.edit_text("❌ Превышено время ожидания (110 сек). Попробуйте задать более простой вопрос.")
    except Exception as e:
        logger.exception("Ошибка LLM при генерации")
        await msg.edit_text(f"❌ Ошибка модели: {e}")

    # END завершает диалог — FSM возвращается в начальное состояние
    return ConversationHandler.END


# /rag: генерация с поиском по базе знаний
async def rag_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Точка входа в диалог /rag.
    Переводит FSM в состояние WAITING_RAG.
    """
    await update.message.reply_text("Введите ваш запрос для поиска в базе знаний:")
    return WAITING_RAG


async def rag_handle(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обрабатывает запрос в режиме RAG:
    1. Ищет релевантный контекст в ChromaDB по семантическому сходству.
    2. Формирует промпт, объединяя найденный контекст и вопрос пользователя.
    3. Передаёт промпт в LLM и возвращает ответ.
    """
    user_text = update.message.text.strip()

    if not user_text:
        await update.message.reply_text("Запрос не может быть пустым. Попробуйте ещё раз.")
        return WAITING_RAG

    msg = await update.message.reply_text("🔍 Ищу в базе знаний...")

    try:
        loop = asyncio.get_running_loop()

        # Семантический поиск в ChromaDB (синхронная операция — запускаем в потоке)
        context_text = await asyncio.wait_for(
            loop.run_in_executor(_executor, db_client.select_query, user_text),
            timeout=30.0  # Поиск должен быть быстрым — таймаут меньше, чем для LLM
        )

        await msg.edit_text("⏳ Генерирую ответ...")

        # Формируем промпт с найденным контекстом
        prompt = _build_rag_prompt(user_text, context_text)

        # Передаём промпт в LLM и получаем ответ
        answer = await asyncio.wait_for(
            loop.run_in_executor(_executor, llm_client.ask, prompt),
            timeout=110.0
        )

        await msg.edit_text(answer)

    except asyncio.TimeoutError:
        await msg.edit_text("❌ Превышено время ожидания. Попробуйте упростить вопрос.")
    except Exception as e:
        logger.exception("Ошибка в RAG-обработчике")
        await msg.edit_text(f"❌ Ошибка: {e}")

    return ConversationHandler.END


# /add: добавление текста в базу знаний
async def add_to_db(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Добавляет текст, переданный после команды, в ChromaDB.
    Пример использования: /add Python — интерпретируемый язык программирования.
    """
    # context.args содержит список слов, переданных после команды
    text = " ".join(context.args).strip() if context.args else ""

    if not text:
        await update.message.reply_text(
            "Использование: /add <текст>\n"
            "Пример: /add Python — интерпретируемый язык программирования."
        )
        return

    try:
        db_client.insert_query(text)
        await update.message.reply_text("✅ Текст добавлен в базу знаний.")
    except Exception as e:
        logger.exception("Ошибка при записи в БД")
        await update.message.reply_text(f"Ошибка при добавлении: {e}")


# /db: просмотр содержимого базы знаний
async def show_db(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Показывает все документы, хранящиеся в базе знаний.
    Ограничивает вывод 4000 символами — это лимит одного сообщения в Telegram.
    """
    try:
        content = db_client.get_db_data()

        # Telegram не позволяет отправить сообщение длиннее 4096 символов
        if len(content) > 4000:
            content = content[:4000] + "\n\n… (обрезано)"

        await update.message.reply_text(f"📚 База знаний:\n\n{content}")
    except Exception as e:
        logger.exception("Ошибка при чтении БД")
        await update.message.reply_text(f"Ошибка чтения базы: {e}")


# /cancel: отмена текущей операции
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Прерывает текущий диалог ConversationHandler.
    Доступен в любом состоянии FSM как fallback-обработчик.
    """
    await update.message.reply_text("Операция отменена.")
    return ConversationHandler.END


async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик неизвестных команд — подсказывает пользователю ввести /start."""
    await update.message.reply_text(
        "Неизвестная команда. Введите /start для списка команд."
    )


# Shutdown
def _shutdown(sig: signal.Signals) -> None:
    """
    Обработчик сигналов завершения (SIGINT = Ctrl+C, SIGTERM = kill).
    Корректно завершает пул потоков, не дожидаясь окончания текущих задач.
    Это предотвращает зависание процесса при остановке бота.
    """
    logger.info(f"Получен сигнал {sig.name}, завершаю работу…")
    if _executor:
        # wait=False — не ждём завершения текущих задач, просто останавливаем пул
        _executor.shutdown(wait=False)


# Точка входа
def main() -> None:
    """
    Инициализирует все компоненты и запускает Telegram-бота.
    Поддерживает два режима работы:
    - Polling:  бот сам периодически опрашивает серверы Telegram (удобно для разработки).
    - Webhook:  Telegram сам присылает обновления на указанный URL (рекомендуется для продакшна).
    """
    global db_client, llm_client, _executor

    # Без токена бот не сможет подключиться к Telegram API
    if not TELEGRAM_TOKEN:
        raise RuntimeError(
            "Токен Telegram не задан. "
            "Установите переменную окружения TELEGRAM_TOKEN."
        )

    # Создаём единый пул потоков для всего приложения.
    _executor = concurrent.futures.ThreadPoolExecutor()

    # Регистрируем обработчики сигналов завершения
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda s, _: _shutdown(signal.Signals(s)))

    # Загружаем языковую модель в память (может занять несколько минут)
    logger.info("Загрузка модели…")
    llm_client = LLMClient(model_path=MODEL_PATH)

    # Подключаемся к векторной базе данных
    logger.info("Подключение к базе данных…")
    db_client = DBClient(db_dir=DB_DIR, collection_name=DB_COLLECTION)

    # Настраиваем HTTP-клиент с увеличенными таймаутами,
    # так как генерация может занимать десятки секунд
    from telegram.request import HTTPXRequest
    request = HTTPXRequest(
        connect_timeout=60.0,  # Таймаут установки соединения
        read_timeout=60.0,     # Таймаут чтения ответа
        write_timeout=60.0,    # Таймаут отправки данных
    )

    # Строим объект приложения бота
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).request(request).build()

    # ConversationHandler для /generate реализует двухшаговый диалог:
    # 1. Пользователь вводит /generate
    # 2. ConversationHandler подхватывает управление и вызывает generate_entry
    # 3. Обработчик, который активен в состоянии WAITING_GENERATE - generate_handle
    generate_conv = ConversationHandler(
        entry_points=[CommandHandler("generate", generate_entry)],
        states={
            WAITING_GENERATE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, generate_handle) # отсеиваем команды
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],  # /cancel доступен в любой момент
    )

    # ConversationHandler для /rag — аналогичная структура, но с RAG-логикой
    rag_conv = ConversationHandler(
        entry_points=[CommandHandler("rag", rag_entry)],
        states={
            WAITING_RAG: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, rag_handle)
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    # Регистрируем все обработчики в приложении
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("add", add_to_db))
    app.add_handler(CommandHandler("db", show_db))
    app.add_handler(generate_conv)
    app.add_handler(rag_conv)
    # Этот обработчик должен быть последним: ловит все нераспознанные команды
    app.add_handler(MessageHandler(filters.COMMAND, unknown))

    if WEBHOOK_URL:
        # Webhook-режим: Telegram сам шлёт POST-запросы на наш сервер
        logger.info(f"Запуск через webhook: {WEBHOOK_URL}")
        app.run_webhook(
            listen="0.0.0.0",      # Слушаем на всех сетевых интерфейсах
            port=PORT,             # Порт из переменной окружения
            webhook_url=WEBHOOK_URL,
        )
    else:
        # Polling-режим: бот сам спрашивает Telegram "есть ли новые сообщения?"
        logger.info("Запуск через polling…")
        app.run_polling()


if __name__ == "__main__":
    main()
