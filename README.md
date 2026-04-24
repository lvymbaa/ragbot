# Telegram-бот с языковой моделью и RAG

Telegram-бот на базе локальной языковой модели [Vikhr-Llama-3.2-1B-Instruct](https://huggingface.co/Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct) с поддержкой RAG (Retrieval-Augmented Generation) — генерации ответов с опорой на базу знаний.

---

## Возможности

- **/generate** — задать вопрос напрямую языковой модели
- **/rag** — задать вопрос с семантическим поиском по базе знаний
- **/add** — добавить текст в базу знаний
- **/db** — просмотреть содержимое базы знаний
- **/cancel** — отменить текущую операцию

---

## Архитектура

```
run.py          — точка входа, Telegram-бот, обработчики команд
llm.py          — клиент языковой модели (Singleton, локальный инференс)
db.py           — клиент векторной базы данных ChromaDB
```

### Как работает RAG

1. Пользователь задаёт вопрос через `/rag`
2. ChromaDB ищет наиболее близкий по смыслу документ в базе знаний (семантический поиск по эмбеддингам)
3. Найденный контекст + вопрос пользователя объединяются в промпт
4. Промпт передаётся в языковую модель, которая генерирует ответ с опорой на контекст

---

## Системные требования

### Программное обеспечение

| Компонент | Минимум | Рекомендуется |
|---|---|---|
| Python | 3.10 | 3.11+ |
| CUDA | 11.8 | 12.1+ |
| pip | 23.0 | последняя версия |

### Аппаратное обеспечение

| Компонент | Минимум | Рекомендуется |
|---|---|---|
| RAM | 8 ГБ | 16 ГБ |
| VRAM (GPU) | 4 ГБ | 8 ГБ |
| Место на диске | 5 ГБ | 10 ГБ |
| CPU | 4 ядра | 8 ядер |

> Без GPU (только CPU) бот будет работать, но генерация одного ответа может занимать несколько минут.

### Дополнительно

- Токен Telegram-бота (получить у [@BotFather](https://t.me/BotFather))
- Локально скачанная модель `Vikhr-Llama-3.2-1B-Instruct`

---

## Установка

### 1. Клонировать репозиторий

```bash
git clone <url>
cd <папка проекта>
```

### 2. Создать и активировать виртуальную среду

Виртуальная среда изолирует зависимости проекта от системного Python — это позволяет избежать конфликтов между пакетами разных проектов.

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

После активации в начале строки терминала появится `(venv)` — это означает, что виртуальная среда активна. Все дальнейшие команды выполняются внутри неё.

Для деактивации (когда закончили работу):
```bash
deactivate
```

### 3. Установить зависимости

```bash
pip install -r requirements.txt
```

### 4. Скачать модель

Скачайте модель с Hugging Face и положите в корень проекта:

```
./Vikhr-Llama-3.2-1B-Instruct/
```

### 5. Создать файл `.env`

Файл `.env` хранит секреты и настройки, которые не должны попадать в репозиторий. Создайте его в корне проекта:

```env
TELEGRAM_TOKEN=ваш_токен_от_BotFather

# Опционально — для webhook-режима:
WEBHOOK_URL=https://ваш-домен.com/webhook
PORT=8080
```

Токен можно получить у [@BotFather](https://t.me/BotFather) в Telegram.

---

## Запуск

```bash
python run.py
```

По умолчанию бот запускается в режиме **polling** — сам опрашивает серверы Telegram. Если задана переменная `WEBHOOK_URL` — запускается в режиме **webhook**.

---

## Структура проекта

```
.
├── run.py                          # Telegram-бот
├── llm.py                          # Клиент языковой модели
├── db.py                           # Клиент ChromaDB
├── requirements.txt                # Зависимости
├── .env                            # Переменные окружения (не коммитить!)
├── venv/                           # Виртуальная среда (не коммитить!)
├── chromadb/                       # Файлы базы данных (создаётся автоматически)
└── Vikhr-Llama-3.2-1B-Instruct/   # Файлы языковой модели
```

---

## Ключевые фрагменты кода

### Паттерн Singleton для языковой модели (llm.py)

Модель загружается в память один раз и переиспользуется при каждом запросе. `__new__` всегда возвращает один и тот же объект, а флаг `_initialized` защищает от повторной загрузки при повторном вызове конструктора.

```python
def __new__(cls, *args, **kwargs):
    if cls.__instance is None:
        cls.__instance = super().__new__(cls)
    return cls.__instance

def __init__(self, model_path: str = "./Vikhr-Llama-3.2-1B-Instruct") -> None:
    if hasattr(self, "_initialized"):
        return
    # ... загрузка модели ...
    self._initialized = True
```

### Семантический поиск в ChromaDB (db.py)

ChromaDB автоматически строит векторные представления (эмбеддинги) текста и ищет документы по смысловой близости, а не по точному совпадению слов. Результат — двойной список: внешний по запросам, внутренний по найденным документам.

```python
def select_query(self, text: str) -> str:
    result = self._collection.query(query_texts=[text], n_results=1)
    if not result["documents"] or not result["documents"][0]:
        return ""
    return f"result: {result['documents'][0][0]}\n"
```

### Запуск блокирующего кода без остановки бота (run.py)

LLM и ChromaDB — синхронные блокирующие операции. Вызов напрямую из async-кода заморозил бы весь event loop и бот перестал бы реагировать на других пользователей. `run_in_executor` запускает их в отдельном потоке из общего пула, не блокируя event loop.

```python
loop = asyncio.get_running_loop()
answer = await asyncio.wait_for(
    loop.run_in_executor(_executor, llm_client.ask, user_text),
    timeout=110.0
)
```

### Формирование RAG-промпта (run.py)

Перед передачей в модель вопрос пользователя объединяется с найденным контекстом из базы знаний. Если контекст не найден — модель об этом предупреждается и отвечает по своим знаниям.

```python
def _build_rag_prompt(user_query: str, context: str) -> str:
    if context:
        return (
            f"Контекст из базы знаний:\n{context}\n\n"
            f"Вопрос пользователя: {user_query}"
        )
    return (
        f"Релевантный контекст не найден.\n\n"
        f"Вопрос пользователя: {user_query}"
    )
```

### Инференс языковой модели (llm.py)

Входной текст форматируется по шаблону модели, токенизируется и передаётся в `generate`. Из результата отрезаются входные токены — остаются только новые, сгенерированные.

```python
text = self._tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

with torch.inference_mode():
    generated_ids = self._model.generate(
        **model_inputs,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        do_sample=config.do_sample,
        pad_token_id=self._tokenizer.eos_token_id
    )

new_tokens = generated_ids[0][len(model_inputs.input_ids[0]):]
return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
```

---

## Зависимости

| Библиотека | Назначение |
|---|---|
| `torch` | Работа с нейросетью и GPU |
| `transformers` | Загрузка модели и токенизатора |
| `accelerate` | Автоматическое распределение модели по устройствам |
| `chromadb` | Векторная база данных для RAG |
| `python-telegram-bot` | Telegram Bot API |
| `python-dotenv` | Загрузка переменных окружения из `.env` |

---

## Примечания

- Файл `.env` и директория `venv/` не должны попадать в git — добавьте их в `.gitignore`
- Директория `chromadb/` создаётся автоматически при первом запуске
- Модель загружается в память один раз при старте бота (паттерн Singleton)
- LLM и ChromaDB работают в отдельных потоках, не блокируя обработку других сообщений
