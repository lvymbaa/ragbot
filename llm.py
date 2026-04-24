import torch
import os
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class GenerationConfig:
    """
    Параметры генерации текста языковой моделью.

    max_new_tokens: Максимальное количество токенов в ответе модели.
    temperature:    Температура сэмплирования — чем выше, тем более случайны ответы.
                    При значении 1.0 распределение не изменяется, ниже — ответы консервативнее.
    do_sample:      Если True — используется случайное сэмплирование (с учётом temperature).
                    Если False — жадный поиск (всегда выбирается наиболее вероятный токен).
    """
    max_new_tokens: int = 2048
    temperature: float = 0.7
    do_sample: bool = True


class LLMClient:
    """
    Клиент для работы с локальной языковой моделью (LLM).
    Реализует паттерн Singleton: в рамках одного процесса существует
    только один экземпляр класса, чтобы модель не загружалась в память дважды.
    """

    # Хранит единственный экземпляр класса
    __instance = None

    # Системный промпт задаёт общее поведение модели при каждом запросе
    SYSTEM_PROMPT = (
        "Ты умный ассистент. Отвечай на вопрос пользователя, опираясь на предоставленный контекст. "
        "Если контекст не содержит нужной информации — скажи об этом и ответь по своим знаниям."
    )

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self, model_path: str = "./Vikhr-Llama-3.2-1B-Instruct") -> None:
        # Если модель уже была загружена ранее — пропускаем инициализацию
        if hasattr(self, "_initialized"):
            return

        self._model_path = os.path.abspath(model_path)
        self._validate_model_path()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._model = None
        self._load_model()
        self._initialized = True

    def _validate_model_path(self) -> None:
        """
        Проверяет существование директории с моделью.
        Бросает FileNotFoundError, если директория не найдена,
        чтобы ошибка была замечена сразу, а не при первом запросе.
        """
        if not os.path.exists(self._model_path):
            raise FileNotFoundError(f"Директория с моделью не найдена: {self._model_path}")

    def _load_model(self) -> None:
        """
        Загружает токенизатор и языковую модель из локальной директории.

        Токенизатор преобразует текст в последовательность числовых токенов
        и обратно. Модель принимает токены и генерирует продолжение.

        local_files_only=True — запрещает попытки скачать файлы из интернета
        torch_dtype=bfloat16  — используем 16-битное представление чисел
        device_map="auto"     — автоматически распределяет слои модели по GPU/CPU
        """
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path,
            local_files_only=True
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            device_map="auto",       # Автоматическое распределение по устройствам
            torch_dtype=torch.bfloat16,  #
            local_files_only=True    # Только локальные файлы
        )

        # Переводим модель в режим инференса
        self._model.eval()

    def ask(self, user_text: str, config: GenerationConfig | None = None) -> str:
        """
        Отправляет запрос языковой модели и возвращает сгенерированный ответ.

        :param user_text: Текст запроса от пользователя.
        :param config:    Параметры генерации. Если не переданы — используются значения по умолчанию.
        :return:          Ответ модели в виде строки.
        """
        if not user_text:
            raise ValueError("Запрос не может быть пустым.")

        # Формирование диалога в формате chat: системный промпт + сообщение пользователя
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_text.strip()},
        ]

        # Если конфиг не передан — используем значения по умолчанию
        config = config or GenerationConfig()

        # apply_chat_template форматирует диалог в строку по шаблону конкретной модели
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,           # Возвращаем строку, а не токены — токенизируем ниже
            add_generation_prompt=True,  # Добавляем маркер начала ответа модели
        )

        # Токенизируем готовую строку и переносим тензоры на нужное устройство
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        # inference_mode отключает вычисление градиентов — ускоряет инференс и экономит память
        with torch.inference_mode():
            generated_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=config.max_new_tokens,  # Ограничение длины ответа
                temperature=config.temperature,         # Степень случайности генерации
                do_sample=config.do_sample,             # Режим сэмплирования
                pad_token_id=self._tokenizer.eos_token_id  # Токен паддинга = токен конца текста
            )

        # Отрезаем входные токены из результата
        new_tokens = generated_ids[0][len(model_inputs.input_ids[0]):]

        # Декодируем токены обратно в текст, убирая служебные символы
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
