import chromadb
import uuid

class DBClient:
    """
    Клиент для работы с векторной базой данных ChromaDB.
    Обеспечивает хранение текстовых документов и семантический поиск по ним.
    """

    def __init__(self, db_dir: str = "./chromadb", collection_name: str = "my_collection") -> None:
        """
        Инициализация клиента базы данных.

        :param db_dir: Путь к директории с БД.
        :param collection_name: Название коллекции.
        """
        try:
            # Инициализация клиента БД (PersistentClient - сохранение данных на диск)
            client = chromadb.PersistentClient(path=db_dir)

            # Получаем существующую коллекцию или создаём новую, если её нет
            self._collection = client.get_or_create_collection(name=collection_name)
        except Exception as e:
            print("Ошибка при инициализации базы данных: ", e)

    def insert_query(self, text: str) -> None:
        """
        Добавляет текстовый документ в коллекцию.
        ChromaDB автоматически строит векторное представление (эмбеддинг) текста
        для последующего семантического поиска.

        :param text: Текст документа для сохранения.
        """
        # Генерация ID для каждого документа
        doc_id = str(uuid.uuid4())

        # Добавление документа в коллекцию
        self._collection.add(
            documents=[text],
            ids=[doc_id]
        )

    def select_query(self, text: str) -> str:
        """
        Выполняет семантический поиск по базе: находит документ,
        наиболее близкий по смыслу к переданному тексту.

        :param text: Поисковый запрос.
        :return: Наиболее релевантный документ в виде строки, либо пустая строка.
        """
        # Поиск документа в БД (n_results=1 - возвращается 1 максимально подходящий документ)
        result = self._collection.query(query_texts=[text], n_results=1)

        # Если коллекция пуста или ничего не найдено — возвращается пустая строка
        if not result["documents"] or not result["documents"][0]:
            return ""

        # result["documents"] — список списков; [0][0] — первый документ первого запроса
        return f"result: {result['documents'][0][0]}\n"

    def get_db_data(self) -> str:
        """
        Возвращает все документы из коллекции в виде одной строки.
        Используется для отображения содержимого базы знаний пользователю.

        :return: Все документы, разделённые двойным переносом строки.
        """
        # Получение всех записей
        db_data = self._collection.get()

        # Если документов нет — вывод сообщения
        if not db_data["documents"]:
            return "База данных пуста!"

        # Объединение результатов в одну строку с двойным переносом (для читаемости)
        return "\n\n".join(db_data["documents"])
