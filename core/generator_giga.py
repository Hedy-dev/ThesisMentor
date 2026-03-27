import os
import time
from typing import List, Dict
from gigachat import GigaChat 
from langchain_core.documents import Document
from dotenv import load_dotenv
# импорт эмбеддингов
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
# Загружаем переменные из .env
load_dotenv()

class GeneratorManager:
    def __init__(self, db_path: str = "./vector_db", collection_name: str = "thesis_kb"):
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Эмбеддинги для поиска в базе знаний
        self.embeddings = HuggingFaceEmbeddings(
            model_name="cointegrated/rubert-tiny2",
            model_kwargs={"device": "cpu"}
        )

        self.client = QdrantClient(path=self.db_path)
        self.vector_store = None

        # Инициализация прямого SDK GigaChat
        self.credentials = os.getenv("GIGACHAT_CREDENTIALS")
        
        if not self.credentials:
            raise ValueError("API ключ GigaChat не найден в .env файле!")
        
        self.giga_client = GigaChat(
            credentials=self.credentials,
            scope="GIGACHAT_API_PERS",
            verify_ssl_certs=False
        )

        self.prompt_template = "Ты — эксперт-методолог. Дай совет студенту.\n\nПРАВИЛА:\n{context}\n\nОШИБКА:\n{error_description}\n\nРекомендация:"

    def _ensure_vector_store(self):
        if self.vector_store is None:
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )

    def generate_recommendations_from_errors(self, errors: List[Dict], graph: Dict) -> List[Dict]:
        """
        Обновленная логика: 
        1. Принимает graph для доступа к текстам.
        2. Разделяет структурные и текстовые правки.
        3. Формирует рекомендации с готовыми исправлениями.
        """
        results = []
        if not errors: return []
        self._ensure_vector_store()

        for error in errors:
            desc = error.get("description", "")
            node_id = error.get("node_id", "unknown")
            
            if not desc or any(x in desc.lower() for x in ["ошибок не найдено", "none"]):
                continue

            # Определяем тип ошибки (флаг структурности)
            # Если в описании есть слова про отсутствие или структуру — это КАРКАС
            is_structural = any(kw in desc.lower() for kw in ["не найден", "отсутствует", "структур", "заключение"])
            
            # Получаем исходный текст из графа, если ошибка не структурная
            original_text = ""
            if not is_structural and node_id in graph.get("nodes", {}):
                original_text = graph["nodes"][node_id].get("raw_text", "")

            # RAG: поиск контекста в методичке
            docs = self.vector_store.similarity_search(desc, k=1)
            context = docs[0].page_content if docs else "Соблюдайте академический стиль и логику изложения."
            sources = list(set([d.metadata.get("source", "Методичка") for d in docs]))

            # Формируем специализированный промпт
            if is_structural:
                # Промпт для структуры
                prompt = (
                    f"Ты — эксперт LISA AI. Студент пропустил важную часть в дипломе.\n"
                    f"МЕТОДИЧКА: {context}\n"
                    f"ОШИБКА: {desc}\n\n"
                    f"ЗАДАЧА: Напиши краткий, мотивирующий совет, что именно нужно добавить "
                    f"и в какой раздел, чтобы пройти проверку. Не пиши 'Рекомендация:' в начале."
                )
            else:
                # Промпт для текста (стиль, ошибки, введение)
                prompt = (
                    f"Ты — научный редактор LISA AI.\n"
                    f"КОНТЕКСТ ИЗ МЕТОДИЧКИ: {context}\n"
                    f"ПРОБЛЕМА В ТЕКСТЕ: {desc}\n"
                    f"ИСХОДНЫЙ ТЕКСТ: '{original_text}'\n\n"
                    f"ЗАДАЧА: \n"
                    f"1. Объясни коротко, почему это ошибка.\n"
                    f"2. ПРЕДЛОЖИ ИСПРАВЛЕННЫЙ ВАРИАНТ ТЕКСТА в академическом стиле.\n"
                    f"Формат ответа:\nСовет: [почему это важно]\nИсправленный текст: [твой вариант]"
                )

            # Вызов GigaChat
            try:
                with GigaChat(
                    credentials=self.credentials,
                    scope="GIGACHAT_API_PERS",
                    verify_ssl_certs=False
                ) as giga:
                    response = giga.chat(prompt)
                    advice = response.choices[0].message.content.strip()
            except Exception as e:
                advice = f"Не удалось сгенерировать совет. Ошибка: {str(e)}"

            # 6. Собираем результат
            results.append({
                "node_id": node_id,
                "error_description": desc,
                "recommendation": advice,
                "suggestion": advice,
                "is_structural": is_structural,
                "sources": sources
            })
            
            time.sleep(1.2) 

        return results

    def add_manual_rules(self):
        """Наполнение базы (вызывать один раз при первом запуске)."""
        self._ensure_vector_store()
        try:
            if self.client.count(collection_name=self.collection_name).count > 0: return
        except: pass

        rules = [
            Document(page_content="Метод 'Карта гипотез' А. Бындю: требует связи технических решений с бизнес-целями и метриками.", metadata={"source": "Бындю"}),
            Document(page_content="Принципы 'Nudge' Р. Талера: дизайн интерфейса должен подталкивать пользователя к целевому действию.", metadata={"source": "Талер"}),
            Document(page_content="Стратегия 'Игра на победу' А. Лафли: требует четкого определения 'Где играть' и 'Как побеждать'.", metadata={"source": "Лафли"})
        ]
        self.vector_store.add_documents(rules)