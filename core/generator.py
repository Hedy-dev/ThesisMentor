import os
from typing import List, Dict
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter

class GeneratorManager:
    def __init__(self, db_path: str = "./vector_db", collection_name: str = "thesis_kb"):
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Инициализация отечественной модели эмбеддингов
        self.embeddings = HuggingFaceEmbeddings(
            model_name="cointegrated/rubert-tiny2",
            model_kwargs={'device': 'cuda'} 
        )
        
        # Подключение к локальной векторной БД Qdrant
        self.vector_store = Qdrant.from_existing_collection(
            embedding=self.embeddings,
            collection_name=self.collection_name,
            path=self.db_path,
        )
        
        # Инициализация LLM
        model_id = "ai-forever/ruGPT-3.5-13B" 
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=150,
            temperature=0.7,
            repetition_penalty=1.15
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        # Шаблон промпта для генератора
        self.prompt_template = PromptTemplate(
            input_variables=["context", "error_description"],
            template=(
                "Опираясь на правила написания студенческих работ: {context}\n\n"
                "Студент допустил следующую ошибку в структуре: {error_description}\n\n"
                "Дай краткую и четкую рекомендацию, как исправить эту ошибку:\n"
            )
        )

    def ingest_documents(self, pdf_paths: List[str]):
        """Парсинг PDF через Docling и загрузка в Qdrant с чанкингом"""
        converter = DocumentConverter()
        documents = []
        
        # Настроки
        # chunk_size - максимальный размер куска (в символах)
        # chunk_overlap - размер "перекрытия между кусками, чтобы не терять контекст на стыках
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        
        for path in pdf_paths:
            print(f"Парсинг документа: {path}")
            result = converter.convert(path)
            # Экспортируем в Markdown
            text = result.document.export_to_markdown()
            
            # разбиение текста на чанки
            chunks = text_splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                documents.append(Document(
                    page_content=chunk, 
                    metadata={"source": path, "chunk": i}
                ))
                
        print(f"Сохранение {len(documents)} фрагментов в БД Qdrant")
        # Сохраняем/обновляем коллекцию
        Qdrant.from_documents(
            documents,
            self.embeddings,
            path=self.db_path,
            collection_name=self.collection_name
        )
        print("База знаний обновлена")

    # def generate_recommendation(self, error_description: str) -> str:
    #     """Поиск контекста в RAG и генерация ответа"""
    #     # Проверка на отсутствие ошибки
    #     if not error_description or error_description.strip().lower() in ["ошибок не найдено", "none", ""]:
    #         return "Отличная работа! Ошибок не найдено, рекомендации не требуются."

    #     # Поиск релевантной информации в учебниках
    #     docs = self.vector_store.similarity_search(error_description, k=3)
        
    #     if not docs:
    #         context = "Специфических правил в базе не найдено. Руководствуйтесь общими требованиями ГОСТ."
    #     else:
    #         context = "\n".join([doc.page_content for doc in docs])
            
    #     # Формируем промпт и вызываем LLM
    #     prompt = self.prompt_template.format(context=context, error_description=error_description)
    #     response = self.llm.invoke(prompt)
        
    #     # Очистка вывода
    #     clean_response = response.replace(prompt, "").strip()
    #     return clean_response
    def generate_recommendations_from_errors(self, errors: List[Dict]) -> List[Dict]:
        """
        Принимает список ошибок от Критика, ищет контекст в RAG и возвращает рекомендации.
        """
        results = []
        if not errors:
            return []

        for error in errors:
            desc = error.get("description", "")
            node_id = error.get("node_id", "unknown")
            
            # Проверка на "пустую" ошибку (если Критик ничего не нашел)
            if not desc or desc.strip().lower() in ["ошибок не найдено", "none", "ok"]:
                continue # рекомендация не нужна

            # Поиск по базе (RAG)
            # Ищем топ-3 релевантных куска 
            docs = self.vector_store.similarity_search(desc, k=3)
            
            if not docs:
                # Если в базе нет ничего похожего на эту ошибку
                context = "Специфических методологических правил в базе не найдено. Используйте общепринятые стандарты логики и оформления ВКР."
                sources = ["Общие требования"]
            else:
                context = "\n".join([d.page_content for d in docs])
                sources = list(set([d.metadata.get('source', 'Неизвестный источник') for d in docs]))
            
            # Генерация через LLM
            prompt = self.prompt_template.format(context=context, error_description=desc)
            raw_advice = self.llm.invoke(prompt)
            
            # Очистка вывода (универсальный способ)
            # Сначала убираем сам промпт, если модель его повторила
            clean_advice = raw_advice.replace(prompt, "").strip()
            # Если в промпте было слово "Рекомендация:", берем текст после него
            if "Рекомендация:" in clean_advice:
                clean_advice = clean_advice.split("Рекомендация:")[-1].strip()
            
            results.append({
                "node_id": node_id,
                "error_description": desc,
                "recommendation": clean_advice,
                "sources": sources
            })
            
        return results