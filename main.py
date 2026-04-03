import io
import uvicorn
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from core.parser import ThesisParser
from core.critic import CriticManager
from core.generator_giga import GeneratorManager 
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv("config.env")
# Load env locally, fallback to Streamlit secrets in cloud

app = FastAPI(title="LISA AI Thesis Critic API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

parser = None
critic = None
generator = None

@app.on_event("startup")
async def startup_event():
    global parser, critic, generator
    print("Инициализация систем...")
    try:
        parser = ThesisParser()
        critic = CriticManager()
        generator = GeneratorManager()
        generator.add_manual_rules()
        print("Все системы успешно запущены")
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА ПРИ ЗАПУСКЕ: {e}")
        traceback.print_exc()
        # В рабочем окружении здесь можно даже завершить процесс sys.exit(1)

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    # Проверка, что системы инициализированы
    if any(s is None for s in [parser, critic, generator]):
        raise HTTPException(status_code=503, detail="Сервис не готов: ошибка инициализации компонентов")    
    """
    Парсинг -> Критика -> Генерация советов
    """
    if not file.filename.endswith('.docx'):
        raise HTTPException(status_code=400, detail="Допустимы только файлы .docx")

    try:
        file_content = await file.read()
        file_stream = io.BytesIO(file_content)

        # Парсим структуру в граф
        print(f"Обработка файла: {file.filename}")
        graph = parser.parse(file_stream)

        # Ищем нарушения правил (Критик)
        errors = critic.run_all(graph)

        # Генерируем рекомендации (Генератор)
        # передаем и список ошибок, и сам граф для доступа к текстам
        recommendations = generator.generate_recommendations_from_errors(errors, graph)

        # Формируем ответ
        return {
            "filename": file.filename,
            "status": "success",
            "results": {
                "errors": errors,
                "recommendations": recommendations,
                "nodes_count": len(graph["nodes"]),
                "detected_sections": [
                    {"id": n["id"], "title": n["title"]} 
                    for n in graph["nodes"].values() if n["type"] == "SECTION"
                ]
            }
        }

    except Exception as e:
        print(f"Ошибка при анализе: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

@app.get("/health")
async def health_check():
    """Проверка жизни сервиса"""
    return {
        "status": "online",
        "components": {
            "parser": "ok",
            "critic": "ok",
            "generator": "ok"
        }
    }

if __name__ == "__main__":
    # Если запуск из-под Docker 0.0.0.0
    uvicorn.run(app, host="127.0.0.1", port=8000)
