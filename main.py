import io
import uvicorn
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from core.parser import ThesisParser
from core.critic import CriticManager
from core.generator_giga import GeneratorManager 
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="LISA AI Thesis Critic API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Инициализация систем")
try:
    parser = ThesisParser()
    critic = CriticManager()
    generator = GeneratorManager()
    
    # Наполняем базу знаний правилами из методички (если она пустая)
    generator.add_manual_rules()
    
    print("Все системы запущены")
except Exception as e:
    print(f"ОШИБКА: {e}")
    traceback.print_exc()

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
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
