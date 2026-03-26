import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from core.parser import ThesisParser
from core.critic import CriticManager

app = FastAPI(title="LISA AI Thesis Critic API")

# Настройка CORS для работы с Streamlit или фронтендом
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализируем компоненты один раз при старте сервера
# Это экономит ресурсы, так как модель NLI тяжелая
parser = ThesisParser()
critic = CriticManager()

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """
    Основной эндпоинт: 
    1. Принимает .docx файл
    2. Строит граф структуры через ThesisParser
    3. Проверяет правила через CriticManager
    4. Генерирует рекомендации на основе найденных ошибок
    """
    if not file.filename.endswith('.docx'):
        raise HTTPException(status_code=400, detail="Допустимы только файлы .docx")

    try:
        # Читаем содержимое файла в память
        file_content = await file.read()
        file_stream = io.BytesIO(file_content)

        # Получаем структуру: {"root_id": ..., "nodes": {id: node_dict}}
        graph = parser.parse(file_stream)

        # Прогоняет Rule 1 (и все будущие правила), возвращает список ошибок
        # Ошибка содержит node_id практического раздела (заголовка)
        errors = critic.run_all(graph)

        # Формируем рекомендации, привязанные к конкретным узлам
        recommendations = []
        for error in errors:
            target_node = graph["nodes"].get(error["node_id"])
            section_title = target_node["title"] if target_node else "неизвестного раздела"
            
            recommendations.append({
                "error_id": error["node_id"], # ID узла-заголовка практического раздела
                "suggestion": (
                    f"В разделе '{section_title}' описаны практические результаты. "
                    f"Необходимо добавить краткое резюме этих результатов в главу 'ВВЕДЕНИЕ', "
                    f"чтобы обеспечить целостность работы."
                ),
                "type": "structural_consistency"
            })

        # Возвращаем полный пакет данных для фронтенда
        return {
            "filename": file.filename,
            "status": "success",
            "results": {
                "errors": errors,
                "recommendations": recommendations,
                # Отправляем небольшое превью графа для отладки на фронте
                "nodes_count": len(graph["nodes"]),
                "detected_sections": [
                    {"id": n["id"], "title": n["title"]} 
                    for n in graph["nodes"].values() if n["type"] == "SECTION"
                ]
            }
        }

    except Exception as e:
        # Логируем ошибку для разработки
        print(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при анализе документа: {str(e)}")

@app.get("/health")
async def health_check():
    """Проверка доступности сервиса и статуса модели"""
    return {
        "status": "online",
        "device": critic.device,
        "rules_loaded": [type(r).__name__ for r in critic.rules]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)