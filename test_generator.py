import os
import sys
from typing import List, Dict
# Указываем путь к кэшу моделей ПЕРЕД загрузкой библиотек
os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\lisa\models_cache"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.generator import GeneratorManager

def check_data_files(files: List[str]):
    """Проверяет наличие файлов в папке data."""
    missing = [f for f in files if not os.path.exists(f)]
    return missing

def main():
    print("Инициализация системы Генератора (LISA AI)")
    
    # Используем названия файлов из папки data
    data_dir = "data"
    my_library = [
        os.path.join(data_dir, "08 - Карта гипотез _ А. В. Бындю.pdf"),
        os.path.join(data_dir, "14_Игра_на_победу_Как_стратегия_работает_на_самом_деле_А_Лафли.pdf"),
        os.path.join(data_dir, "Byindyu_A ._ Antihrupkost_V_It.a4.pdf"),
        os.path.join(data_dir, "NUDGE_Архитектура_выбора_Р_Талер,_К_Санстейн_1.pdf"),
        os.path.join(data_dir, "Предсказуемая_иррациональность_Ариели.pdf")
    ]

    # Инициализируем менеджер (загрузка моделей эмбеддингов и LLM)
    generator = GeneratorManager(db_path="./vector_db", collection_name="thesis_kb")

    # Проверка и индексация базы знаний
    if not os.path.exists("./vector_db") or not os.listdir("./vector_db"):
        print("\n Индексация базы знаний")
        missing_files = check_data_files(my_library)
        
        if missing_files:
            print(f" Ошибка: Следующие файлы не найдены в папке {data_dir}:")
            for f in missing_files: print(f"  - {f}")
            print("\nПожалуйста, разместите PDF-файлы в папке и запустите снова.")
            return
        
        print(f"Найдено {len(my_library)} книг. Парсинг через Docling")
        generator.ingest_documents(my_library)
        print(" База знаний готова и сохранена в ./vector_db")
    else:
        print("\n Обнаружена существующая векторная база знаний. Пропускаем индексацию.")

    # Эмуляция различных сценариев от модели Критика
    # Сценарий 1: Реальная структурная ошибка (несоответствие практики и введения)
    # Сценарий 2: Ошибок нет (как часто бывает после исправлений)
    # Сценарий 3: Технический пропуск
    
    test_scenarios = [
        {
            "case_name": "Несоответствие Практики и Введения (Кейс Бындю)",
            "critic_output": [
                {
                    "description": "Раздел 'Разработка карты гипотез' классифицирован как практический (score: 0.92), но его результаты недостаточно отражены во введении (сходство: 0.35).",
                    "error_status": "found",
                    "node_id": "node_section_001"
                }
            ]
        },
        {
            "case_name": "Идеальная работа",
            "critic_output": [] # Пустой список — ошибок нет
        },
        {
            "case_name": "Ошибок не найдено (строковое уведомление)",
            "critic_output": [{"description": "Ошибок не найдено", "node_id": "root"}]
        }
    ]

    # Запуск тестирования
    print("\n Генерация рекомендаций по кейсам")

    for scenario in test_scenarios:
        print(f"\n ТЕСТ: {scenario['case_name']}")

        recommendations = generator.generate_recommendations_from_errors(scenario['critic_output'])
        
        if not recommendations:
            print("Результат: Критик не предоставил ошибок. Рекомендаций нет.")
        else:
            for rec in recommendations:
                print(f"ID Узла: {rec['node_id']}")
                print(f"Найдено: {rec['error_description']}")
                print(f"Рекомендация ИИ:\n{rec['recommendation']}")
                print(f"Источники контекста: {', '.join(rec['sources'])}")
        
        print("-" * 60)

if __name__ == "__main__":
    main()