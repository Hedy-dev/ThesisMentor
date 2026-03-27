import time
import os
from typing import List, Dict
from .base import BaseRule
from gigachat import GigaChat
from dotenv import load_dotenv

load_dotenv()

class IntroLLMRule(BaseRule):
    """
    Правило на базе LLM: Проверяет, насколько содержательно результаты 
    практических глав отражены во Введении, игнорируя пустые формальные фразы.
    """
    
    def __init__(self, model, tokenizer, device):
        super().__init__(model, tokenizer, device)
        self.credentials = os.getenv("GIGACHAT_CREDENTIALS")
        
        if not self.credentials:
            raise ValueError("API ключ GigaChat не найден в .env файле!")
        
    def _get_node_text(self, graph: Dict, node_id: str, limit: int = 2000) -> str:
        texts = []
        node = graph["nodes"].get(node_id)
        if not node: return ""
        
        for child_id in node.get("children", []):
            child = graph["nodes"].get(child_id)
            if child and child.get("type") == "PARAGRAPH":
                texts.append(child.get("raw_text", ""))
            elif child and child.get("type") in ["SECTION", "SUBSECTION"]:
                texts.append(self._get_node_text(graph, child_id))
        
        full_text = " ".join(texts)
        return full_text[:limit] # Ограничиваем для контекста LLM

    def evaluate(self, graph: Dict) -> List[Dict]:
        errors = []
        
        # 1. Находим Введение
        intro_id = next((nid for nid, n in graph["nodes"].items() if "ВВЕДЕНИЕ" in n["title"].upper()), None)
        if not intro_id:
            return []

        intro_text = self._get_node_text(graph, intro_id)

        # Находим практические разделы (например, по ключевым словам из методички)
        practice_keywords = ["ЭКОНОМИЧ", "РЕЗУЛЬТАТ", "МОДЕЛЬ", "UNIT", "ГИПОТЕЗ", "ПРОДУКТ"]
        practice_nodes = [
            nid for nid, n in graph["nodes"].items() 
            if n["type"] == "SECTION" and any(k in n["title"].upper() for k in practice_keywords)
        ]

        if not practice_nodes:
            return []

        # Проверка через GigaChat
        with GigaChat(credentials=self.credentials, scope="GIGACHAT_API_PERS", verify_ssl_certs=False) as giga:
            for p_id in practice_nodes:
                p_title = graph["nodes"][p_id]["title"]
                p_text = self._get_node_text(graph, p_id)
                prompt = (
                    f"Ты — строгий рецензент LISA AI. Сравни текст Введения и Практической главы.\n"
                    f"ВВЕДЕНИЕ: {intro_text}\n\n"
                    f"ПРАКТИКА '{p_title}': {p_text}\n\n"
                    f"КРИТЕРИЙ: Введение в бизнес-тезисе ДОЛЖНО содержать краткое резюме ключевых результатов "
                    f"практической части (например: 'Разработана стратегия маркетинга для компании X, "
                    f"прогнозный рост выручки — 15%').\n"
                    f"ЗАДАЧА: Если введение содержит только общую теорию и описание рынка, но НЕ УПОМИНАЕТ "
                    f"результаты конкретно этой работы — ответь 'НЕТ'.\n"
                    f"ОТВЕТЬ В ФОРМАТЕ:\nРезультат: [ДА/НЕТ]\nПричина: [кратко]"
                )
                try:
                    response = giga.chat(prompt)
                    content = response.choices[0].message.content
                    
                    if "Результат: НЕТ" in content.upper() or "Результат: [НЕТ]" in content.upper():
                        reason = content.split("Причина:")[-1].strip()
                        errors.append({
                            "node_id": p_id,
                            "description": f"Раздел '{p_title}' не отражен во введении. {reason}",
                            "error_status": "found"
                        })
                    
                    time.sleep(1)
                except Exception as e:
                    print(f"Ошибка LLM правила: {e}")

        return errors