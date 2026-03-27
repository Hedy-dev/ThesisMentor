from .base import BaseRule
from gigachat import GigaChat
from typing import List, Dict
import os

class BusinessMethodologyRule(BaseRule):
    """
    Проверка на соответствие бизнес-методологиям: 
    Карта гипотез (Бындю), Nudge (Талер), Игра на победу (Лафли).
    """
    def __init__(self, model, tokenizer, device):
        super().__init__(model, tokenizer, device)
        self.credentials = os.getenv("GIGACHAT_CREDENTIALS")

    def evaluate(self, graph: Dict) -> List[Dict]:
        errors = []
        # Собираем весь текст работы (ограниченно), чтобы понять контекст
        full_text = ""
        for node in graph["nodes"].values():
            if node["type"] == "PARAGRAPH":
                full_text += (node.get("raw_text", "") + " ")
                if len(full_text) > 4000: break 

        prompt = f"""
        Ты — эксперт по бизнес-стратегиям. Проверь текст бизнес-тезиса на соответствие методологиям:
        1. Карта гипотез (А. Бындю): связаны ли технические фичи с бизнес-метриками?
        2. Стратегия 'Игра на победу' (А. Лафли): четко ли определено 'Где играть' (рынок) и 'Как побеждать' (УТП)?
        
        ТЕКСТ: {full_text}
        
        ЗАДАЧА: Если в тексте упоминаются технические решения, но нет их связи с деньгами или метриками — это ошибка по Бындю.
        Если неясно, на каком конкретно рынке работает стартап — это ошибка по Лафли.
        
        ФОРМАТ ОТВЕТА:
        ОШИБКА: [Есть/Нет]
        ОПИСАНИЕ: [В чем именно нарушение]
        МЕТОДИКА: [Имя автора]
        """

        with GigaChat(credentials=self.credentials, scope="GIGACHAT_API_PERS", verify_ssl_certs=False) as giga:
            try:
                res = giga.chat(prompt)
                content = res.choices[0].message.content
                if "ОШИБКА: Есть" in content:
                    errors.append({
                        "node_id": graph.get("root_id", "root"),
                        "description": content.split("ОПИСАНИЕ:")[-1].strip(),
                        "error_status": "found"
                    })
            except Exception as e:
                print(f"Ошибка бизнес-анализа: {e}")
        
        return errors