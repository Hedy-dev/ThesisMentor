import re
from typing import List, Dict
from .base import BaseRule

class PersonalPronounsRule(BaseRule):
    """
    Проверка на использование личных местоимений (я, мой, меня и т.д.).
    Академический стиль требует безличного изложения.
    """
    def __init__(self, model, tokenizer, device):
        super().__init__(model, tokenizer, device)
        # Список стоп-слов (можно расширять)
        self.forbidden_patterns = [
            r'\bя\b', r'\bмой\b', r'\bменя\b', r'\bмне\b', 
            r'\bмною\b', r'\bавтор (считает|сделал|разработал)\b'
        ]

    def evaluate(self, graph: Dict) -> List[Dict]:
        errors = []
        for node_id, node in graph["nodes"].items():
            if node.get("type") == "PARAGRAPH":
                text = node.get("clean_text", "")
                for pattern in self.forbidden_patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        # Находим родительский раздел для понятного вывода
                        parent_title = "Неизвестный раздел"
                        parent_id = node.get("parent_id")
                        if parent_id in graph["nodes"]:
                            parent_title = graph["nodes"][parent_id]["title"]

                        errors.append({
                            "description": f"В разделе '{parent_title}' обнаружено личное местоимение или авторское 'я'. Используйте безличные конструкции (например, 'было разработано' вместо 'я разработал').",
                            "error_status": "found",
                            "node_id": node_id
                        })
                        break # Достаточно одной ошибки на параграф
        return errors