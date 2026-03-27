from typing import List, Dict
from .base import BaseRule

class MandatorySectionsRule(BaseRule):
    """
    Проверка наличия обязательных структурных элементов.
    """
    def evaluate(self, graph: Dict) -> List[Dict]:
        errors = []
        mandatory = {
            "ВВЕДЕНИЕ": False,
            "ЗАКЛЮЧЕНИЕ": False,
            "СПИСОК": False # Для "Список литературы" или "Список источников"
        }
        
        for node in graph["nodes"].values():
            if node.get("type") == "SECTION":
                title_upper = node["title"].upper()
                for key in mandatory:
                    if key in title_upper:
                        mandatory[key] = True
        
        for section, found in mandatory.items():
            if not found:
                errors.append({
                    "description": f"В структуре работы не найден обязательный раздел: {section}.",
                    "error_status": "found",
                    "node_id": graph.get("root_id", "root")
                })
        return errors