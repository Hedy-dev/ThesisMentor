from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict
from .rules.rule_1 import PracticeIntroRule

class CriticManager:
    def __init__(self):
        model_name = "cointegrated/rubert-base-cased-nli-threeway"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        
        # Регистрируем правила
        self.rules = [
            PracticeIntroRule(self.model, self.tokenizer, self.device),
            # Сюда просто добавляем новые классы правил
        ]

    def run_all(self, graph: Dict) -> List[Dict]:
        all_errors = []
        for rule in self.rules:
            all_errors.extend(rule.evaluate(graph))
        return all_errors