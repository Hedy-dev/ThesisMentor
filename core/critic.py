import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .rules.structure_rule  import MandatorySectionsRule
from .rules.style_rule import PersonalPronounsRule
from .rules.intro_LLM_rule import IntroLLMRule

class CriticManager:
    def __init__(self):
        model_name = "cointegrated/rubert-base-cased-nli-threeway"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        
        # Регистрируем правила
        self.rules = [
            # PracticeIntroRule(self.model, self.tokenizer, self.device),
            IntroLLMRule(self.model, self.tokenizer, self.device),
            PersonalPronounsRule(self.model, self.tokenizer, self.device),
            MandatorySectionsRule(self.model, self.tokenizer, self.device)
        ]

    def run_all(self, graph: Dict) -> List[Dict]:
        all_errors = []
        for rule in self.rules:
            all_errors.extend(rule.evaluate(graph))
        return all_errors
