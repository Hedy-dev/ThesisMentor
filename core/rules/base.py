from abc import ABC, abstractmethod
from typing import List, Dict

class BaseRule(ABC):
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @abstractmethod
    def evaluate(self, graph: Dict) -> List[Dict]:
        """Возвращает список ошибок в формате ..."""
        pass
