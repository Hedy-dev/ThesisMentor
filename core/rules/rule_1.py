import torch
import torch.nn.functional as F
from typing import List, Dict
from .base import BaseRule

class PracticeIntroRule(BaseRule):
    """
    Правило 1: Проверка корректности отражения практических результатов во введении.
    Использует NLI модель для поиска практических секций и их сопоставления с текстом введения.
    """
    
    def __init__(self, model, tokenizer, device):
        super().__init__(model, tokenizer, device)
        # Пороги срабатывания (можно вынести в конфиг)
        self.threshold_practice = 0.7
        self.threshold_reflection = 0.6
        self.entailment_id = 0  # Для модели rubert-base-cased-nli-threeway

    def nli_score(self, premise: str, hypothesis: str) -> float:
        """Расчет логического следования между двумя текстами."""
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=1)
        return probs[0][self.entailment_id].item()

    def evaluate(self, graph: Dict) -> List[Dict]:
        errors = []
        
        # 1. Поиск введения
        intro_node_id = self._find_intro_node(graph)
        if not intro_node_id:
            # Если введения нет, это отдельная системная ошибка, 
            # но для данного правила мы просто прекращаем проверку
            return errors

        intro_paragraphs = self._get_paragraphs(graph, intro_node_id)
        
        # 2. Идентификация практических разделов
        practice_sections = self._detect_practice_sections(graph)

        # 3. Проверка отражения каждой практической секции во введении
        for section in practice_sections:
            section_id = section["id"]
            section_title = section["title"]
            section_paragraphs = self._get_paragraphs(graph, section_id)

            best_match_score = 0.0
            
            # Сравниваем каждый параграф практики с каждым параграфом введения
            for sp in section_paragraphs:
                for ip in intro_paragraphs:
                    score = self.nli_score(
                        sp["clean_text"], # Тезис из практики
                        ip["clean_text"]  # Подтверждение во введении
                    )
                    if score > best_match_score:
                        best_match_score = score

            # 4. Формирование ошибки, если соответствие ниже порога
            if best_match_score < self.threshold_reflection:
                errors.append({
                    "description": (
                        f"Раздел '{section_title}' классифицирован как практический "
                        f"(score: {section['practice_score']:.2f}), но его результаты "
                        f"недостаточно отражены во введении (сходство: {best_match_score:.2f})."
                    ),
                    "error_status": "found",
                    "node_id": section_id,
                    "start_offset": 0,
                    "end_offset": 0
                })

        return errors

    def _detect_practice_sections(self, graph: Dict) -> List[Dict]:
        """Ищет узлы типа SECTION, которые содержат практический вклад."""
        practice_sections = []
        
        for node_id, node in graph["nodes"].items():
            if node.get("type") != "SECTION":
                continue
            
            # Игнорируем служебные разделы
            title_upper = node["title"].strip().upper()
            if title_upper in ["ВВЕДЕНИЕ", "ЗАКЛЮЧЕНИЕ", "СПИСОК ЛИТЕРАТУРЫ", "СОДЕРЖАНИЕ"]:
                continue

            paragraphs = self._get_paragraphs(graph, node_id)
            if not paragraphs:
                continue

            # Проверяем параграфы раздела на наличие признаков практики
            max_p_score = 0.0
            for p in paragraphs:
                score = self.nli_score(
                    p["clean_text"],
                    "В данном разделе описывается практическая реализация, программный продукт или расчеты."
                )
                max_p_score = max(max_p_score, score)

            if max_p_score > self.threshold_practice:
                practice_sections.append({
                    "id": node_id,
                    "title": node["title"],
                    "practice_score": max_p_score
                })
        
        return practice_sections

    def _get_paragraphs(self, graph: Dict, start_node_id: str) -> List[Dict]:
        """Рекурсивно собирает все текстовые блоки (параграфы) внутри узла."""
        paragraphs = []
        node = graph["nodes"].get(start_node_id)
        
        if not node:
            return []

        for child_id in node.get("children", []):
            child = graph["nodes"].get(child_id)
            if not child:
                continue
                
            if child.get("type") == "PARAGRAPH":
                if child.get("clean_text"):
                    paragraphs.append(child)
            else:
                # Идем глубже в подразделы
                paragraphs.extend(self._get_paragraphs(graph, child_id))
        
        return paragraphs

    def _find_intro_node(self, graph: Dict) -> str:
        """Поиск ID узла 'Введение'."""
        for node_id, node in graph["nodes"].items():
            if node["title"].strip().upper() == "ВВЕДЕНИЕ":
                return node_id
        return None