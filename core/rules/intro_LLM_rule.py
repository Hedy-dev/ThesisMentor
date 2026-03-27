import time
import os
import torch
import torch.nn.functional as F
from typing import List, Dict
from .base import BaseRule
from gigachat import GigaChat

class IntroLLMRule(BaseRule):
    """
    Гибридное правило: 
    1. Находит практические главы через NLI (анализ смысла).
    2. Проверяет содержательность Введения через GigaChat.
    """
    
    def __init__(self, model, tokenizer, device):
        super().__init__(model, tokenizer, device)
        self.credentials = os.getenv("GIGACHAT_CREDENTIALS")
        
        # Пороги для NLI (поиск практики)
        self.threshold_practice = 0.7
        self.entailment_id = 0 # Для rubert-base-cased-nli-threeway

        if not self.credentials:
            raise ValueError("API ключ GigaChat не найден в .env файле!")

    def nli_score(self, premise: str, hypothesis: str) -> float:
        """Расчет логического следования (для классификации типа раздела)."""
        inputs = self.tokenizer(
            premise, hypothesis, 
            truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=1)
        return probs[0][self.entailment_id].item()

    def _get_node_text(self, graph: Dict, node_id: str, limit: int = 3000) -> str:
        """Собирает текст раздела для отправки в LLM."""
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
        return full_text[:limit]

    def _detect_practice_nodes(self, graph: Dict) -> List[str]:
        """Ищет ID практических разделов (через ключевые слова + NLI)."""
        practice_ids = []
        practice_keywords = ["ЭКОНОМИЧ", "РЕЗУЛЬТАТ", "МОДЕЛЬ", "UNIT", "ГИПОТЕЗ", "ПРОДУКТ", "РАЗРАБОТКА"]
        
        for nid, n in graph["nodes"].items():
            if n["type"] != "SECTION": continue
            title_upper = n["title"].upper()
            
            # 1. Сначала проверяем по ключевым словам (быстро)
            if any(k in title_upper for k in practice_keywords):
                practice_ids.append(nid)
                continue
            
            # 2. Если по названию не ясно, проверяем смысл первого параграфа через NLI
            children = n.get("children", [])
            if children:
                first_child = graph["nodes"].get(children[0])
                if first_child and first_child.get("type") == "PARAGRAPH":
                    score = self.nli_score(
                        first_child.get("clean_text", ""),
                        "В данном разделе описывается практическая реализация, расчеты или результаты проекта."
                    )
                    if score > self.threshold_practice:
                        practice_ids.append(nid)
        
        return list(set(practice_ids))

    def evaluate(self, graph: Dict) -> List[Dict]:
        errors = []
        
        # 1. Ищем Введение
        intro_id = next((nid for nid, n in graph["nodes"].items() if "ВВЕДЕНИЕ" in n["title"].upper()), None)
        if not intro_id:
            return []

        intro_text = self._get_node_text(graph, intro_id)

        # 2. Ищем практические главы (умный поиск)
        practice_nodes = self._detect_practice_nodes(graph)
        if not practice_nodes:
            return []

        # 3. Анализ через GigaChat
        with GigaChat(credentials=self.credentials, scope="GIGACHAT_API_PERS", verify_ssl_certs=False) as giga:
            for p_id in practice_nodes:
                p_title = graph["nodes"][p_id]["title"]
                p_text = self._get_node_text(graph, p_id)
                
                prompt = (
                    f"Ты — строгий академический рецензент бизнес-проектов.\n"
                    f"ЗАДАЧА: Проверить, отражены ли РЕЗУЛЬТАТЫ практической главы во Введении.\n\n"
                    f"ВВЕДЕНИЕ: {intro_text}\n\n"
                    f"ПРАКТИЧЕСКАЯ ГЛАВА '{p_title}': {p_text}\n\n"
                    f"КРИТЕРИЙ: Введение в бизнес-проекте обязано содержать конкретику: "
                    f"какие гипотезы проверены, какие метрики (CAC, LTV, ROI) получены, какой продукт создан.\n"
                    f"Если во введении только общие слова (история, описание рынка), а конкретных итогов "
                    f"из главы '{p_title}' НЕТ — ответь НЕТ.\n"
                    f"Если итоги/результаты кратко упомянуты — ответь ДА.\n\n"
                    f"ОТВЕТЬ СТРОГО В ФОРМАТЕ:\nРЕЗУЛЬТАТ: [ДА или НЕТ]\nПРИЧИНА: [почему, будь краток]"
                )
                
                try:
                    response = giga.chat(prompt)
                    content = response.choices[0].message.content.strip()
                    
                    print(f"🔍 Анализ для '{p_title}':\n{content}\n")
                    
                    if "РЕЗУЛЬТАТ: НЕТ" in content.upper() or "РЕЗУЛЬТАТ: [НЕТ]" in content.upper():
                        reason = content.split("ПРИЧИНА:")[-1].strip() if "ПРИЧИНА:" in content else "Результаты не отражены."
                        errors.append({
                            "node_id": p_id,
                            "description": f"Введение носит слишком теоретический характер. В нем не отражены результаты раздела '{p_title}': {reason}",
                            "error_status": "found"
                        })
                    
                    time.sleep(1)
                except Exception as e:
                    print(f"Ошибка LLM при проверке главы {p_title}: {e}")

        return errors
