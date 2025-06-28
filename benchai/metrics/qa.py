# benchai/metrics/qa.py

from typing import Union
from sentence_transformers import SentenceTransformer, util as st_util
from rouge_score import rouge_scorer
from benchai.utils import fuzzy_score

class QAValidator:
    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def score(self, actual: str, expected: Union[str, list]) -> tuple[float, bool, str]:
        actual = actual.strip().lower()
        if isinstance(expected, str):
            expected_list = [expected.strip().lower()]
        else:
            expected_list = [e.strip().lower() for e in expected]

        max_score = 0.0
        best_feedback = ""

        for expected in expected_list:
            rouge_score = self.rouge.score(expected, actual)['rougeL'].fmeasure
            fuzzy = fuzzy_score(actual, expected)

            # Semantic similarity
            embeddings = self.embedder.encode([actual, expected], convert_to_tensor=True)
            cosine_sim = st_util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

            # Final blend
            final_score = (
                0.5 * cosine_sim +     # semantic
                0.3 * rouge_score +    # structure/phrasing
                0.2 * fuzzy            # token/character overlap
            )

            if final_score > max_score:
                max_score = final_score
                best_feedback = (
                    f"Cosine: {cosine_sim:.2f}, ROUGE-L: {rouge_score:.2f}, Fuzzy: {fuzzy:.2f}"
                )

        passed = max_score >= 0.65  # adjustable based on use case

        return max_score, passed, best_feedback
