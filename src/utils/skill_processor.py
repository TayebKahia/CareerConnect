import re
import json
import string
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Any
import os
from src.utils.helpers import debug_log

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Define stop words and custom filter words
stop_words = set(stopwords.words('english'))
custom_filter_words = {'additionally', 'also', 'furthermore',
                       'moreover', 'including', 'like', 'career', 'etc'}


def clean_text(text: str) -> str:
    """Clean and tokenize input text."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    return " ".join([token for token in tokens if token not in stop_words])


def is_meaningful(phrase: str) -> bool:
    """Check if a phrase is meaningful enough to be considered."""
    tokens = [t.lower() for t in word_tokenize(phrase) if t.isalpha()]
    if not tokens:
        return False
    if any(token in custom_filter_words for token in tokens):
        return False
    if len(tokens) == 1 and tokens[0] in stop_words:
        return False
    if sum(1 for t in tokens if t in stop_words)/len(tokens) > 0.5:
        return False
    return True


class SkillProcessor:
    def __init__(self, model_name: str = "sentence-transformers/msmarco-distilbert-base-v4"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        # Go up from src/utils/ to project root, then 'models'
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(project_root, "models")
        debug_log(f"Loading data from directory: {data_dir}")

        # Load embeddings
        embeddings_path = os.path.join(
            data_dir, f"onet_concept_embeddings_{model_name.replace('/', '_')}.npz")
        debug_log(f"Loading embeddings from: {embeddings_path}")
        data = np.load(embeddings_path)
        self.main_embeddings = data['main']
        self.abbr_embeddings = data['abbr']

        # Load processed concepts
        concepts_path = os.path.join(
            project_root, "data", "processed", f"processed_onet_concepts_{model_name.replace('/', '_')}.json")
        debug_log(f"Loading concepts from: {concepts_path}")
        with open(concepts_path, "r", encoding="utf-8") as f:
            self.processed_concepts = json.load(f)

    def process_text(self, text: str, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Process input text and return matched concepts with scores."""
        debug_log("Processing text input...")
        # Clean and tokenize text
        cleaned_text = clean_text(text)
        tokens_clean = word_tokenize(cleaned_text)

        # Generate candidate phrases using n-grams
        candidate_phrases = []
        for n in [3, 2, 1]:
            for gram in ngrams(tokens_clean, n):
                phrase = " ".join(gram)
                if phrase.strip() and is_meaningful(phrase):
                    candidate_phrases.append(phrase)
        candidate_phrases = list(set(candidate_phrases))
        debug_log(f"Generated {len(candidate_phrases)} candidate phrases")

        # Get embeddings for candidate phrases
        candidate_embeddings = self.model.encode(
            candidate_phrases, convert_to_numpy=True)

        # Match candidates to concepts
        recognized_candidates = []
        for i, cand_emb in enumerate(candidate_embeddings):
            sim_main = cosine_similarity([cand_emb], self.main_embeddings)[0]
            sim_abbr = cosine_similarity([cand_emb], self.abbr_embeddings)[0]
            best_scores = np.maximum(sim_main, sim_abbr)
            best_idx = best_scores.argmax()
            best_score = best_scores[best_idx]

            if best_score >= threshold:
                concept = self.processed_concepts[best_idx]
                source = "main" if sim_main[best_idx] >= sim_abbr[best_idx] else "abbr"
                phrase = candidate_phrases[i]
                # Keep original type (technology_name) to match /api/recommend expectations
                recognized_candidates.append({
                    "name": concept["name"],
                    # Keep original type (technology_name)
                    "type": concept["type"],
                    "similarity": float(best_score)
                })

        debug_log(f"Found {len(recognized_candidates)} matching concepts")
        # Filter and sort candidates
        filtered_candidates = self._filter_candidates(recognized_candidates)
        debug_log(f"After filtering: {len(filtered_candidates)} concepts")
        return filtered_candidates

    def _filter_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and deduplicate candidates, keeping highest scores."""
        # Sort by similarity score
        candidates.sort(key=lambda x: x["similarity"], reverse=True)

        # Group by name and keep highest score
        unique_candidates = {}
        for candidate in candidates:
            name = candidate["name"]
            if name not in unique_candidates or candidate["similarity"] > unique_candidates[name]["similarity"]:
                unique_candidates[name] = candidate

        # Convert back to list and sort by similarity
        filtered = list(unique_candidates.values())
        filtered.sort(key=lambda x: x["similarity"], reverse=True)

        return filtered
