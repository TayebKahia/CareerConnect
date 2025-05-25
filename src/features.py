import pandas as pd
import numpy as np
import string
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Ensure required NLTK resources are available.
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")


# Singleton metaclass to ensure only one instance of ConceptMatcher exists
class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ConceptMatcher(metaclass=Singleton):
    def __init__(
        self,
        csv_path="../data/processed/technologies_with_abbreviations.csv",  # Updated path to new CSV with abbreviations
        columns=None,
        model_name="sentence-transformers/msmarco-distilbert-base-v4",
        similarity_threshold_graph=0.7,
        ngram_threshold=0.7,
        filter_similarity_threshold=0.90,
    ):
        if columns is None:
            columns = [
                "LanguageHaveWorkedWith",
                "DatabaseHaveWorkedWith",
                "PlatformHaveWorkedWith",
                "WebframeHaveWorkedWith",
                "EmbeddedHaveWorkedWith",
                "MiscTechHaveWorkedWith",
                "ToolsTechHaveWorkedWith",
            ]
        self.csv_path = csv_path
        self.columns = columns
        self.model_name = model_name
        self.similarity_threshold_graph = similarity_threshold_graph
        self.ngram_threshold = ngram_threshold
        self.filter_similarity_threshold = filter_similarity_threshold

        # Initialize NLTK stop words and custom filter words.
        self.stop_words = set(stopwords.words("english"))
        self.custom_filter_words = {
            "additionally",
            "also",
            "furthermore",
            "moreover",
            "including",
            "like",
            "career",
            "etc",
        }

        # Initialize the SentenceTransformer model.
        self.model = SentenceTransformer(self.model_name)

        # Placeholders for later processing.
        self.tech_data = None  # DataFrame to store technology names and abbreviations
        self.stack_concepts = []  # List of concept dictionaries.
        self.concept_embeddings = None  # Numpy array of concept embeddings.
        self.candidate_phrases = []  # Candidate n‑gram phrases from input text.
        self.candidate_embeddings = None  # Numpy array of candidate embeddings.
        self.recognized_candidates_ngram = (
            []
        )  # Matched candidates with similarity scores.
        self.filtered_by_concept = {}  # Final grouped output after global filtering.

    def clean_text(self, text):
        """
        Lowercase the text and remove punctuation except for hyphens and parentheses.
        """
        # Preserve '-' and parentheses by removing other punctuation.
        punctuation_to_remove = "".join(
            ch for ch in string.punctuation if ch not in "-()"
        )
        text = text.lower().translate(str.maketrans("", "", punctuation_to_remove))
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        return " ".join(tokens)

    def is_meaningful(self, phrase):
        tokens = [t.lower() for t in word_tokenize(phrase) if t.isalpha()]
        if not tokens:
            return False
        if any(token in self.custom_filter_words for token in tokens):
            return False
        if len(tokens) == 1 and tokens[0] in self.stop_words:
            return False
        if (
            tokens
            and sum(1 for t in tokens if t in self.stop_words) / len(tokens) > 0.5
        ):
            return False
        return True

    def load_concepts(self):
        # Load the technology data with abbreviations
        self.tech_data = pd.read_csv(self.csv_path)
        print(f"Loaded technologies data with {len(self.tech_data)} entries")

        # Create concepts from both technology names and their abbreviations
        self.stack_concepts = []

        # Add technology names first
        for _, row in self.tech_data.iterrows():
            tech_name = row["Technology"]
            self.stack_concepts.append(
                {"name": tech_name, "type": "Technology", "original": tech_name}
            )

        # Add abbreviations if they exist
        for _, row in self.tech_data.iterrows():
            if (
                pd.notna(row["abrv"]) and row["abrv"].strip()
            ):  # Check if abbreviation exists and is not empty
                tech_name = row["Technology"]
                abbr = row["abrv"]
                self.stack_concepts.append(
                    {"name": abbr, "type": "Abbreviation", "original": tech_name}
                )

        print(
            f"Total StackOverflow Concepts (including abbreviations): {len(self.stack_concepts)}"
        )

    def generate_concept_embeddings(self, save_embeddings=False, load_if_exists=True):
        """
        Generate embeddings for all technology concepts or load existing ones if available.

        Parameters:
        - save_embeddings: Whether to save newly generated embeddings
        - load_if_exists: Whether to try loading existing embeddings first
        """
        # Get project root directory (assuming the script is in src/features.py)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # Check notebooks directory with absolute path
        filename_notebook = os.path.join(
            project_root,
            "notebooks/tk",
            f"stack_concept_embeddings_{self.model_name.replace('/', '_')}.npy",
        )

        # Try to load existing embeddings if requested
        if load_if_exists:
            tried_paths = []
            try:
                # Try notebook path first
                if os.path.exists(filename_notebook):
                    self.concept_embeddings = np.load(filename_notebook)
                    print(
                        f"Loaded existing concept embeddings from {filename_notebook}"
                    )
                    return
                tried_paths.append(filename_notebook)

                # Then try current directory
                if os.path.exists(filename_current):
                    self.concept_embeddings = np.load(filename_current)
                    print(f"Loaded existing concept embeddings from {filename_current}")
                    return
                tried_paths.append(filename_current)

                # Finally try project root
                if os.path.exists(filename_project_root):
                    self.concept_embeddings = np.load(filename_project_root)
                    print(
                        f"Loaded existing concept embeddings from {filename_project_root}"
                    )
                    return
                tried_paths.append(filename_project_root)

                print(f"No existing embeddings found, generating new ones...")
                print(f"Checked paths: {tried_paths}")
            except Exception as e:
                print(f"Error loading embeddings: {e}. Generating new ones...")
                print(f"Checked paths: {tried_paths}")

        # Generate new embeddings
        print("Generating new concept embeddings...")
        concept_texts = [concept["name"] for concept in self.stack_concepts]
        self.concept_embeddings = self.model.encode(
            concept_texts, convert_to_numpy=True
        )

        if save_embeddings:
            # Save to the most appropriate location - prefer notebooks directory if it exists
            if os.path.exists(os.path.dirname(filename_notebook)):
                save_path = filename_notebook
            else:
                save_path = filename_current

            np.save(save_path, self.concept_embeddings)
            print(f"Concept embeddings saved to {save_path}")

    def prepare_candidate_phrases(self, long_text):
        cleaned_full_text = self.clean_text(long_text)
        tokens_clean = word_tokenize(cleaned_full_text)
        candidate_phrases = []
        for n in [3, 2, 1]:
            for gram in ngrams(tokens_clean, n):
                phrase = " ".join(gram)
                if phrase.strip() and self.is_meaningful(phrase):
                    candidate_phrases.append(phrase)
        self.candidate_phrases = list(set(candidate_phrases))
        print(f"Total candidate phrases generated: {len(self.candidate_phrases)}")

    def vectorized_match_candidates(self):
        self.candidate_embeddings = self.model.encode(
            self.candidate_phrases, convert_to_numpy=True
        )
        similarity_matrix = cosine_similarity(
            self.candidate_embeddings, self.concept_embeddings
        )
        max_similarities = similarity_matrix.max(axis=1)
        max_indices = similarity_matrix.argmax(axis=1)
        valid_indices = np.where(max_similarities >= self.ngram_threshold)[0]
        self.recognized_candidates_ngram = []
        for idx in valid_indices:
            max_sim = max_similarities[idx]
            max_idx = max_indices[idx]
            concept_name = self.stack_concepts[max_idx]["name"]
            concept_type = self.stack_concepts[max_idx]["type"]
            original_name = self.stack_concepts[max_idx].get("original", concept_name)
            phrase = self.candidate_phrases[idx]
            n_val = len(phrase.split())
            tokens_phrase = phrase.split()
            self.recognized_candidates_ngram.append(
                (original_name, concept_type, phrase, max_sim, n_val, tokens_phrase)
            )
        print(
            f"Total recognized candidate matches: {len(self.recognized_candidates_ngram)}"
        )

    def global_filtering(self):
        recognized = sorted(
            self.recognized_candidates_ngram, key=lambda x: x[3], reverse=True
        )
        global_used_words = set()
        filtered_candidates = []
        for candidate in recognized:
            concept_name, concept_type, phrase, score, n_val, tokens_phrase = candidate
            if any(token in global_used_words for token in tokens_phrase):
                continue
            filtered_candidates.append(candidate)
            if score > self.filter_similarity_threshold:
                global_used_words.update(tokens_phrase)
        self.filtered_by_concept = {}
        for (
            concept_name,
            concept_type,
            phrase,
            score,
            n_val,
            tokens_phrase,
        ) in filtered_candidates:
            self.filtered_by_concept.setdefault(
                concept_name, {"type": concept_type, "phrases": []}
            )
            self.filtered_by_concept[concept_name]["phrases"].append(
                (phrase, score, n_val, tokens_phrase)
            )
        print("Global filtering completed.")

    def get_recognized_technologies(self):
        """Get a simple list of recognized technologies."""
        return list(self.filtered_by_concept.keys())

    def get_technologies_with_scores(self):
        """
        Get list of technologies with their highest similarity scores and other threshold information.

        Returns:
        - List of dictionaries with technology name, similarity score, and threshold information
        """
        result = []

        for tech_name, info in self.filtered_by_concept.items():
            # Get the highest similarity score for this technology
            highest_score = (
                max([score for _, score, _, _ in info["phrases"]])
                if info["phrases"]
                else 0.0
            )

            # Get the threshold values used in matching
            tech_info = {
                "name": tech_name,
                "similarity_score": round(highest_score, 4),
                "thresholds": {
                    "ngram_threshold": self.ngram_threshold,
                    "filter_similarity_threshold": self.filter_similarity_threshold,
                },
                "type": info["type"],
                "phrases": [
                    {"text": phrase, "score": round(score, 4)}
                    for phrase, score, _, _ in info["phrases"]
                ],
            }

            result.append(tech_info)

        # Sort by similarity score (highest first)
        result.sort(key=lambda x: x["similarity_score"], reverse=True)
        return result

    def process_text(self, long_text):
        """
        Process input text to identify technology concepts in one function call.

        Parameters:
        - long_text: The text to analyze for technology concepts

        Returns:
        - List of recognized technology concepts with similarity scores and thresholds
        """
        # Reset internal state for this new input
        self.candidate_phrases = []
        self.candidate_embeddings = None
        self.recognized_candidates_ngram = []
        self.filtered_by_concept = {}

        # Process the text through the pipeline
        self.prepare_candidate_phrases(long_text)
        if not self.candidate_phrases:
            print("No candidate phrases found in the input text.")
            return []

        self.vectorized_match_candidates()
        if not self.recognized_candidates_ngram:
            print("No technology concepts recognized in the input text.")
            return []

        self.global_filtering()
        return self.get_technologies_with_scores()
