from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import uvicorn
from scipy.sparse import load_npz
import logging
import os
from langdetect import detect
import time

# Configurer le logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Télécharger les ressources NLTK nécessaires
nltk.download('punkt', quiet=True)

# Initialisation de l'application FastAPI
app = FastAPI(title="ESCO Job Matching API with GCN", description="API pour prédire un métier à partir d'un texte en langage naturel en utilisant GCN")

# Modèle Pydantic pour valider les données d'entrée
class TextInput(BaseModel):
    text: str
    threshold: float = 0.5
    similarity_threshold: float = 0.5  # Réduit pour capturer plus de compétences
    gcn_weight: float = 0.3

# Définition du GCNEncoder
class GCNEncoder(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super().__init__()
        self.fc1 = nn.Linear(in_feats, hidden_feats)
        self.fc2 = nn.Linear(hidden_feats, out_feats)

    def forward(self, X, A):
        H = A @ self.fc1(X)
        H = F.relu(H)
        H = A @ self.fc2(H)
        return H

# Synonymes pour les compétences
SYNONYMS = {
    "kubernetes": "containerisation",
    "terraform": "infrastructure as code",
    "gitlab ci/cd": "continuous integration",
    "prometheus": "monitoring systems",
    "grafana": "monitoring systems",
    "cloud architecture": "cloud computing",
    "docker": "containerisation",
    "aws": "cloud computing",
    "azure": "cloud computing",
    "gcp": "cloud computing",
    "javascript": "web programming",
    "react": "JavaScript frameworks",
    "node.js": "JavaScript frameworks",
    "mongodb": "NoSQL databases",
    "rest api": "API development",
    "rest apis": "API development",
    "python": "web programming",
    "django": "web programming",
    "postgresql": "SQL databases",
    "jenkins": "continuous integration",
    "microservices": "software architecture",
    "microservice": "software architecture"
}

# Compétences techniques prioritaires
TECHNICAL_SKILLS = {
    "web programming", "JavaScript frameworks", "NoSQL databases", "SQL databases", "API development",
    "containerisation", "cloud computing", "continuous integration", "monitoring systems", "software architecture",
    "database development tools", "use software libraries"
}

# Variables globales
job_titles = []
job_skills_texts = []
job_skills_lists = []
job_skills_importance = []
skill_to_embedding = {}
job_embeddings = None
model = None
all_unique_skills = set()
gcn_encoder = None
X_tfidf = None
A_norm = None
n_user = None
concepts = None
gcn_available = False
gcn_initialized = False

# Charger les données et initialiser les modèles au démarrage
@app.on_event("startup")
def load_data_and_model():
    global job_titles, job_skills_texts, job_skills_lists, job_skills_importance, skill_to_embedding, job_embeddings, model, all_unique_skills
    
    start_time = time.time()
    logger.info("Démarrage du chargement des données et des modèles...")
    
    # Timeout pour détecter les démarrages lents
    if time.time() - start_time > 30:
        logger.error("Démarrage trop lent, possible problème de cache ou de réseau")
        raise HTTPException(status_code=500, detail="Démarrage trop lent")

    # Vérifier et charger les données ESCO
    esco_file = "cleaned_IT_data_from_esco.json"
    if not os.path.exists(esco_file):
        logger.error(f"Fichier {esco_file} introuvable")
        raise HTTPException(status_code=500, detail=f"Fichier {esco_file} introuvable")

    try:
        with open(esco_file, "r", encoding="utf-8") as f:
            esco_data = json.load(f)
        logger.info(f"Données ESCO chargées : {len(esco_data)} entrées ({time.time() - start_time:.2f}s)")
    except Exception as e:
        logger.error(f"Erreur lors du chargement de {esco_file}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du chargement de {esco_file}: {str(e)}")

    # Pré-traitement des données ESCO
    importance_weights = {
        "essential_skills": 1.5,
        "essential_knowledge": 1.3,
        "optional_skills": 0.8,
        "optional_knowledge": 0.6
    }

    critical_skills = TECHNICAL_SKILLS
    for entry in esco_data:
        title = entry.get("title", "")
        skills_section = entry.get("skills", {})
        all_skills = []
        skills_with_importance = {}

        for category in importance_weights:
            weight = importance_weights[category]
            for skill in skills_section.get(category, []):
                if "title" in skill:
                    skill_title = skill["title"]
                    all_skills.append(skill_title)
                    skills_with_importance[skill_title] = weight

        if all_skills:
            job_titles.append(title)
            enhanced_text = []
            for skill, importance in skills_with_importance.items():
                repetitions = int(importance * 2)
                enhanced_text.extend([skill] * repetitions)
            job_skills_texts.append(", ".join(enhanced_text))
            job_skills_lists.append(all_skills)
            job_skills_importance.append(skills_with_importance)
            all_unique_skills.update(all_skills)

    logger.info(f"{len(job_titles)} métiers chargés depuis ESCO ({time.time() - start_time:.2f}s)")
    logger.info(f"Exemple de compétences ESCO : {list(all_unique_skills)[:10]}")
    missing_critical = critical_skills - all_unique_skills
    if missing_critical:
        logger.warning(f"Compétences critiques manquantes dans ESCO : {missing_critical}")

    # Initialiser le modèle SentenceTransformer
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info(f"Modèle SentenceTransformer chargé ({time.time() - start_time:.2f}s)")
    except Exception as e:
        logger.error(f"Erreur lors du chargement de SentenceTransformer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du chargement de SentenceTransformer: {str(e)}")

    # Charger ou générer les embeddings
    skill_emb_file = "skill_embeddings.npz"
    job_emb_file = "job_embeddings.npz"
    
    if os.path.exists(skill_emb_file) and os.path.exists(job_emb_file):
        try:
            skill_data = np.load(skill_emb_file, allow_pickle=True)
            skill_to_embedding.update({k: v for k, v in zip(skill_data['skills'], skill_data['embeddings'])})
            job_embeddings = np.load(job_emb_file)['embeddings']
            logger.info(f"Embeddings chargés depuis le cache ({time.time() - start_time:.2f}s)")
        except Exception as e:
            logger.warning(f"Erreur lors du chargement des embeddings en cache: {str(e)}. Régénération...")
            generate_embeddings()
    else:
        logger.info("Fichiers de cache absents, génération des embeddings...")
        generate_embeddings()

    logger.info(f"Startup terminé ({time.time() - start_time:.2f}s)")

def generate_embeddings():
    global skill_to_embedding, job_embeddings
    start_time = time.time()
    skill_list = list(all_unique_skills)
    skill_embeddings = model.encode(skill_list, batch_size=128, show_progress_bar=True)
    skill_to_embedding.update({skill: emb for skill, emb in zip(skill_list, skill_embeddings)})
    job_embeddings = model.encode(job_skills_texts, batch_size=128, show_progress_bar=True)
    
    try:
        np.savez(skill_emb_file, skills=skill_list, embeddings=skill_embeddings)
        np.savez(job_emb_file, embeddings=job_embeddings)
        logger.info(f"Embeddings sauvegardés ({time.time() - start_time:.2f}s)")
    except Exception as e:
        logger.warning(f"Erreur lors de la sauvegarde des embeddings: {str(e)}")

# Charger les données GCN (lazy loading)
def load_gcn_data():
    global gcn_encoder, X_tfidf, A_norm, n_user, concepts, gcn_available, gcn_initialized
    if gcn_initialized:
        return
    
    start_time = time.time()
    gcn_data_file = "gcn_data.npz"
    if not os.path.exists(gcn_data_file):
        logger.warning(f"Fichier {gcn_data_file} introuvable. Le GCN ne sera pas utilisé.")
        gcn_available = False
    else:
        try:
            data = np.load(gcn_data_file)
            X_tfidf = torch.tensor(data['X_tfidf'], dtype=torch.float32)
            A_norm = torch.tensor(data['A_norm'], dtype=torch.float32)
            n_user = int(data['n_user'])
            concepts = data['concepts'].tolist()
            logger.info(f"Données GCN chargées - X_tfidf shape: {X_tfidf.shape}, A_norm shape: {A_norm.shape} ({time.time() - start_time:.2f}s)")
            logger.info(f"Exemple de concepts GCN : {concepts[:10]}")
            missing_tech = TECHNICAL_SKILLS - set(concepts)
            if missing_tech:
                logger.warning(f"Compétences techniques manquantes dans GCN concepts : {missing_tech}")
            gcn_available = True
        except Exception as e:
            logger.error(f"Erreur lors du chargement de {gcn_data_file}: {str(e)}")
            gcn_available = False
            return

        gcn_model_file = "gcn_skill_model_2.pt"
        if not os.path.exists(gcn_model_file):
            logger.warning(f"Fichier {gcn_model_file} introuvable. Le GCN ne sera pas utilisé.")
            gcn_available = False
        else:
            try:
                gcn_encoder = GCNEncoder(in_feats=X_tfidf.shape[1], hidden_feats=128, out_feats=64)
                gcn_encoder.load_state_dict(torch.load(gcn_model_file, map_location=torch.device('cpu'), weights_only=True))
                gcn_encoder.eval()
                logger.info(f"Modèle GCN chargé ({time.time() - start_time:.2f}s)")
                with torch.no_grad():
                    test_emb = gcn_encoder(X_tfidf[:10], A_norm[:10, :])
                    logger.info(f"Test GCN embeddings shape: {test_emb.shape}")
            except Exception as e:
                logger.error(f"Erreur lors du chargement de {gcn_model_file}: {str(e)}")
                gcn_available = False
    
    gcn_initialized = True

# Fonction pour extraire les compétences du texte
def extract_skills_from_text(text: str, similarity_threshold: float = 0.5):
    try:
        start_time = time.time()
        text_lang = detect(text)
        logger.info(f"Langue détectée du texte : {text_lang}")
        esco_lang = detect(next(iter(all_unique_skills), "unknown"))
        if text_lang != esco_lang:
            logger.warning(f"Langue du texte ({text_lang}) différente de la langue ESCO ({esco_lang})")

        sentences = sent_tokenize(text)
        extracted_skills = []
        skill_scores = []
        matched_keywords = []
        unmatched_keywords = []

        sentence_embeddings = model.encode(sentences, batch_size=32, show_progress_bar=False)
        skill_embeddings = np.array(list(skill_to_embedding.values()))
        skill_names = list(skill_to_embedding.keys())

        for sent_emb in sentence_embeddings:
            similarities = cosine_similarity([sent_emb], skill_embeddings)[0]
            for idx, sim in enumerate(similarities):
                if sim > similarity_threshold:
                    extracted_skills.append(skill_names[idx])
                    skill_scores.append((skill_names[idx], float(sim)))

        # Extraction basée sur les mots-clés et synonymes
        keywords = set(word_tokenize(text.lower()))
        keyword_skills = []
        for keyword in keywords:
            if keyword in SYNONYMS:
                mapped_skill = SYNONYMS[keyword]
                if mapped_skill in all_unique_skills:
                    keyword_skills.append(mapped_skill)
                    skill_scores.append((mapped_skill, 1.0))
                    matched_keywords.append(keyword)
                else:
                    unmatched_keywords.append((keyword, mapped_skill))
            elif keyword in all_unique_skills:
                keyword_skills.append(keyword)
                skill_scores.append((keyword, 1.0))
                matched_keywords.append(keyword)
            else:
                unmatched_keywords.append((keyword, None))

        # Prioriser les compétences techniques
        prioritized_skills = [s for s in keyword_skills if s in TECHNICAL_SKILLS]
        other_technical = [s for s in extracted_skills if s in TECHNICAL_SKILLS and s not in prioritized_skills]
        non_technical = [s for s in extracted_skills if s not in TECHNICAL_SKILLS and s in all_unique_skills]
        extracted_skills = prioritized_skills + other_technical + non_technical[:2]  # Réduit à 2 non-techniques

        seen = set()
        unique_skills = [skill for skill in extracted_skills if not (skill in seen or seen.add(skill))]
        logger.info(f"Mots-clés correspondants : {matched_keywords}")
        logger.info(f"Mots-clés non correspondants : {unmatched_keywords[:10]}")
        logger.info(f"Compétences extraites : {unique_skills}")
        logger.info(f"Scores des compétences : {skill_scores} ({time.time() - start_time:.2f}s)")
        
        # Fallback si moins de 4 compétences
        if len(unique_skills) < 4 and similarity_threshold > 0.4:
            logger.warning(f"Seulement {len(unique_skills)} compétences extraites avec threshold {similarity_threshold}. Réessayer avec 0.4...")
            return extract_skills_from_text(text, similarity_threshold=0.4)
        
        return unique_skills
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction des compétences : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'extraction des compétences : {str(e)}")

# Fonction pour générer des embeddings GCN
def generate_gcn_embeddings(skills):
    load_gcn_data()
    if not gcn_available:
        logger.warning("GCN non disponible, retour à None")
        return None

    try:
        start_time = time.time()
        skill_indices = [concepts.index(skill) for skill in skills if skill in concepts]
        logger.info(f"Compétences GCN correspondantes : {[skills[i] for i in skill_indices]}")
        if not skill_indices:
            logger.warning("Aucune compétence correspondante dans les concepts GCN")
            return None

        user_X_tfidf = X_tfidf[skill_indices]
        with torch.no_grad():
            embeddings = gcn_encoder(user_X_tfidf, A_norm[skill_indices, :])
        logger.info(f"Embeddings GCN générés : shape {embeddings.shape} ({time.time() - start_time:.2f}s)")
        return embeddings
    except Exception as e:
        logger.error(f"Erreur lors de la génération des embeddings GCN : {str(e)}")
        return None

# Fonction de prédiction
def predire_metier_depuis_skills(user_skills, threshold=0.5, gcn_weight=0.3):
    start_time = time.time()
    if not user_skills:
        logger.warning("Aucune compétence extraite, retour à une réponse vide")
        raise ValueError("Aucune compétence extraite du texte")

    try:
        user_text = ", ".join(user_skills)
        user_embedding = model.encode([user_text], batch_size=1)
        similarities_text = cosine_similarity(user_embedding, job_embeddings)[0]

        individual_skill_scores = []
        for i, job_skills in enumerate(job_skills_lists):
            job_importance = job_skills_importance[i]
            matching_score = 0
            total_weight = 0
            user_skill_embeddings = [model.encode(skill, batch_size=1) for skill in user_skills]

            for job_skill in job_skills:
                weight = job_importance.get(job_skill, 1.0)
                job_skill_embedding = skill_to_embedding[job_skill]
                best_similarity = 0
                for user_skill_emb in user_skill_embeddings:
                    sim = cosine_similarity([user_skill_emb], [job_skill_embedding])[0][0]
                    best_similarity = max(best_similarity, sim)
                if best_similarity > threshold:
                    matching_score += best_similarity * weight
                total_weight += weight

            normalized_score = matching_score / total_weight if total_weight > 0 else 0
            individual_skill_scores.append(normalized_score)

        combined_scores_st = 0.5 * np.array(similarities_text) + 0.5 * np.array(individual_skill_scores)
        logger.info(f"Scores SentenceTransformer : min={combined_scores_st.min():.4f}, max={combined_scores_st.max():.4f}")
    except Exception as e:
        logger.error(f"Erreur lors du calcul des scores SentenceTransformer : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du calcul des scores SentenceTransformer : {str(e)}")

    if gcn_available:
        try:
            gcn_user_emb = generate_gcn_embeddings(user_skills)
            if gcn_user_emb is not None:
                with torch.no_grad():
                    Z = gcn_encoder(X_tfidf, A_norm)
                    concept_emb = Z[n_user:]
                    sim_matrix = cosine_similarity(gcn_user_emb.numpy(), concept_emb.numpy())
                    gcn_similarities = np.max(sim_matrix, axis=0)

                gcn_scores = []
                for job_skills in job_skills_lists:
                    job_skill_indices = [concepts.index(skill) for skill in job_skills if skill in concepts]
                    if job_skill_indices:
                        job_sim = np.mean([gcn_similarities[i - n_user] for i in job_skill_indices if i >= n_user])
                        gcn_scores.append(job_sim)
                    else:
                        gcn_scores.append(0.0)
                gcn_scores = np.array(gcn_scores)
                logger.info(f"Scores GCN : min={gcn_scores.min():.4f}, max={gcn_scores.max():.4f}")
            else:
                gcn_scores = np.zeros(len(job_titles))
                logger.warning("Aucun embedding GCN généré, utilisation des scores SentenceTransformer uniquement")
        except Exception as e:
            logger.error(f"Erreur lors du calcul des scores GCN : {str(e)}")
            gcn_scores = np.zeros(len(job_titles))
    else:
        gcn_scores = np.zeros(len(job_titles))
        logger.warning("GCN non disponible, utilisation des scores SentenceTransformer uniquement")

    combined_scores = gcn_weight * gcn_scores + (1 - gcn_weight) * combined_scores_st
    common_skills_count = []
    for job_skills in job_skills_lists:
        job_skills_set = set(job_skills)
        user_skills_set = set(user_skills)
        common = len(job_skills_set.intersection(user_skills_set))
        total = len(job_skills_set)
        ratio = common / total if total > 0 else 0
        common_skills_count.append(ratio * 0.5)  # Augmenté pour pénaliser faible overlap

    final_scores = combined_scores + np.array(common_skills_count)
    min_score = np.min(final_scores)
    max_score = np.max(final_scores)
    normalized_final_scores = (final_scores - min_score) / (max_score - min_score) if max_score > min_score else final_scores
    percentage_scores = np.clip(normalized_final_scores * 100, 0, 80)  # Limité à 80%

    best_idx = np.argmax(percentage_scores)
    common_skills = len(set(job_skills_lists[best_idx]).intersection(set(user_skills)))
    if common_skills < 4:
        logger.warning(f"Correspondance faible : seulement {common_skills} compétences communes pour {job_titles[best_idx]}")
        percentage_scores[best_idx] *= 0.6  # Pénaliser fortement
        best_idx = np.argmax(percentage_scores)

    missing_skills = []
    if percentage_scores[best_idx] < 80:
        best_job_skills = set(job_skills_lists[best_idx])
        user_skills_set = set(user_skills)
        essential_missing = []
        optional_missing = []
        for skill in best_job_skills - user_skills_set:
            importance = job_skills_importance[best_idx].get(skill, 0)
            if importance >= 1.0:
                essential_missing.append(skill)
            else:
                optional_missing.append(skill)
        missing_skills = {
            "essential": essential_missing[:5],
            "optional": optional_missing[:3]
        }

    result = {
        "predicted_job": job_titles[best_idx],
        "similarity_score": float(percentage_scores[best_idx]),
        "top_5": [(job_titles[i], float(percentage_scores[i])) for i in np.argsort(percentage_scores)[::-1][:5]],
        "job_skills_matched": ", ".join(job_skills_lists[best_idx][:10]),
        "missing_skills": missing_skills if missing_skills else None,
        "extracted_skills": user_skills
    }
    logger.info(f"Prédiction terminée ({time.time() - start_time:.2f}s)")
    return result

# Endpoint pour prédire le métier
@app.post("/predict_job_from_text")
async def predict_job_from_text(input: TextInput):
    start_time = time.time()
    logger.info(f"Requête reçue : {input.text}")
    try:
        extracted_skills = extract_skills_from_text(input.text, input.similarity_threshold)
        if not extracted_skills:
            logger.warning("Aucune compétence reconnue dans le texte")
            raise HTTPException(status_code=400, detail="Aucune compétence reconnue dans le texte fourni. Essayez de réduire similarity_threshold.")
        
        result = predire_metier_depuis_skills(extracted_skills, input.threshold, input.gcn_weight)
        logger.info(f"Résultat : {result['predicted_job']} (score: {result['similarity_score']:.2f}%) ({time.time() - start_time:.2f}s)")
        return result
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la requête : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement de la requête : {str(e)}")

# Endpoint de débogage pour extraire les compétences
@app.post("/debug_skills")
async def debug_skills(input: TextInput):
    start_time = time.time()
    try:
        skills = extract_skills_from_text(input.text, input.similarity_threshold)
        skill_scores = []
        unmatched_keywords = []
        sentence_embeddings = model.encode(sent_tokenize(input.text), batch_size=32, show_progress_bar=False)
        skill_embeddings = np.array(list(skill_to_embedding.values()))
        skill_names = list(skill_to_embedding.keys())
        for sent_emb in sentence_embeddings:
            similarities = cosine_similarity([sent_emb], skill_embeddings)[0]
            for idx, sim in enumerate(similarities):
                if sim > input.similarity_threshold and skill_names[idx] in skills:
                    skill_scores.append((skill_names[idx], float(sim)))
        keywords = set(word_tokenize(input.text.lower()))
        matched_keywords = [k for k in keywords if k in SYNONYMS and SYNONYMS[k] in skills]
        unmatched_keywords = [(k, SYNONYMS.get(k)) for k in keywords if k not in matched_keywords and k not in all_unique_skills]
        logger.info(f"Débogage compétences terminé ({time.time() - start_time:.2f}s)")
        return {
            "extracted_skills": skills,
            "skill_scores": skill_scores,
            "matched_keywords": matched_keywords,
            "unmatched_keywords": unmatched_keywords[:10]
        }
    except Exception as e:
        logger.error(f"Erreur lors du débogage des compétences : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du débogage des compétences : {str(e)}")

# Endpoint de santé pour vérifier le démarrage
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "esco_loaded": len(job_titles) > 0,
        "embeddings_cached": os.path.exists("skill_embeddings.npz") and os.path.exists("job_embeddings.npz"),
        "gcn_available": gcn_available
    }

# Lancer l'API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)