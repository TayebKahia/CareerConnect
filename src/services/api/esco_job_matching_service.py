import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from langdetect import detect
import logging
import os
import time
import pandas as pd
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK resources
nltk.download('punkt', quiet=True)

# Synonyms for skills
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

# Technical skills prioritization
TECHNICAL_SKILLS = {
    "web programming", "JavaScript frameworks", "NoSQL databases", "SQL databases", "API development",
    "containerisation", "cloud computing", "continuous integration", "monitoring systems", "software architecture",
    "database development tools", "use software libraries"
}

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

class ESCOJobMatchingService:
    def __init__(self):
        self.job_titles = []
        self.job_skills_texts = []
        self.job_skills_lists = []
        self.job_skills_importance = []
        self.skill_to_embedding = {}
        self.job_embeddings = None
        self.model = None
        self.all_unique_skills = set()
        self.gcn_encoder = None
        self.X_tfidf = None
        self.A_norm = None
        self.n_user = None
        self.concepts = None
        self.gcn_available = False
        self.gcn_initialized = False
        self.salary_data = None

    def initialize_model(self):
        """Initialize all required models and data"""
        start_time = time.time()
        logger.info("Starting model initialization...")
        
        # Load salary data
        try:
            salary_file = os.path.join("data", "processed", "esco_job_salaries.csv")
            self.salary_data = pd.read_csv(salary_file)
            logger.info(f"Loaded salary data for {len(self.salary_data)} jobs")
        except Exception as e:
            logger.error(f"Failed to load salary data: {str(e)}")
            return False

        # Load ESCO data
        esco_file = os.path.join("data", "processed", "cleaned_IT_data_from_esco.json")
        try:
            with open(esco_file, "r", encoding="utf-8") as f:
                esco_data = json.load(f)
            logger.info(f"ESCO data loaded: {len(esco_data)} entries ({time.time() - start_time:.2f}s)")
        except Exception as e:
            logger.error(f"Failed to load {esco_file}: {str(e)}")
            return False

        # Initialize SentenceTransformer first
        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info(f"SentenceTransformer model loaded ({time.time() - start_time:.2f}s)")
        except Exception as e:
            logger.error(f"Error loading SentenceTransformer: {str(e)}")
            return False

        # Process ESCO data
        importance_weights = {
            "essential_skills": 1.5,
            "essential_knowledge": 1.3,
            "optional_skills": 0.8,
            "optional_knowledge": 0.6
        }

        for entry in esco_data:
            title = entry.get("title", "").strip()
            code = entry.get("isco_code", "").strip()
            description = entry.get("description", "")
            
            # Skip entries without title or code
            if not title or not code:
                continue

            all_skills = []
            skills_with_importance = {}
            skills_section = entry.get("skills", {})

            # Process skills by category
            for category, weight in importance_weights.items():
                for skill in skills_section.get(category, []):
                    if isinstance(skill, dict) and "title" in skill:
                        skill_title = skill["title"]
                        all_skills.append(skill_title)
                        skills_with_importance[skill_title] = weight

            # Add alternative titles as skills with lower weight
            alt_titles = entry.get("alternative_titles", [])
            for alt_title in alt_titles:
                if alt_title:
                    all_skills.append(alt_title)
                    skills_with_importance[alt_title] = 0.5

            if all_skills:
                # Add description to enhance semantic matching
                job_text = f"{title}. {description} Required skills: {', '.join(all_skills)}"
                
                self.job_titles.append(f"{title} ({code})")
                self.job_skills_texts.append(job_text)
                self.job_skills_lists.append(all_skills)
                self.job_skills_importance.append(skills_with_importance)
                self.all_unique_skills.update(all_skills)

        if not self.job_titles:
            logger.error("No jobs were loaded from the data file")
            return False

        logger.info(f"Loaded {len(self.job_titles)} jobs with {len(self.all_unique_skills)} unique skills")

        # Generate or load embeddings
        self._load_or_generate_embeddings()
        
        # Load GCN data
        self._load_gcn_data()
        
        logger.info(f"Initialization completed ({time.time() - start_time:.2f}s)")
        return True

    def _load_or_generate_embeddings(self):
        """Load or generate embeddings for skills and jobs"""
        skill_emb_file = os.path.join("data", "processed", "skill_embeddings.npz")
        job_emb_file = os.path.join("data", "processed", "job_embeddings.npz")
        
        if os.path.exists(skill_emb_file) and os.path.exists(job_emb_file):
            try:
                skill_data = np.load(skill_emb_file, allow_pickle=True)
                self.skill_to_embedding.update({k: v for k, v in zip(skill_data['skills'], skill_data['embeddings'])})
                self.job_embeddings = np.load(job_emb_file)['embeddings']
                logger.info("Embeddings loaded from cache")
            except Exception as e:
                logger.warning(f"Error loading cached embeddings: {str(e)}. Regenerating...")
                self._generate_embeddings(skill_emb_file, job_emb_file)
        else:
            logger.info("Cache files not found, generating embeddings...")
            self._generate_embeddings(skill_emb_file, job_emb_file)

    def _generate_embeddings(self, skill_emb_file, job_emb_file):
        """Generate and save embeddings"""
        start_time = time.time()
        skill_list = list(self.all_unique_skills)
        skill_embeddings = self.model.encode(skill_list, batch_size=128, show_progress_bar=True)
        self.skill_to_embedding.update({skill: emb for skill, emb in zip(skill_list, skill_embeddings)})
        self.job_embeddings = self.model.encode(self.job_skills_texts, batch_size=128, show_progress_bar=True)
        
        try:
            np.savez(skill_emb_file, skills=skill_list, embeddings=skill_embeddings)
            np.savez(job_emb_file, embeddings=self.job_embeddings)
            logger.info(f"Embeddings saved ({time.time() - start_time:.2f}s)")
        except Exception as e:
            logger.warning(f"Error saving embeddings: {str(e)}")

    def _load_gcn_data(self):
        """Load or initialize GCN model and data"""
        try:
            # Load GCN model if available
            gcn_model_path = os.path.join("models", "rao", "gcn_skill_model_2.pt")
            if os.path.exists(gcn_model_path):
                # Initialize GCN with dimensions matching the saved model
                in_feats = 500  # Input feature dimension from saved model
                hidden_feats = 128  # Hidden layer dimension from saved model
                out_feats = 64  # Output dimension from saved model
                
                self.gcn_encoder = GCNEncoder(in_feats, hidden_feats, out_feats)
                self.gcn_encoder.load_state_dict(torch.load(gcn_model_path, map_location=torch.device('cpu')))
                self.gcn_encoder.eval()
                
                # Load graph data
                graph_data_path = os.path.join("models", "rao", "gcn_data.npz")
                if os.path.exists(graph_data_path):
                    data = np.load(graph_data_path, allow_pickle=True)
                    
                    # Convert to torch tensors, handling both sparse and dense matrices
                    X_tfidf = data['X_tfidf']
                    A_norm = data['A_norm']
                    
                    # Convert to dense if sparse
                    if hasattr(X_tfidf, 'toarray'):
                        self.X_tfidf = torch.FloatTensor(X_tfidf.toarray())
                    else:
                        self.X_tfidf = torch.FloatTensor(X_tfidf)
                        
                    if hasattr(A_norm, 'toarray'):
                        self.A_norm = torch.FloatTensor(A_norm.toarray())
                    else:
                        self.A_norm = torch.FloatTensor(A_norm)
                    
                    self.concepts = data['concepts'].tolist()
                    
                    # Verify dimensions
                    n_nodes = len(self.concepts) + len(self.job_titles)
                    if self.X_tfidf.shape[0] != n_nodes or self.A_norm.shape[0] != n_nodes:
                        logger.error(f"Dimension mismatch: X_tfidf shape={self.X_tfidf.shape}, A_norm shape={self.A_norm.shape}, expected {n_nodes} nodes")
                        return False
                    
                    if self.X_tfidf.shape[1] != in_feats:
                        logger.error(f"Feature dimension mismatch: X_tfidf features={self.X_tfidf.shape[1]}, expected {in_feats}")
                        return False
                    
                    self.gcn_available = True
                    logger.info("GCN model loaded successfully")
                    return True
                else:
                    logger.error(f"Graph data file not found: {graph_data_path}")
            else:
                logger.warning(f"GCN model file not found: {gcn_model_path}")
            
            return False
            
        except Exception as e:
            logger.error(f"Error loading GCN data: {str(e)}")
            return False

    def extract_skills_from_text(self, text: str, similarity_threshold: float = 0.5) -> list:
        """Extract skills from text using semantic similarity and keyword matching"""
        # First try direct keyword matching
        matched_keywords = []
        text_lower = text.lower()
        
        # Add common skill variations
        skill_variations = {
            'python': ['python programming', 'python development', 'python developer'],
            'javascript': ['js', 'javascript development', 'javascript programming'],
            'web': ['web development', 'web programming', 'web technologies'],
            'sql': ['mysql', 'postgresql', 'database', 'databases'],
            'api': ['rest api', 'restful api', 'web services'],
            'cloud': ['aws', 'azure', 'gcp', 'cloud computing'],
            'java': ['java programming', 'java development', 'j2ee'],
            'c++': ['cpp', 'c plus plus'],
            'react': ['reactjs', 'react.js'],
            'node': ['nodejs', 'node.js'],
        }

        # Check for skill variations
        for base_skill, variations in skill_variations.items():
            if base_skill in text_lower or any(var in text_lower for var in variations):
                if base_skill in self.all_unique_skills:
                    matched_keywords.append(base_skill)
                for var in variations:
                    if var in self.all_unique_skills and var in text_lower:
                        matched_keywords.append(var)

        # Direct skill matching
        for skill in self.all_unique_skills:
            if skill.lower() in text_lower:
                matched_keywords.append(skill)
        
        logger.info(f"Matched keywords: {matched_keywords}")

        # Then try semantic matching
        text_embedding = self.model.encode([text], show_progress_bar=False)[0]
        extracted_skills = set()

        # Add exact matches first
        extracted_skills.update(matched_keywords)

        # Get embeddings for all skills if not cached
        if not hasattr(self, '_skill_embeddings_cache'):
            self._skill_embeddings_cache = {}
            for skill in self.all_unique_skills:
                if skill not in self._skill_embeddings_cache:
                    self._skill_embeddings_cache[skill] = self.model.encode([skill], show_progress_bar=False)[0]

        # Find semantically similar skills
        for skill in self.all_unique_skills:
            if skill in extracted_skills:
                continue
            
            skill_embedding = self._skill_embeddings_cache[skill]
            similarity = cosine_similarity([text_embedding], [skill_embedding])[0][0]
            
            if similarity >= similarity_threshold:
                extracted_skills.add(skill)

        extracted_skills = list(extracted_skills)
        logger.info(f"Extracted skills: {extracted_skills}")
        logger.info(f"Skill extraction completed ({len(extracted_skills)} skills)")
        
        return extracted_skills

    def _generate_gcn_embeddings(self, skills):
        """Generate GCN embeddings for the given skills"""
        try:
            # Create a user node feature vector with same shape as X_tfidf
            user_features = torch.zeros_like(self.X_tfidf[0]).unsqueeze(0)  # Shape: (1, feature_dim)
            
            # Add features for matching skills
            for skill in skills:
                if skill in self.concepts:
                    idx = self.concepts.index(skill)
                    user_features += self.X_tfidf[idx]
            
            if user_features.sum() == 0:
                logger.warning("No matching skills found in GCN concepts")
                return None

            # Normalize user features
            user_features = F.normalize(user_features, p=2, dim=1)

            # Forward pass through GCN
            with torch.no_grad():
                # Get embeddings for all nodes
                all_node_embeddings = self.gcn_encoder(self.X_tfidf, self.A_norm)
                
                # Get user embedding using the same adjacency matrix
                user_embedding = self.gcn_encoder(user_features, self.A_norm)
                
                # Get job embeddings (last nodes in the graph)
                job_embeddings = all_node_embeddings[-len(self.job_titles):]
                
                # Calculate cosine similarity between user embedding and job embeddings
                similarities = F.cosine_similarity(
                    user_embedding.unsqueeze(1),  # Shape: (1, 1, hidden_dim)
                    job_embeddings.unsqueeze(0),  # Shape: (1, n_jobs, hidden_dim)
                    dim=2
                )
                
                return similarities.squeeze().numpy()

        except Exception as e:
            logger.error(f"Error generating GCN embeddings: {str(e)}")
            return None

    def get_salary_range(self, job_title: str) -> str:
        """Get or generate the salary range for a given job title"""
        try:
            # Remove ISCO code if present
            clean_title = job_title.split(" (")[0] if " (" in job_title else job_title
            
            # Try to find an exact match first
            exact_match = self.salary_data[
                self.salary_data['job_title'].str.lower() == clean_title.lower()
            ]
            
            if not exact_match.empty:
                return exact_match['salary_range'].iloc[0]
            
            # Try partial match
            partial_matches = self.salary_data[
                self.salary_data['job_title'].str.lower().str.contains(clean_title.lower())
            ]
            
            if not partial_matches.empty:
                return partial_matches['salary_range'].iloc[0]
            
            # If no match found, generate a reasonable salary range based on keywords
            base_salary = 75000  # Default base salary
            
            # Adjust base salary based on keywords in the title
            keywords_salary_adjustments = {
                'senior': 30000,
                'lead': 35000,
                'principal': 45000,
                'architect': 40000,
                'manager': 35000,
                'director': 50000,
                'junior': -15000,
                'intern': -35000,
                'trainee': -30000,
                'head': 45000,
                'chief': 60000,
                'specialist': 15000,
                'expert': 25000,
                'analyst': 10000,
                'engineer': 20000,
                'developer': 15000,
                'consultant': 20000
            }
            
            # Technology-specific adjustments
            tech_adjustments = {
                'ai': 25000,
                'machine learning': 25000,
                'data science': 20000,
                'cloud': 20000,
                'security': 20000,
                'blockchain': 25000,
                'devops': 20000,
                'full stack': 15000,
                'frontend': 10000,
                'backend': 12000,
                'mobile': 15000,
                'ios': 15000,
                'android': 15000,
                'web': 10000
            }
            
            title_lower = clean_title.lower()
            
            # Apply keyword adjustments
            for keyword, adjustment in keywords_salary_adjustments.items():
                if keyword in title_lower:
                    base_salary += adjustment
            
            # Apply technology adjustments
            for tech, adjustment in tech_adjustments.items():
                if tech in title_lower:
                    base_salary += adjustment
            
            # Add some randomization (±5%)
            variation = random.uniform(-0.05, 0.05)
            base_salary = int(base_salary * (1 + variation))
            
            # Format the salary
            return f"${base_salary:,}"
            
        except Exception as e:
            logger.warning(f"Error generating salary for job {job_title}: {str(e)}")
            return "$120,000"  # Default fallback salary

    def predict_job(self, text: str, threshold: float = 0.3, similarity_threshold: float = 0.5, gcn_weight: float = 0.3):
        """Predict the most suitable job based on the input text"""
        try:
            # Clean and normalize input text
            text = text.strip()
            if not text:
                return {
                    "error": "Please provide some text describing your skills and experience",
                    "status": "error"
                }, 400

            # Extract skills from text without language detection
            skills = self.extract_skills_from_text(text, similarity_threshold)
            if not skills:
                logger.warning("Only 0 skills extracted. Retrying with lower threshold...")
                skills = self.extract_skills_from_text(text, similarity_threshold * 0.7)

            if not skills:
                return {
                    "error": "No relevant skills found in the text",
                    "status": "error"
                }, 400

            # Get text embedding
            text_embedding = self.model.encode([text], show_progress_bar=False)[0]

            # Get GCN predictions if available
            gcn_scores = None
            if self.gcn_available:
                try:
                    gcn_scores = self._generate_gcn_embeddings(skills)
                except Exception as e:
                    logger.error(f"Error getting GCN predictions: {str(e)}")

            # Calculate similarity scores
            job_scores = []
            all_semantic_scores = []
            all_skill_scores = []
            
            # First pass to collect all scores for normalization
            for i, job_text in enumerate(self.job_skills_texts):
                # Get semantic similarity
                job_embedding = self.job_embeddings[i]
                semantic_score = cosine_similarity([text_embedding], [job_embedding])[0][0]
                all_semantic_scores.append(semantic_score)

                # Calculate skill overlap score
                job_skills = set(self.job_skills_lists[i])
                user_skills = set(skills)
                matched_skills = job_skills.intersection(user_skills)
                
                # Calculate weighted skill score based on importance
                weighted_score = 0
                total_importance = sum(self.job_skills_importance[i].values())
                
                for skill in matched_skills:
                    importance = self.job_skills_importance[i].get(skill, 1.0)
                    weighted_score += importance
                
                skill_score = weighted_score / total_importance if total_importance > 0 else 0
                all_skill_scores.append(skill_score)

            # Normalize scores to 0-100 range
            if all_semantic_scores:
                min_semantic = min(all_semantic_scores)
                max_semantic = max(all_semantic_scores)
                semantic_range = max_semantic - min_semantic
            
            if all_skill_scores:
                min_skill = min(all_skill_scores)
                max_skill = max(all_skill_scores)
                skill_range = max_skill - min_skill

            # Second pass to calculate final scores with normalization
            for i, job_text in enumerate(self.job_skills_texts):
                semantic_score = all_semantic_scores[i]
                skill_score = all_skill_scores[i]

                # Normalize scores to 0-100 range
                if semantic_range > 0:
                    normalized_semantic = ((semantic_score - min_semantic) / semantic_range) * 85  # Cap at 85
                else:
                    normalized_semantic = semantic_score * 85

                if skill_range > 0:
                    normalized_skill = ((skill_score - min_skill) / skill_range) * 90  # Cap at 90
                else:
                    normalized_skill = skill_score * 90

                # Calculate skill overlap for additional boost
                job_skills = set(self.job_skills_lists[i])
                user_skills = set(skills)
                matched_skills = job_skills.intersection(user_skills)
                overlap_ratio = len(matched_skills) / len(job_skills) if job_skills else 0

                # Apply skill match bonus (up to 8% boost for perfect skill match)
                skill_match_bonus = overlap_ratio * 8

                # Combine scores with weights
                skill_weight = 0.65  # 65% weight to skill score
                semantic_weight = 0.35  # 35% weight to semantic score
                
                # Calculate base score with diminishing returns
                base_score = (skill_weight * normalized_skill) + (semantic_weight * normalized_semantic)
                base_score = base_score * (0.95 - (base_score * 0.0005))  # Apply slight diminishing returns
                
                # Add skill match bonus
                final_score = min(93, base_score + skill_match_bonus)  # Hard cap at 93

                # Add GCN score if available (with reduced impact)
                if gcn_scores is not None:
                    gcn_score = gcn_scores[i] * 85  # Normalize GCN score to 0-85 scale
                    final_score = min(93, (1 - gcn_weight) * final_score + gcn_weight * gcn_score)  # Maintain 93 cap

                # Lower the threshold for jobs with matching skills
                effective_threshold = threshold * 100  # Convert threshold to 0-100 scale
                if len(matched_skills) > 0:
                    effective_threshold = effective_threshold * 0.8  # 20% lower threshold if any skills match

                if final_score >= effective_threshold:
                    job_title = self.job_titles[i]
                    title_part = job_title.split(" (")[0]
                    code_part = job_title.split(" (")[1].rstrip(")")
                    
                    # Get salary range
                    salary_range = self.get_salary_range(title_part)
                    
                    # Get missing skills
                    missing_skills = list(job_skills - user_skills)
                    missing_skills.sort(
                        key=lambda x: self.job_skills_importance[i].get(x, 0),
                        reverse=True
                    )
                    
                    # Calculate confidence level based on score ranges
                    confidence_level = "Very High" if final_score >= 85 else \
                                     "High" if final_score >= 75 else \
                                     "Moderate" if final_score >= 65 else \
                                     "Fair" if final_score >= 55 else "Low"
                    
                    job_scores.append({
                        "title": title_part,
                        "code": code_part,
                        "score": round(final_score, 1),
                        "confidence": confidence_level,
                        "matching_skills": list(matched_skills),
                        "missing_skills": missing_skills[:5],
                        "total_skills_required": len(job_skills),
                        "skills_matched": len(matched_skills),
                        "match_percentage": round((len(matched_skills) / len(job_skills)) * 100, 1),
                        "semantic_score": round(normalized_semantic, 1),
                        "skill_score": round(normalized_skill, 1),
                        "skill_match_bonus": round(skill_match_bonus, 1),
                        "salary_range": salary_range
                    })

            # Sort by score
            job_scores.sort(key=lambda x: x["score"], reverse=True)

            if not job_scores:
                # If no jobs match, return the closest ones below threshold
                all_scores = []
                for i, job_text in enumerate(self.job_skills_texts):
                    semantic_score = all_semantic_scores[i]
                    normalized_score = ((semantic_score - min_semantic) / semantic_range * 100) if semantic_range > 0 else semantic_score * 100
                    
                    job_title = self.job_titles[i]
                    title_part = job_title.split(" (")[0]
                    code_part = job_title.split(" (")[1].rstrip(")")
                    
                    job_skills = set(self.job_skills_lists[i])
                    user_skills = set(skills)
                    matched_skills = job_skills.intersection(user_skills)
                    
                    all_scores.append({
                        "title": title_part,
                        "code": code_part,
                        "score": round(normalized_score, 2),
                        "skills_required": list(job_skills),
                        "matching_skills": list(matched_skills),
                        "skills_matched": len(matched_skills),
                        "match_percentage": round((len(matched_skills) / len(job_skills)) * 100, 1) if job_skills else 0
                    })
                all_scores.sort(key=lambda x: x["score"], reverse=True)
                
                return {
                    "message": "No exact matches found, showing closest matches",
                    "jobs": all_scores[:5],
                    "extracted_skills": skills,
                    "threshold": threshold * 100,  # Convert to percentage
                    "note": "Consider lowering the threshold for more matches",
                    "status": "success"
                }, 200

            return {
                "jobs": job_scores,
                "extracted_skills": skills,
                "message": "Jobs found successfully",
                "threshold_used": threshold * 100,  # Convert to percentage
                "status": "success"
            }, 200

        except Exception as e:
            logger.error(f"Error predicting job: {str(e)}")
            return {
                "error": str(e),
                "status": "error"
            }, 500 