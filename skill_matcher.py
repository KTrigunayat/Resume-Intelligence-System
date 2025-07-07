#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Skill-to-JD Semantic Matcher for Resume Intelligence System

This module measures how well a candidate's listed skills align with a target job description,
yielding an overall "alignment %" and pinpointing missing critical competencies.
"""

import json
from pathlib import Path
import re
import os
import openai
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv
import csv

import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkillMatcher:
    """Matcher for comparing resume skills with job description requirements."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the skill matcher.
        
        Args:
            model_name (str): Name of the sentence transformer model to use.
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            self.logger.warning("OPENAI_API_KEY not found. Using fallback methods.")
            self.use_llm = False
        else:
            openai.api_key = self.openai_api_key
            self.use_llm = True
        
        # Initialize sentence transformer model
        try:
            self.model = SentenceTransformer(model_name)
            self.use_embeddings = True
        except Exception as e:
            self.logger.warning(f"Could not load model {model_name}. Using TF-IDF instead. Error: {e}")
            self.model = None
            self.use_embeddings = False
        
        # Initialize TF-IDF vectorizer as fallback
        self.tfidf = TfidfVectorizer(stop_words='english')
        
        # Load skill synonyms and variations
        self.skill_synonyms = self._load_skill_synonyms()
        
        # Load valid skills set from skill2vec.csv
        self.valid_skills = self._load_valid_skills()
        # Common English stopwords and section headers to filter out
        self.stopwords_and_headers = set([
            'minimum', 'team', 'summary', 'education', 'skills', 'projects', 'work', 'experience',
            'publications', 'certifications', 'languages', 'preferred', 'benefits', 'duration',
            'about', 'as', 'at', 'this', 'you', 'your', 'varies', 'multiple', 'location', 'locations',
            'present', 'currently', 'excellent', 'strong', 'passion', 'familiarity', 'access', 'perks',
            'ux', 'look', 'looked', 'looker', 'india', 'cloud', 'internship', 'intern', 'phd', 'bachelor',
            'master', 'gpa', 'native', 'expected', 'may', 'january', 'doe', 'inc', 'conference', 'symposium',
            'student', 'basic', 'relevant', 'coursework', 'achieved', 'created', 'developed', 'built', 'analyzed',
            'conducted', 'presented', 'assist', 'assistant', 'solutions', 'applications', 'platforms', 'tools',
            'technology', 'technologies', 'media', 'notebooks', 'practitioner', 'clean', 'processed', 'mining',
            'statistical', 'visualization', 'analytics', 'analysis', 'science', 'learning', 'machine', 'big',
            'aws', 'google', 'tableau', 'spark', 'sql', 'python', 'java', 'r', 'matplotlib', 'seaborn', 'numpy',
            'pandas', 'scikit', 'scikit-learn', 'docker', 'git', 'tensorflow', 'pytorch', 'hadoop', 'bigquery',
            'azure', 'gcp', 'ci/cd', 'power', 'power bi', 'cloud', 'languages', 'english', 'spanish', 'mandarin',
            'certificate', 'certified', 'publications', 'presentations', 'projects', 'skills', 'education', 'summary',
            'work experience', 'certifications', 'languages', 'skills', 'projects', 'experience', 'publications',
            'certifications', 'languages', 'skills', 'projects', 'experience', 'publications', 'certifications',
            'languages', 'skills', 'projects', 'experience', 'publications', 'certifications', 'languages',
            # Add more as needed
        ])
    
    def _load_skill_synonyms(self) -> Dict[str, List[str]]:
        """Load skill synonyms and variations from a JSON file."""
        try:
            synonyms_path = Path(__file__).parent / "data" / "skill_synonyms.json"
            if synonyms_path.exists():
                with open(synonyms_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load skill synonyms: {e}")
        return {}
    
    def _load_valid_skills(self):
        """Efficiently load valid skills from skill2vec.csv (second column and beyond, ignoring empty values)."""
        skills = set()
        try:
            skills_path = Path(__file__).parent.parent / "skill2vec.csv"
            if skills_path.exists():
                with open(skills_path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        for skill in row[1:]:
                            skill = skill.strip().lower()
                            if skill:
                                skills.add(skill)
        except Exception as e:
            self.logger.warning(f"Could not load valid skills from skill2vec.csv: {e}")
        return skills
    
    def _filter_skills(self, skills: list) -> list:
        """Filter out stopwords/headers and keep only valid skills."""
        filtered = []
        for skill in skills:
            skill_l = skill.strip().lower()
            if skill_l and skill_l not in self.stopwords_and_headers and (
                skill_l in self.valid_skills or len(skill_l) > 2 and not skill_l.isnumeric()):
                filtered.append(skill)
        return filtered
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from text using LLM if available, otherwise use regex.
        
        Args:
            text (str): Text containing skills.
            
        Returns:
            list: List of extracted skills.
        """
        if not text:
            return []
        
        # Try LLM-based extraction first
        if self.use_llm:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a skill extraction expert. Extract technical skills, programming languages, tools, frameworks, and technologies from the given text. Return only the skills as a comma-separated list."},
                        {"role": "user", "content": f"Extract skills from this text: {text}"}
                    ],
                    temperature=0.3,
                    max_tokens=150
                )
                
                skills_text = response.choices[0].message.content.strip()
                skills = [skill.strip() for skill in skills_text.split(',') if skill.strip()]
                
                # Expand skills using synonyms
                expanded_skills = self._expand_skills_with_synonyms(skills)
                # Filter out generic/common words and section headers
                filtered_skills = self._filter_skills(expanded_skills)
                return filtered_skills
                
            except Exception as e:
                self.logger.warning(f"Error using LLM for skill extraction: {e}")
        
        # Fallback to regex-based extraction
        skills = self._extract_skills_regex(text)
        
        # Expand skills using synonyms
        expanded_skills = self._expand_skills_with_synonyms(skills)
        # Filter out generic/common words and section headers
        filtered_skills = self._filter_skills(expanded_skills)
        return filtered_skills
    
    def _extract_skills_regex(self, text: str) -> List[str]:
        """Extract skills using regex patterns."""
        # Common skill patterns
        patterns = [
            r'\b[A-Z][A-Za-z+#]+\b',  # Capitalized words (e.g., Python, Java)
            r'\b[A-Za-z]+\.js\b',     # JavaScript frameworks
            r'\b[A-Za-z]+\.NET\b',    # .NET technologies
            r'\b[A-Za-z]+\.py\b',     # Python packages
            r'\b[A-Za-z]+\.io\b',     # Web technologies
            r'\b[A-Za-z]+\.sh\b',     # Shell scripts
            r'\b[A-Za-z]+\.sql\b',    # SQL variants
        ]
        
        skills = set()
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                skill = match.group(0).strip()
                if len(skill) > 2:  # Filter out very short matches
                    skills.add(skill)
        
        # Also split by common delimiters
        delimiters = r'[,;•\n]'
        parts = re.split(delimiters, text)
        for part in parts:
            skill = part.strip()
            if len(skill) > 2 and not re.search(r'\s', skill):  # Single words only
                skills.add(skill)
        
        return list(skills)
    
    def _expand_skills_with_synonyms(self, skills: List[str]) -> List[str]:
        """Expand skills using known synonyms and variations."""
        expanded_skills = set(skills)
        
        for skill in skills:
            # Check for exact matches in synonyms
            if skill in self.skill_synonyms:
                expanded_skills.update(self.skill_synonyms[skill])
            
            # Check for case-insensitive matches
            skill_lower = skill.lower()
            for key, synonyms in self.skill_synonyms.items():
                if key.lower() == skill_lower:
                    expanded_skills.update(synonyms)
        
        return list(expanded_skills)
    
    def extract_jd_requirements(self, jd_text: str) -> List[str]:
        """Extract requirements from job description using LLM if available.
        
        Args:
            jd_text (str): Job description text.
            
        Returns:
            list: List of extracted requirements.
        """
        if not jd_text:
            return []
        
        # Try LLM-based extraction first
        if self.use_llm:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": """You are a job requirement extraction expert. Extract technical requirements, skills, and qualifications from the given job description. 
                        Focus on:
                        1. Technical skills (e.g., Python, SQL, Machine Learning)
                        2. Programming languages
                        3. Tools and frameworks
                        4. Technologies and platforms
                        5. Required experience levels
                        Return only the requirements as a comma-separated list. Each requirement should be a single skill or technology, not a full sentence."""},
                        {"role": "user", "content": f"Extract technical requirements from this job description: {jd_text}"}
                    ],
                    temperature=0.3,
                    max_tokens=200
                )
                
                requirements_text = response.choices[0].message.content.strip()
                requirements = [req.strip() for req in requirements_text.split(',') if req.strip()]
                
                # Expand requirements using synonyms
                expanded_requirements = self._expand_skills_with_synonyms(requirements)
                # Filter out generic/common words and section headers
                filtered_requirements = self._filter_skills(expanded_requirements)
                return filtered_requirements
                
            except Exception as e:
                self.logger.warning(f"Error using LLM for requirement extraction: {e}")
        
        # Fallback to regex-based extraction
        requirements = self._extract_requirements_regex(jd_text)
        
        # Expand requirements using synonyms
        expanded_requirements = self._expand_skills_with_synonyms(requirements)
        # Filter out generic/common words and section headers
        filtered_requirements = self._filter_skills(expanded_requirements)
        return filtered_requirements
    
    def _extract_requirements_regex(self, jd_text: str) -> List[str]:
        """Extract requirements using regex patterns."""
        requirements = set()
        
        # Common technical skills and technologies
        tech_skills = [
            r'\bPython\b', r'\bJava\b', r'\bJavaScript\b', r'\bSQL\b', r'\bR\b',
            r'\bMachine Learning\b', r'\bDeep Learning\b', r'\bData Science\b',
            r'\bBig Data\b', r'\bHadoop\b', r'\bSpark\b', r'\bTensorFlow\b',
            r'\bPyTorch\b', r'\bScikit-learn\b', r'\bPandas\b', r'\bNumPy\b',
            r'\bTableau\b', r'\bPower BI\b', r'\bAWS\b', r'\bAzure\b', r'\bGCP\b',
            r'\bDocker\b', r'\bKubernetes\b', r'\bGit\b', r'\bCI/CD\b'
        ]
        
        # Extract technical skills
        for pattern in tech_skills:
            matches = re.finditer(pattern, jd_text, re.IGNORECASE)
            for match in matches:
                requirements.add(match.group(0))
        
        # Try to find requirements section
        req_patterns = [
            r'(?i)requirements\s*:([\s\S]*?)(?:\n\n|\Z)',
            r'(?i)qualifications\s*:([\s\S]*?)(?:\n\n|\Z)',
            r'(?i)skills\s*required\s*:([\s\S]*?)(?:\n\n|\Z)',
            r'(?i)what\s*you\'ll\s*need\s*:([\s\S]*?)(?:\n\n|\Z)'
        ]
        
        requirements_section = ""
        for pattern in req_patterns:
            match = re.search(pattern, jd_text)
            if match:
                requirements_section = match.group(1).strip()
                break
        
        if not requirements_section:
            requirements_section = jd_text
        
        # Extract bullet points
        bullet_points = re.findall(r'[•\-*]\s*([^•\-\*\n]+)', requirements_section)
        if bullet_points:
            for point in bullet_points:
                # Extract individual skills from bullet points
                skills = re.findall(r'\b[A-Z][A-Za-z+#]+\b', point)
                requirements.update(skills)
        
        # Extract lines
        lines = requirements_section.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('•', '-', '*', '#')):
                # Extract individual skills from lines
                skills = re.findall(r'\b[A-Z][A-Za-z+#]+\b', line)
                requirements.update(skills)
        
        return list(requirements)
    
    def compute_alignment(self, resume_text: str, job_description: str, sections: Dict[str, str]) -> Dict[str, Any]:
        """Compute alignment between resume and job description.
        
        Args:
            resume_text (str): Full resume text
            job_description (str): Job description text
            sections (Dict[str, str]): Resume sections
            
        Returns:
            Dict[str, Any]: Alignment results
        """
        try:
            # Extract skills and requirements
            skills = self.extract_skills(resume_text)
            requirements = self.extract_jd_requirements(job_description)
            
            if not skills or not requirements:
                return {
                    "overall_alignment_score": 0,
                    "matching_skills": [],
                    "missing_skills": requirements if requirements else [],
                    "skill_scores": {},
                    "requirement_scores": {},
                    "error": "No skills or requirements found"
                }
            
            # Compute alignment using embeddings if available
            if self.use_embeddings:
                results = self._compute_alignment_with_embeddings(skills, requirements)
            else:
                results = self._compute_alignment_with_tfidf(skills, requirements)
            
            # Add section scores
            section_scores = self._calculate_section_scores(sections, job_description)
            results["section_scores"] = section_scores
            
            # Ensure overall_alignment_score is present for downstream reporting
            if "overall_alignment" in results:
                results["overall_alignment_score"] = results["overall_alignment"]
            return results
            
        except Exception as e:
            self.logger.error(f"Error computing alignment: {e}")
            return {
                "overall_alignment_score": 0,
                "matching_skills": [],
                "missing_skills": [],
                "skill_scores": {},
                "requirement_scores": {},
                "error": str(e)
            }
    
    def _compute_alignment_with_embeddings(self, candidate_skills, jd_requirements):
        """Compute alignment using sentence embeddings.
        
        Args:
            candidate_skills (list): List of candidate's skills.
            jd_requirements (list): List of job requirements.
            
        Returns:
            dict: Alignment results.
        """
        # Encode skills and requirements
        skill_embeddings = self.model.encode(candidate_skills)
        req_embeddings = self.model.encode(jd_requirements)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(skill_embeddings, req_embeddings)
        
        # For each requirement, find the best matching skill
        req_scores = {}
        for i, req in enumerate(jd_requirements):
            best_score = np.max(similarity_matrix[:, i]) if similarity_matrix.shape[0] > 0 else 0
            req_scores[req] = best_score
        
        # For each skill, find the best matching requirement
        skill_scores = {}
        for i, skill in enumerate(candidate_skills):
            best_score = np.max(similarity_matrix[i, :]) if similarity_matrix.shape[1] > 0 else 0
            skill_scores[skill] = best_score
        
        # Identify missing skills (requirements with low match scores)
        threshold = 0.45  # Slightly lower threshold to be more lenient
        missing_skills = [req for req, score in req_scores.items() if score < threshold]
        
        # Calculate overall alignment score
        overall_alignment = np.mean(list(req_scores.values())) * 100 if req_scores else 0
        
        return {
            "overall_alignment": overall_alignment,
            "skill_scores": skill_scores,
            "requirement_scores": req_scores,
            "missing_skills": missing_skills,
            "candidate_skills": candidate_skills,
            "jd_requirements": jd_requirements
        }
    
    def _compute_alignment_with_tfidf(self, candidate_skills, jd_requirements):
        """Compute alignment using TF-IDF and cosine similarity.
        
        Args:
            candidate_skills (list): List of candidate's skills.
            jd_requirements (list): List of job requirements.
            
        Returns:
            dict: Alignment results.
        """
        # Combine skills and requirements for TF-IDF
        all_texts = candidate_skills + jd_requirements
        
        # Compute TF-IDF matrix
        tfidf_matrix = self.tfidf.fit_transform(all_texts)
        
        # Split matrix back into skills and requirements
        skill_vectors = tfidf_matrix[:len(candidate_skills)]
        req_vectors = tfidf_matrix[len(candidate_skills):]
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(skill_vectors, req_vectors)
        
        # For each requirement, find the best matching skill
        req_scores = {}
        for i, req in enumerate(jd_requirements):
            best_score = np.max(similarity_matrix[:, i]) if similarity_matrix.shape[0] > 0 else 0
            req_scores[req] = best_score
        
        # For each skill, find the best matching requirement
        skill_scores = {}
        for i, skill in enumerate(candidate_skills):
            best_score = np.max(similarity_matrix[i, :]) if similarity_matrix.shape[1] > 0 else 0
            skill_scores[skill] = best_score
        
        # Identify missing skills (requirements with low match scores)
        threshold = 0.3  # Lower threshold for TF-IDF
        missing_skills = [req for req, score in req_scores.items() if score < threshold]
        
        # Calculate overall alignment score
        overall_alignment = np.mean(list(req_scores.values())) * 100 if req_scores else 0
        
        return {
            "overall_alignment": overall_alignment,
            "skill_scores": skill_scores,
            "requirement_scores": req_scores,
            "missing_skills": missing_skills,
            "candidate_skills": candidate_skills,
            "jd_requirements": jd_requirements
        }
    
    def _calculate_section_scores(self, sections, jd_text):
        """Calculate weighted scores for each section based on job description relevance.
        
        Args:
            sections (dict): Dictionary of resume sections.
            jd_text (str): Job description text.
            
        Returns:
            dict: Section scores and total weighted score.
        """
        # Define section weights
        section_weights = {
            'Summary': 0.15,
            'Education': 0.15,
            'Work Experience': 0.30,
            'Skills': 0.25,
            'Projects': 0.10,
            'Certifications': 0.05
        }
        
        # Initialize scores
        section_scores = {}
        total_score = 0
        total_weight = 0
        
        # Calculate score for each section
        for section_name, content in sections.items():
            if section_name in section_weights:
                # Extract requirements from job description
                jd_requirements = self.extract_jd_requirements(jd_text)
                
                # Calculate similarity between section content and job requirements
                if self.model:
                    # Use sentence embeddings
                    section_embedding = self.model.encode([content])[0]
                    req_embeddings = self.model.encode(jd_requirements)
                    similarities = cosine_similarity([section_embedding], req_embeddings)[0]
                    section_score = np.mean(similarities) * 100 if len(similarities) > 0 else 0
                else:
                    # Use TF-IDF
                    all_texts = [content] + jd_requirements
                    tfidf_matrix = self.tfidf.fit_transform(all_texts)
                    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
                    section_score = np.mean(similarities) * 100 if len(similarities) > 0 else 0
                
                # Store section score
                section_scores[section_name] = section_score
                
                # Add to total weighted score
                weight = section_weights[section_name]
                total_score += section_score * weight
                total_weight += weight
        
        # Normalize total score
        if total_weight > 0:
            total_score = total_score / total_weight
        
        return {
            "section_scores": section_scores,
            "total_score": total_score
        }
    
    def save_results(self, results, output_path):
        """Save alignment results to a JSON file.
        
        Args:
            results (dict): Alignment results.
            output_path (str): Path to save the results.
        """
        def convert_numpy_values(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_values(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_values(item) for item in obj]
            return obj
        
        # Convert numpy values to Python types
        results = convert_numpy_values(results)
        
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def visualize_alignment(self, results, output_path):
        """Create a visualization of the alignment results.
        
        Args:
            results (dict): Alignment results.
            output_path (str): Path to save the visualization.
        """
        # Extract data for visualization
        skill_scores = results.get('skill_scores', {})
        req_scores = results.get('requirement_scores', {})
        
        if not skill_scores or not req_scores:
            print("No data available for visualization")
            return
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot skill scores
        skills = list(skill_scores.keys())
        scores = list(skill_scores.values())
        
        ax1.barh(skills, scores)
        ax1.set_xlabel('Match Score')
        ax1.set_title('Skill Match Scores')
        ax1.set_xlim(0, 1)
        
        # Plot requirement scores
        reqs = list(req_scores.keys())
        req_scores_list = list(req_scores.values())
        
        ax2.barh(reqs, req_scores_list)
        ax2.set_xlabel('Match Score')
        ax2.set_title('Requirement Match Scores')
        ax2.set_xlim(0, 1)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def recalculate_alignment(self, resume_text, job_description, sections, 
                           trustworthiness_results, credibility_results, link_extraction_results):
        """Recalculate alignment with additional context.
        
        Args:
            resume_text (str): Full resume text.
            job_description (str): Job description text.
            sections (dict): Dictionary of resume sections.
            trustworthiness_results (dict): Trustworthiness analysis results.
            credibility_results (dict): Credibility analysis results.
            link_extraction_results (dict): Link extraction results.
            
        Returns:
            dict: Updated alignment results.
        """
        # Get base alignment results
        base_results = self.compute_alignment(
            sections.get('Skills', ''),
            job_description,
            sections
        )
        
        # Adjust scores based on trustworthiness and credibility
        if trustworthiness_results and credibility_results:
            trust_score = trustworthiness_results.get('overall_score', 0.5)
            cred_score = credibility_results.get('overall_score', 0.5)
            
            # Adjust overall alignment score
            base_results['overall_alignment'] *= (trust_score * cred_score)
            base_results['overall_alignment_score'] = base_results['overall_alignment']
            
            # Adjust section scores
            if 'section_scores' in base_results:
                for section in base_results['section_scores']['section_scores']:
                    base_results['section_scores']['section_scores'][section] *= (trust_score * cred_score)
                base_results['section_scores']['total_score'] *= (trust_score * cred_score)
        
        return base_results