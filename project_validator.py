#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Project Validation System for Resume Intelligence System

This module verifies that each project entry truly reflects the claimed skillset,
scoring projects on both technical depth and relevance.
"""

import json
import re
from pathlib import Path

import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ProjectValidator:
    """Validator for verifying project entries against claimed skills."""
    
    def __init__(self):
        """Initialize the project validator."""
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_lg")
        except OSError:
            # If model is not installed, use a smaller one
            print("Warning: en_core_web_lg not found. Using en_core_web_sm instead.")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Warning: No spaCy models found. Downloading en_core_web_sm...")
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(stop_words='english')
        
        # Define action verbs commonly used in project descriptions
        self.action_verbs = [
            "developed", "created", "designed", "implemented", "built", "architected",
            "engineered", "programmed", "coded", "constructed", "established", "formulated",
            "devised", "conceived", "initiated", "launched", "spearheaded", "directed",
            "led", "managed", "coordinated", "supervised", "orchestrated", "oversaw",
            "analyzed", "evaluated", "assessed", "examined", "investigated", "researched",
            "tested", "debugged", "troubleshot", "resolved", "fixed", "solved",
            "improved", "enhanced", "optimized", "upgraded", "streamlined", "refactored",
            "maintained", "supported", "administered", "operated", "monitored"
        ]
    
    def extract_projects(self, projects_text):
        """Extract individual projects from projects section text.
        
        Args:
            projects_text (str): Text from the projects section.
            
        Returns:
            dict: Dictionary mapping project titles to their descriptions.
        """
        projects = {}
        
        # Split text into lines
        lines = projects_text.split('\n')
        
        current_project = None
        current_description = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line looks like a project title
            if (line.endswith(':') or 
                any(verb in line.lower() for verb in self.action_verbs) or
                re.match(r'^[A-Z][^\n.]*(?:\.|:)$', line)):
                
                # Save previous project if exists
                if current_project and current_description:
                    projects[current_project] = '\n'.join(current_description)
                
                # Start new project
                current_project = line.rstrip(':.')
                current_description = []
            elif current_project:
                current_description.append(line)
        
        # Add the last project
        if current_project and current_description:
            projects[current_project] = '\n'.join(current_description)
        
        # If no projects were identified, try a different approach
        if not projects:
            # Look for bullet points or numbered lists
            project_blocks = re.split(r'\n\s*(?:•|\*|\-|\d+\.)\s+', '\n' + projects_text)
            
            if len(project_blocks) > 1:
                for i, block in enumerate(project_blocks[1:], 1):  # Skip the first split which is empty
                    if block.strip():
                        # Use first line as title, rest as description
                        lines = block.split('\n', 1)
                        title = lines[0].strip()
                        desc = lines[1].strip() if len(lines) > 1 else ""
                        projects[f"Project {i}: {title}"] = desc
        
        return projects
    
    def extract_skills(self, skills_text):
        """Extract skills from skills section text.
        
        Args:
            skills_text (str): Text from the skills section.
            
        Returns:
            list: List of extracted skills.
        """
        # Split by common delimiters and clean up
        skills = re.split(r'[,;•\n]', skills_text)
        skills = [skill.strip() for skill in skills if skill.strip()]
        
        return skills
    
    def extract_action_tech_pairs(self, project_description):
        """Extract action-verb-technology pairs from project description.
        
        Args:
            project_description (str): Description of a project.
            
        Returns:
            list: List of (action_verb, technology) pairs.
        """
        pairs = []
        doc = self.nlp(project_description)
        
        # Process each sentence
        for sent in doc.sents:
            # Find action verbs
            action_verbs = [token for token in sent 
                           if token.lemma_.lower() in self.action_verbs or 
                              token.pos_ == "VERB"]
            
            # Find potential technologies (nouns, proper nouns, and adjectives followed by nouns)
            techs = []
            for token in sent:
                if token.pos_ in ["NOUN", "PROPN"]:
                    # Check if it's a compound noun
                    if any(child.dep_ == "compound" for child in token.children):
                        compound = ' '.join([child.text for child in token.children 
                                           if child.dep_ == "compound"])
                        techs.append(f"{compound} {token.text}")
                    else:
                        techs.append(token.text)
            
            # Create pairs
            for verb in action_verbs:
                for tech in techs:
                    pairs.append((verb.text, tech))
        
        return pairs
    
    def validate_projects(self, projects_text, skills_text):
        """Validate projects against claimed skills.
        
        Args:
            projects_text (str): Text from the projects section.
            skills_text (str): Text from the skills section.
            
        Returns:
            dict: Validation results including project scores and flagged projects.
        """
        # Extract projects and skills
        projects = self.extract_projects(projects_text)
        skills = self.extract_skills(skills_text)
        
        if not projects or not skills:
            return {
                "project_scores": {},
                "flagged_projects": [],
                "projects": projects,
                "skills": skills,
                "validation_metrics": {}
            }
        
        # Prepare results
        results = {
            "project_scores": {},
            "flagged_projects": [],
            "projects": projects,
            "skills": skills,
            "validation_metrics": {}
        }
        
        # Validate each project
        for project_title, description in projects.items():
            # Extract action-technology pairs
            action_tech_pairs = self.extract_action_tech_pairs(description)
            
            # Extract technologies mentioned
            technologies = [tech for _, tech in action_tech_pairs]
            
            # Calculate technical depth score (based on number of action-tech pairs)
            # Enhanced technical depth evaluation
            # Check for technical keywords in the project description with more comprehensive list
            tech_keywords = [
                'algorithm', 'implemented', 'developed', 'architecture', 'designed', 'optimized',
                'engineered', 'built', 'created', 'constructed', 'programmed', 'coded',
                'deployed', 'integrated', 'configured', 'maintained', 'refactored', 'improved',
                'analyzed', 'debugged', 'tested', 'validated', 'benchmarked', 'profiled',
                'scaled', 'distributed', 'parallelized', 'containerized', 'automated'
            ]
            
            # Weight different technical terms differently
            tech_term_weights = {
                'architecture': 1.5,
                'algorithm': 1.5,
                'optimized': 1.3,
                'engineered': 1.3,
                'deployed': 1.2,
                'scaled': 1.2,
                'distributed': 1.2,
                'parallelized': 1.2,
                'containerized': 1.2,
                'automated': 1.1
            }
            
            # Calculate weighted technical depth
            tech_score = 0
            for keyword in tech_keywords:
                if keyword in description.lower():
                    tech_score += tech_term_weights.get(keyword, 1.0)
            
            # Also consider action-tech pairs in the score
            tech_score += len(action_tech_pairs) / 2
            
            # Normalize and cap at 1.0
            tech_depth_score = min(1.0, tech_score / 5)  # Cap at 1.0
            
            # Calculate relevance score (based on overlap with skills)
            relevance_score = self._calculate_relevance_score(technologies, skills)
            
            # Calculate overall score (weighted average)
            overall_score = 0.4 * tech_depth_score + 0.6 * relevance_score
            
            # Store score and metrics
            results["project_scores"][project_title] = overall_score
            results["validation_metrics"][project_title] = {
                "skill_alignment": relevance_score,
                "technical_depth": tech_depth_score,
                "quantifiable_results": min(1.0, len([p for p in description.split() if p.isdigit() or p.endswith('%')]) / 3)
            }
            
            # Flag project if score is too low
            if overall_score < 0.4:
                if relevance_score < 0.3:
                    results["flagged_projects"].append(f"{project_title} (Unrelated to claimed skills)")
                else:
                    results["flagged_projects"].append(f"{project_title} (Lacks technical depth)")
        
        # Calculate overall project validation score (average of all project scores)
        if results["project_scores"]:
            # Calculate average score (0-1 range) and scale to 0-100 range
            results["overall_score"] = (sum(results["project_scores"].values()) / len(results["project_scores"]) if results["project_scores"] else 0.0) * 100
        else:
            results["overall_score"] = 0.0
            
        return results
    
    def _calculate_relevance_score(self, technologies, skills):
        """Calculate relevance score based on overlap between technologies and skills.
        
        Args:
            technologies (list): Technologies mentioned in project.
            skills (list): Claimed skills.
            
        Returns:
            float: Relevance score between 0 and 1.
        """
        if not technologies or not skills:
            return 0.0
        
        # Enhanced skill alignment calculation
        # 1. Check for exact skill mentions
        exact_mentions = 0
        for skill in skills:
            for tech in technologies:
                if re.search(r'\b' + re.escape(skill.lower()) + r'\b', tech.lower()):
                    exact_mentions += 1
                    break
        
        exact_score = exact_mentions / len(skills) if skills else 0
        
        # 2. Use TF-IDF for semantic similarity
        try:
            # Combine all texts for TF-IDF
            all_texts = technologies + skills
            
            # Compute TF-IDF matrix
            tfidf_matrix = self.tfidf.fit_transform(all_texts)
            
            # Split matrix back into technologies and skills
            tech_vectors = tfidf_matrix[:len(technologies)]
            skill_vectors = tfidf_matrix[len(technologies):]
            
            # Compute similarity matrix
            similarity_matrix = cosine_similarity(tech_vectors, skill_vectors)
            
            # Calculate average of maximum similarities
            max_similarities = np.max(similarity_matrix, axis=1)
            semantic_score = np.mean(max_similarities)
            
            # Combine exact matches and semantic alignment
            relevance_score = 0.7 * exact_score + 0.3 * semantic_score
            
            return relevance_score
        except Exception as e:
            print(f"Error calculating relevance score: {e}")
            
            # Fallback: simple word overlap
            tech_words = set(' '.join(technologies).lower().split())
            skill_words = set(' '.join(skills).lower().split())
            
            overlap = len(tech_words.intersection(skill_words))
            total = len(tech_words.union(skill_words))
            
            return overlap / total if total > 0 else 0.0
            
    def _extract_tech_keywords(self, text):
        """Extract technology keywords from text.
        
        Args:
            text (str): Text to extract keywords from.
            
        Returns:
            list: List of extracted technology keywords.
        """
        # Extract noun phrases as potential technology keywords
        doc = self.nlp(text)
        keywords = []
        
        # Extract noun chunks (noun phrases)
        for chunk in doc.noun_chunks:
            keywords.append(chunk.text)
        
        # Also extract named entities that might be technologies
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT"]:
                keywords.append(ent.text)
        
        # Add individual technical nouns
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and token.text.lower() not in [k.lower() for k in keywords]:
                keywords.append(token.text)
        
        return keywords
    
    def _compute_project_skill_alignment(self, project_title, description, skills):
        """Compute alignment between project and claimed skills.
        
        Args:
            project_title (str): Title of the project.
            description (str): Description of the project.
            skills (list): List of claimed skills.
            
        Returns:
            float: Skill alignment score (0-1).
        """
        if not skills:
            return 0.0
        
        # Calculate skill alignment score (0-1)
        # Check if any of the claimed skills are mentioned in the project
        skill_alignment = 0
        
        # Extract technologies from project description
        tech_keywords = self._extract_tech_keywords(description)
        
        # Check for direct skill mentions in project title or description
        for skill in skills:
            skill_lower = skill.lower()
            # Check for direct mentions in title or description
            if skill_lower in project_title.lower() or skill_lower in description.lower():
                skill_alignment = 1
                break
            # Check for semantic similarity with extracted tech keywords
            for keyword in tech_keywords:
                # Check for partial matches (e.g., "API" in "Gemini API")
                if (skill_lower in keyword.lower() or 
                    keyword.lower() in skill_lower or
                    any(term in keyword.lower() for term in skill_lower.split())):
                    skill_alignment = max(skill_alignment, 0.8)  # Partial match gets 0.8 score
        
        return skill_alignment
        
    def _evaluate_project_quality(self, project_title, description, skills):
        """Evaluate project quality based on multiple metrics.
        
        Args:
            project_title (str): Project title.
            description (str): Project description.
            skills (list): List of skills.
            
        Returns:
            dict: Dictionary of evaluation metrics.
        """
        # Combine title and description
        project_text = f"{project_title} {description}"
        project_doc = self.nlp(project_text)
        
        # Initialize metrics
        metrics = {
            "skill_alignment": 0.0,
            "technical_depth": 0.0,
            "quantifiable_results": 0.0,
            "implementation_details": 0.0,
            "complexity": 0.0
        }
        
        # 1. Skill alignment - how well the project aligns with claimed skills
        metrics["skill_alignment"] = self._compute_project_skill_alignment(project_title, description, skills)
        
        # 2. Technical depth - presence of technical terms and implementation details
        tech_terms = [
            "algorithm", "implemented", "developed", "built", "designed", "architecture",
            "database", "API", "framework", "library", "system", "infrastructure",
            "backend", "frontend", "full-stack", "cloud", "deployment", "containerization",
            "testing", "CI/CD", "version control", "optimization", "scalability"
        ]
        
        tech_term_count = sum(1 for term in tech_terms if term.lower() in project_text.lower())
        metrics["technical_depth"] = min(1.0, tech_term_count / 5)  # Cap at 1.0
        
        # 3. Quantifiable results - presence of numbers, percentages, metrics
        # Look for digits and percentage signs
        quantifiable_patterns = [
            r'\d+%',  # Percentages
            r'\d+x',  # Multipliers
            r'reduced by \d+',  # Reductions
            r'improved by \d+',  # Improvements
            r'increased \d+',  # Increases
            r'decreased \d+',  # Decreases
            r'\d+ times',  # Multipliers
            r'\$\d+',  # Dollar amounts
        ]
        
        quantifiable_count = sum(1 for pattern in quantifiable_patterns if re.search(pattern, project_text.lower()))
        metrics["quantifiable_results"] = min(1.0, quantifiable_count / 2)  # Cap at 1.0
        
        # 4. Implementation details - specific technologies mentioned
        # Extract entities that might be technologies
        tech_entities = [ent.text for ent in project_doc.ents if ent.label_ in ["ORG", "PRODUCT"]]
        
        # Add common technologies that might not be recognized as entities
        common_techs = [
            "Python", "Java", "JavaScript", "React", "Angular", "Vue", "Node.js",
            "Express", "Django", "Flask", "Spring", "Hibernate", "SQL", "NoSQL",
            "MongoDB", "PostgreSQL", "MySQL", "Redis", "Elasticsearch", "AWS", "Azure",
            "GCP", "Docker", "Kubernetes", "Jenkins", "Git", "GitHub", "GitLab",
            "TensorFlow", "PyTorch", "scikit-learn", "pandas", "NumPy", "OpenCV",
            "NLTK", "spaCy", "Transformers", "BERT", "GPT", "REST", "GraphQL",
            "WebSockets", "gRPC", "Kafka", "RabbitMQ", "Celery", "Redux", "MobX",
            "Vuex", "CSS", "SASS", "LESS", "Tailwind", "Bootstrap", "Material-UI"
        ]
        
        tech_count = sum(1 for tech in common_techs if tech.lower() in project_text.lower())
        tech_count += len(tech_entities)
        metrics["implementation_details"] = min(1.0, tech_count / 3)  # Cap at 1.0
        
        # 5. Project complexity - based on sentence structure and length
        sentence_count = len(list(project_doc.sents))
        avg_sentence_length = len(project_text) / max(1, sentence_count)
        metrics["complexity"] = min(1.0, avg_sentence_length / 100)  # Cap at 1.0
        
        return metrics
    
    def save_results(self, results, output_path):
        """Save validation results to a JSON file.
        
        Args:
            results (dict): Validation results.
            output_path (str or Path): Path to save the JSON file.
        """
        # Convert NumPy values to native Python types before serialization
        def convert_numpy_values(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_values(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_values(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return convert_numpy_values(obj.tolist())
            else:
                return obj
                
        # Apply conversion to results
        serializable_results = convert_numpy_values(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2)