import hashlib
import re
import requests
import json
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
import logging
from datetime import datetime, timedelta
import time

class TrustworthinessDetector:
    """
    Trustworthiness & Inflation Detector: Uncover inflated claims, copied text, 
    or "skill > evidence" mismatches in the résumé.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        self.github_api_base = "https://api.github.com"
        self.rate_limit_delay = 1  # seconds between API calls
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration for trustworthiness detection"""
        return {
            "min_hash_similarity_threshold": 0.8,
            "skill_evidence_mismatch_threshold": 0.3,
            "github_check_enabled": True,
            "linkedin_check_enabled": False,  # Requires API access
            "common_phrases_penalty": 0.1,
            "inflation_keywords": [
                "expert", "guru", "ninja", "rockstar", "wizard", "master",
                "extensive experience", "deep expertise", "comprehensive knowledge",
                "cutting-edge", "state-of-the-art", "revolutionary", "groundbreaking"
            ],
            "evidence_keywords": [
                "implemented", "developed", "created", "built", "designed",
                "optimized", "improved", "reduced", "increased", "achieved",
                "delivered", "managed", "led", "collaborated"
            ],
            "quantifiable_patterns": [
                r'\d+%', r'\$\d+', r'\d+\s*(users|customers|clients)',
                r'\d+\s*(hours|days|weeks|months)', r'\d+x\s*faster',
                r'\d+\s*(projects|applications|systems)'
            ]
        }
    
    def analyze_trustworthiness(self, resume_text: str, sections: Dict[str, str], 
                              skills: List[str], github_username: str = None) -> Dict[str, Any]:
        """
        Comprehensive trustworthiness analysis
        
        Args:
            resume_text: Full resume text
            sections: Parsed resume sections
            skills: Extracted skills list
            github_username: Optional GitHub username for verification
            
        Returns:
            Dictionary containing trustworthiness scores and analysis
        """
        try:
            # Text duplication analysis
            duplication_analysis = self._analyze_text_duplication(resume_text)
            
            # Skill-evidence mismatch analysis
            skill_evidence_analysis = self._analyze_skill_evidence_mismatch(sections, skills)
            
            # Inflation detection
            inflation_analysis = self._detect_inflation_patterns(resume_text, sections)
            
            # External verification (if username provided)
            external_verification = {}
            if github_username and self.config["github_check_enabled"]:
                external_verification = self._verify_github_activity(github_username, skills)
            
            # Calculate overall trustworthiness score
            trust_score = self._calculate_trust_score(
                duplication_analysis,
                skill_evidence_analysis,
                inflation_analysis,
                external_verification
            )
            
            return {
                "trust_score": trust_score,
                "max_score": 100,
                "analysis_details": {
                    "text_duplication": duplication_analysis,
                    "skill_evidence_mismatch": skill_evidence_analysis,
                    "inflation_detection": inflation_analysis,
                    "external_verification": external_verification
                },
                "flags": self._generate_trust_flags(
                    duplication_analysis,
                    skill_evidence_analysis,
                    inflation_analysis,
                    external_verification
                ),
                "recommendations": self._generate_trust_recommendations(
                    duplication_analysis,
                    skill_evidence_analysis,
                    inflation_analysis
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error in trustworthiness analysis: {e}")
            return self._get_default_trust_score()
    
    def _analyze_text_duplication(self, text: str) -> Dict[str, Any]:
        """Detect potential text duplication using hash-based fingerprinting"""
        # Split text into sentences and paragraphs
        sentences = re.split(r'[.!?]+', text)
        paragraphs = text.split('\n\n')
        
        # Generate hashes for sentences
        sentence_hashes = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Only analyze substantial sentences
                hash_obj = hashlib.md5(sentence.lower().encode())
                sentence_hashes.append(hash_obj.hexdigest())
        
        # Check for duplicates
        hash_counter = Counter(sentence_hashes)
        duplicates = {h: count for h, count in hash_counter.items() if count > 1}
        
        # Analyze common phrases
        common_phrases = self._find_common_phrases(text)
        
        return {
            "total_sentences": len([s for s in sentences if s.strip()]),
            "duplicate_sentences": len(duplicates),
            "duplication_ratio": len(duplicates) / max(len(sentence_hashes), 1),
            "common_phrases": common_phrases,
            "suspicious_patterns": self._detect_template_patterns(text)
        }
    
    def _find_common_phrases(self, text: str) -> List[Dict[str, Any]]:
        """Find commonly used phrases that might indicate template usage"""
        common_resume_phrases = [
            "responsible for", "worked on", "involved in", "participated in",
            "team player", "detail-oriented", "self-motivated", "results-driven",
            "excellent communication skills", "problem-solving skills",
            "fast learner", "ability to work", "strong background"
        ]
        
        found_phrases = []
        text_lower = text.lower()
        
        for phrase in common_resume_phrases:
            count = text_lower.count(phrase)
            if count > 0:
                found_phrases.append({
                    "phrase": phrase,
                    "count": count,
                    "severity": "high" if count > 2 else "medium" if count > 1 else "low"
                })
        
        return found_phrases
    
    def _detect_template_patterns(self, text: str) -> List[str]:
        """Detect patterns that suggest template usage"""
        patterns = []
        
        # Check for placeholder patterns
        if re.search(r'\[.*?\]|\{.*?\}|<.*?>', text):
            patterns.append("Contains placeholder brackets")
        
        # Check for repeated formatting patterns
        bullet_patterns = re.findall(r'^[•▪▫◦‣\-\*]\s', text, re.MULTILINE)
        if len(set(bullet_patterns)) == 1 and len(bullet_patterns) > 10:
            patterns.append("Highly repetitive bullet formatting")
        
        # Check for generic job descriptions
        generic_phrases = ["various tasks", "as needed", "and other duties", "etc."]
        for phrase in generic_phrases:
            if phrase in text.lower():
                patterns.append(f"Generic language: '{phrase}'")
        
        return patterns
    
    def _analyze_skill_evidence_mismatch(self, sections: Dict[str, str], skills: List[str]) -> Dict[str, Any]:
        """Analyze mismatches between claimed skills and evidence"""
        if not sections or not skills:
            return {"analysis_possible": False}
        
        # Get experience and project sections
        experience_text = ""
        for section_name, content in sections.items():
            if any(keyword in section_name.lower() for keyword in ["experience", "work", "employment", "project"]):
                experience_text += content + " "
        
        skill_evidence = {}
        evidence_keywords = self.config["evidence_keywords"]
        
        for skill in skills:
            skill_lower = skill.lower()
            evidence_count = 0
            
            # Count evidence mentions for this skill
            for keyword in evidence_keywords:
                # Look for patterns like "developed Python applications"
                pattern = rf'\b{keyword}\b.*?\b{re.escape(skill_lower)}\b|\b{re.escape(skill_lower)}\b.*?\b{keyword}\b'
                matches = re.findall(pattern, experience_text.lower(), re.IGNORECASE)
                evidence_count += len(matches)
            
            # Check for quantifiable results
            quantifiable_evidence = 0
            for pattern in self.config["quantifiable_patterns"]:
                if re.search(pattern, experience_text, re.IGNORECASE):
                    quantifiable_evidence += 1
            
            skill_evidence[skill] = {
                "evidence_mentions": evidence_count,
                "quantifiable_evidence": quantifiable_evidence,
                "evidence_ratio": evidence_count / max(experience_text.count(skill_lower), 1)
            }
        
        # Identify skills with low evidence
        low_evidence_skills = [
            skill for skill, data in skill_evidence.items()
            if data["evidence_ratio"] < self.config["skill_evidence_mismatch_threshold"]
        ]
        
        return {
            "analysis_possible": True,
            "skill_evidence_mapping": skill_evidence,
            "low_evidence_skills": low_evidence_skills,
            "mismatch_ratio": len(low_evidence_skills) / max(len(skills), 1),
            "total_skills_analyzed": len(skills)
        }
    
    def _detect_inflation_patterns(self, text: str, sections: Dict[str, str]) -> Dict[str, Any]:
        """Detect inflated claims and exaggerated language"""
        inflation_keywords = self.config["inflation_keywords"]
        text_lower = text.lower()
        
        # Count inflation keywords
        inflation_matches = []
        for keyword in inflation_keywords:
            count = text_lower.count(keyword.lower())
            if count > 0:
                inflation_matches.append({
                    "keyword": keyword,
                    "count": count,
                    "severity": "high" if keyword in ["expert", "guru", "master"] else "medium"
                })
        
        # Analyze superlative usage
        superlatives = re.findall(r'\b(best|top|leading|premier|ultimate|perfect|flawless)\b', text_lower)
        
        # Check for unrealistic claims
        unrealistic_patterns = [
            r'100%\s*(success|accuracy|efficiency)',
            r'zero\s*(errors|bugs|downtime)',
            r'never\s*(failed|missed)',
            r'always\s*(delivered|succeeded)'
        ]
        
        unrealistic_claims = []
        for pattern in unrealistic_patterns:
            matches = re.findall(pattern, text_lower)
            unrealistic_claims.extend(matches)
        
        # Calculate inflation score
        total_words = len(text.split())
        inflation_density = (len(inflation_matches) + len(superlatives) + len(unrealistic_claims)) / max(total_words, 1)
        
        return {
            "inflation_keywords": inflation_matches,
            "superlatives_count": len(superlatives),
            "unrealistic_claims": unrealistic_claims,
            "inflation_density": inflation_density,
            "severity_level": "high" if inflation_density > 0.02 else "medium" if inflation_density > 0.01 else "low"
        }
    
    def _verify_github_activity(self, username: str, skills: List[str]) -> Dict[str, Any]:
        """Verify GitHub activity against claimed skills"""
        try:
            time.sleep(self.rate_limit_delay)  # Rate limiting
            
            # Get user profile
            user_response = requests.get(f"{self.github_api_base}/users/{username}", timeout=10)
            if user_response.status_code != 200:
                return {"verification_possible": False, "reason": "User not found"}
            
            user_data = user_response.json()
            
            # Get repositories
            repos_response = requests.get(f"{self.github_api_base}/users/{username}/repos", timeout=10)
            if repos_response.status_code != 200:
                return {"verification_possible": False, "reason": "Cannot access repositories"}
            
            repos_data = repos_response.json()
            
            # Analyze repositories
            languages_used = Counter()
            total_repos = len(repos_data)
            recent_activity = 0
            
            cutoff_date = datetime.now() - timedelta(days=365)  # Last year
            
            for repo in repos_data:
                # Count languages
                if repo.get('language'):
                    languages_used[repo['language']] += 1
                
                # Check recent activity
                updated_at = datetime.strptime(repo['updated_at'], '%Y-%m-%dT%H:%M:%SZ')
                if updated_at > cutoff_date:
                    recent_activity += 1
            
            # Match skills with GitHub activity
            skill_verification = {}
            for skill in skills:
                skill_lower = skill.lower()
                github_evidence = 0
                
                # Check if skill appears in languages
                for lang, count in languages_used.items():
                    if skill_lower in lang.lower() or lang.lower() in skill_lower:
                        github_evidence += count
                
                # Check repository names and descriptions
                for repo in repos_data:
                    repo_text = f"{repo.get('name', '')} {repo.get('description', '')}".lower()
                    if skill_lower in repo_text:
                        github_evidence += 1
                
                skill_verification[skill] = {
                    "github_evidence": github_evidence,
                    "verification_level": "high" if github_evidence > 3 else "medium" if github_evidence > 0 else "low"
                }
            
            return {
                "verification_possible": True,
                "profile_data": {
                    "public_repos": user_data.get('public_repos', 0),
                    "followers": user_data.get('followers', 0),
                    "account_age_days": (datetime.now() - datetime.strptime(user_data['created_at'], '%Y-%m-%dT%H:%M:%SZ')).days
                },
                "activity_analysis": {
                    "total_repositories": total_repos,
                    "recent_activity_repos": recent_activity,
                    "languages_used": dict(languages_used),
                    "activity_level": "high" if recent_activity > 5 else "medium" if recent_activity > 0 else "low"
                },
                "skill_verification": skill_verification,
                "overall_credibility": self._calculate_github_credibility(user_data, total_repos, recent_activity, skill_verification)
            }
            
        except Exception as e:
            self.logger.error(f"Error verifying GitHub activity: {e}")
            return {"verification_possible": False, "reason": str(e)}
    
    def _calculate_github_credibility(self, user_data: Dict, total_repos: int, 
                                    recent_activity: int, skill_verification: Dict) -> str:
        """Calculate overall GitHub credibility level"""
        score = 0
        
        # Account age (older accounts are more credible)
        account_age = (datetime.now() - datetime.strptime(user_data['created_at'], '%Y-%m-%dT%H:%M:%SZ')).days
        if account_age > 365:
            score += 2
        elif account_age > 180:
            score += 1
        
        # Repository count
        if total_repos > 10:
            score += 2
        elif total_repos > 3:
            score += 1
        
        # Recent activity
        if recent_activity > 5:
            score += 2
        elif recent_activity > 0:
            score += 1
        
        # Skill verification
        verified_skills = sum(1 for data in skill_verification.values() if data["verification_level"] != "low")
        if verified_skills > len(skill_verification) * 0.7:
            score += 2
        elif verified_skills > len(skill_verification) * 0.3:
            score += 1
        
        if score >= 6:
            return "high"
        elif score >= 3:
            return "medium"
        else:
            return "low"
    
    def _calculate_trust_score(self, duplication_analysis: Dict, skill_evidence_analysis: Dict,
                             inflation_analysis: Dict, external_verification: Dict) -> float:
        """Calculate overall trustworthiness score (0-100)"""
        score = 100.0
        
        # Penalize text duplication
        duplication_penalty = duplication_analysis.get("duplication_ratio", 0) * 30
        score -= duplication_penalty
        
        # Penalize common phrases
        common_phrases_penalty = len(duplication_analysis.get("common_phrases", [])) * self.config["common_phrases_penalty"] * 10
        score -= common_phrases_penalty
        
        # Penalize skill-evidence mismatches
        if skill_evidence_analysis.get("analysis_possible", False):
            mismatch_penalty = skill_evidence_analysis.get("mismatch_ratio", 0) * 25
            score -= mismatch_penalty
        
        # Penalize inflation
        inflation_density = inflation_analysis.get("inflation_density", 0)
        inflation_penalty = min(30, inflation_density * 1000)  # Cap at 30 points
        score -= inflation_penalty
        
        # Bonus for external verification
        if external_verification.get("verification_possible", False):
            credibility = external_verification.get("overall_credibility", "low")
            if credibility == "high":
                score += 10
            elif credibility == "medium":
                score += 5
        
        return max(0, min(100, score))
    
    def _generate_trust_flags(self, duplication_analysis: Dict, skill_evidence_analysis: Dict,
                            inflation_analysis: Dict, external_verification: Dict) -> List[Dict[str, str]]:
        """Generate trust-related flags"""
        flags = []
        
        # Duplication flags
        if duplication_analysis.get("duplication_ratio", 0) > 0.1:
            flags.append({
                "type": "duplication",
                "severity": "high",
                "message": "High text duplication detected - possible template usage"
            })
        
        # Common phrases flags
        high_severity_phrases = [p for p in duplication_analysis.get("common_phrases", []) if p["severity"] == "high"]
        if high_severity_phrases:
            flags.append({
                "type": "generic_content",
                "severity": "medium",
                "message": f"Overuse of common phrases: {', '.join([p['phrase'] for p in high_severity_phrases])}"
            })
        
        # Skill-evidence mismatch flags
        if skill_evidence_analysis.get("mismatch_ratio", 0) > 0.5:
            flags.append({
                "type": "skill_mismatch",
                "severity": "high",
                "message": "Many claimed skills lack supporting evidence"
            })
        
        # Inflation flags
        if inflation_analysis.get("severity_level") == "high":
            flags.append({
                "type": "inflation",
                "severity": "high",
                "message": "Excessive use of superlative language and inflated claims"
            })
        
        # External verification flags
        if external_verification.get("verification_possible", False):
            if external_verification.get("overall_credibility") == "low":
                flags.append({
                    "type": "low_credibility",
                    "severity": "medium",
                    "message": "GitHub activity doesn't strongly support claimed skills"
                })
        
        return flags
    
    def _generate_trust_recommendations(self, duplication_analysis: Dict, skill_evidence_analysis: Dict,
                                      inflation_analysis: Dict) -> List[str]:
        """Generate recommendations for improving trustworthiness"""
        recommendations = []
        
        if duplication_analysis.get("duplication_ratio", 0) > 0.05:
            recommendations.append("Avoid repetitive language and template-like content")
            recommendations.append("Write unique, personalized descriptions for each experience")
        
        if len(duplication_analysis.get("common_phrases", [])) > 3:
            recommendations.append("Replace generic phrases with specific, quantifiable achievements")
            recommendations.append("Use action verbs and concrete examples instead of buzzwords")
        
        if skill_evidence_analysis.get("mismatch_ratio", 0) > 0.3:
            recommendations.append("Provide specific examples of how you've used each claimed skill")
            recommendations.append("Include quantifiable results and project outcomes")
        
        if inflation_analysis.get("severity_level") in ["medium", "high"]:
            recommendations.append("Reduce superlative language and focus on factual achievements")
            recommendations.append("Replace subjective claims with objective, measurable results")
        
        if not recommendations:
            recommendations.append("Resume demonstrates good trustworthiness - maintain factual, evidence-based content")
        
        return recommendations
    
    def _get_default_trust_score(self) -> Dict[str, Any]:
        """Return default trust score when analysis fails"""
        return {
            "trust_score": 50.0,
            "max_score": 100,
            "analysis_details": {},
            "flags": [],
            "recommendations": ["Could not perform trustworthiness analysis - ensure resume content is accessible"]
        }