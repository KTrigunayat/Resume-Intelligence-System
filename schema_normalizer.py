import json
from typing import Dict, List, Any, Optional
from datetime import datetime

class MetadataNormalizer:
    """
    Normalizes and standardizes metadata from different platforms into a common schema.
    Provides a unified interface for downstream components to access platform data.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.schema_version = "1.0.0"
        
    def normalize(self, platform_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize platform-specific data into a common schema
        
        Args:
            platform_data: Raw data extracted from a specific platform
            
        Returns:
            Normalized data following the common schema
        """
        platform = platform_data.get("platform", "unknown").lower()
        
        # Route to appropriate normalizer based on platform
        if platform == "github":
            return self._normalize_github(platform_data)
        elif platform == "leetcode":
            return self._normalize_leetcode(platform_data)
        elif platform == "linkedin":
            return self._normalize_linkedin(platform_data)
        elif platform == "stackoverflow":
            return self._normalize_stackoverflow(platform_data)
        elif platform == "medium":
            return self._normalize_medium(platform_data)
        elif platform == "kaggle":
            return self._normalize_kaggle(platform_data)
        elif platform == "behance":
            return self._normalize_behance(platform_data)
        elif platform == "dribbble":
            return self._normalize_dribbble(platform_data)
        elif platform == "figma":
            return self._normalize_figma(platform_data)
        else:
            return self._normalize_generic(platform_data)
    
    def _normalize_github(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize GitHub data"""
        link_type = data.get("link_type", "unknown")
        
        if link_type == "profile":
            profile_data = data.get("profile_data", {})
            repo_stats = data.get("repository_stats", {})
            activity_stats = data.get("activity_stats", {})
            
            return {
                "schema_version": self.schema_version,
                "platform": "github",
                "link_type": "profile",
                "url": data.get("url"),
                "username": data.get("username"),
                "extraction_date": self._get_current_timestamp(),
                "extraction_successful": data.get("extraction_successful", True),
                "basic_info": {
                    "name": profile_data.get("name"),
                    "bio": profile_data.get("bio"),
                    "location": profile_data.get("location"),
                    "company": profile_data.get("company"),
                    "website": profile_data.get("blog"),
                    "email": profile_data.get("email"),
                    "account_created": profile_data.get("created_at"),
                    "account_age_days": profile_data.get("account_age_days")
                },
                "engagement_metrics": {
                    "followers": profile_data.get("followers", 0),
                    "following": profile_data.get("following", 0),
                    "total_repositories": repo_stats.get("total_repositories", 0),
                    "total_stars": repo_stats.get("total_stars", 0),
                    "total_forks": repo_stats.get("total_forks", 0),
                    "total_watchers": repo_stats.get("total_watchers", 0),
                    "recent_activity_count": activity_stats.get("recent_events", 0)
                },
                "skill_indicators": {
                    "languages": repo_stats.get("languages", {}),
                    "top_language": repo_stats.get("top_language"),
                    "activity_types": activity_stats.get("event_types", {}),
                    "most_active_contribution_type": activity_stats.get("most_active_event")
                },
                "credibility_signals": {
                    "account_age_days": profile_data.get("account_age_days"),
                    "has_bio": bool(profile_data.get("bio")),
                    "has_company": bool(profile_data.get("company")),
                    "has_location": bool(profile_data.get("location")),
                    "has_website": bool(profile_data.get("blog")),
                    "has_email": bool(profile_data.get("email")),
                    "original_repos_ratio": self._calculate_original_repos_ratio(repo_stats)
                }
            }
        elif link_type == "repository":
            repo_data = data.get("repository_data", {})
            
            return {
                "schema_version": self.schema_version,
                "platform": "github",
                "link_type": "repository",
                "url": data.get("url"),
                "username": data.get("username"),
                "repository_name": data.get("repository_name"),
                "extraction_date": self._get_current_timestamp(),
                "extraction_successful": data.get("extraction_successful", True),
                "basic_info": {
                    "name": repo_data.get("name"),
                    "full_name": repo_data.get("full_name"),
                    "description": repo_data.get("description"),
                    "created_at": repo_data.get("created_at"),
                    "updated_at": repo_data.get("updated_at"),
                    "pushed_at": repo_data.get("pushed_at"),
                    "is_fork": repo_data.get("is_fork", False)
                },
                "engagement_metrics": {
                    "stars": repo_data.get("stars", 0),
                    "forks": repo_data.get("forks", 0),
                    "watchers": repo_data.get("watchers", 0),
                    "open_issues": repo_data.get("open_issues", 0),
                    "contributors_count": repo_data.get("contributors_count", 0),
                    "recent_commits": repo_data.get("recent_commits", 0)
                },
                "skill_indicators": {
                    "primary_language": repo_data.get("language"),
                    "languages": repo_data.get("languages", {}),
                    "topics": repo_data.get("topics", [])
                },
                "credibility_signals": {
                    "has_description": bool(repo_data.get("description")),
                    "has_topics": bool(repo_data.get("topics")),
                    "is_original": not repo_data.get("is_fork", False),
                    "recent_activity": bool(repo_data.get("recent_commits", 0) > 0),
                    "multiple_contributors": repo_data.get("contributors_count", 0) > 1
                }
            }
        else:
            return self._normalize_generic(data)
    
    def _normalize_leetcode(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize LeetCode data"""
        profile_data = data.get("profile_data", {})
        submission_stats = data.get("submission_stats", {})
        contest_data = data.get("contest_data", {})
        
        # Extract solved problems by difficulty
        solved_by_difficulty = {}
        for difficulty, stats in submission_stats.get("solved_problems", {}).get("by_difficulty", {}).items():
            solved_by_difficulty[difficulty] = stats.get("count", 0)
        
        return {
            "schema_version": self.schema_version,
            "platform": "leetcode",
            "link_type": "profile",
            "url": data.get("url"),
            "username": data.get("username"),
            "extraction_date": self._get_current_timestamp(),
            "extraction_successful": data.get("extraction_successful", True),
            "basic_info": {
                "name": profile_data.get("real_name"),
                "about": profile_data.get("about_me"),
                "location": profile_data.get("location"),
                "skill_tags": profile_data.get("skill_tags", [])
            },
            "engagement_metrics": {
                "ranking": profile_data.get("ranking"),
                "reputation": profile_data.get("reputation"),
                "star_rating": profile_data.get("star_rating"),
                "badges_count": len(profile_data.get("badges", [])),
                "contests_attended": contest_data.get("contests_attended", 0)
            },
            "skill_indicators": {
                "total_solved": submission_stats.get("solved_problems", {}).get("total", 0),
                "solved_by_difficulty": solved_by_difficulty,
                "acceptance_rate": submission_stats.get("acceptance_rate", 0),
                "contest_rating": contest_data.get("rating", 0),
                "contest_global_ranking": contest_data.get("global_ranking"),
                "contest_ranking_percentile": 100 - contest_data.get("top_percentage", 100)
            },
            "credibility_signals": {
                "has_profile_info": bool(profile_data.get("about_me")),
                "has_contests": contest_data.get("contests_attended", 0) > 0,
                "has_badges": len(profile_data.get("badges", [])) > 0,
                "solved_hard_problems": solved_by_difficulty.get("hard", 0) > 0,
                "contest_percentile": 100 - contest_data.get("top_percentage", 100)
            }
        }
    
    def _normalize_linkedin(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize LinkedIn data"""
        profile_data = data.get("profile_data", {})
        experience = data.get("experience", [])
        education = data.get("education", [])
        skills = data.get("skills", [])
        
        # Calculate experience duration (approximate)
        total_experience_months = 0
        for exp in experience:
            date_range = exp.get("date_range", "")
            # Simple heuristic: extract years from strings like "2018 - 2020" or "2018 - Present"
            years = [int(y) for y in re.findall(r'\b(19\d{2}|20\d{2})\b', date_range)]
            if len(years) >= 2:
                total_experience_months += (years[1] - years[0]) * 12
            elif len(years) == 1 and "present" in date_range.lower():
                current_year = datetime.now().year
                total_experience_months += (current_year - years[0]) * 12
        
        return {
            "schema_version": self.schema_version,
            "platform": "linkedin",
            "link_type": "profile",
            "url": data.get("url"),
            "username": data.get("username"),
            "extraction_date": self._get_current_timestamp(),
            "extraction_successful": data.get("extraction_successful", True),
            "basic_info": {
                "name": profile_data.get("name"),
                "headline": profile_data.get("headline"),
                "location": profile_data.get("location"),
                "about": profile_data.get("about")
            },
            "engagement_metrics": {
                "connections": self._parse_connections(profile_data.get("connections", "0")),
                "experience_count": len(experience),
                "education_count": len(education),
                "skills_count": len(skills)
            },
            "skill_indicators": {
                "skills": skills,
                "experience": [{
                    "title": exp.get("title"),
                    "company": exp.get("company"),
                    "date_range": exp.get("date_range"),
                    "location": exp.get("location")
                } for exp in experience],
                "education": [{
                    "school": edu.get("school"),
                    "degree": edu.get("degree"),
                    "field": edu.get("field_of_study"),
                    "date_range": edu.get("date_range")
                } for edu in education],
                "total_experience_months": total_experience_months
            },
            "credibility_signals": {
                "has_photo": True,  # Placeholder, can't reliably detect from scraping
                "has_about": bool(profile_data.get("about")),
                "has_experience": len(experience) > 0,
                "has_education": len(education) > 0,
                "has_skills": len(skills) > 0,
                "connection_level": "500+" in str(profile_data.get("connections", ""))
            }
        }
    
    def _normalize_stackoverflow(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Stack Overflow data"""
        # Placeholder for Stack Overflow normalizer
        # Would follow similar pattern to the other normalizers
        return self._normalize_generic(data)
    
    def _normalize_medium(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Medium data"""
        # Placeholder for Medium normalizer
        return self._normalize_generic(data)
    
    def _normalize_kaggle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Kaggle data"""
        # Placeholder for Kaggle normalizer
        return self._normalize_generic(data)
    
    def _normalize_behance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Behance data"""
        # Placeholder for Behance normalizer
        return self._normalize_generic(data)
    
    def _normalize_dribbble(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Dribbble data"""
        # Placeholder for Dribbble normalizer
        return self._normalize_generic(data)
    
    def _normalize_figma(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Figma data"""
        # Placeholder for Figma normalizer
        return self._normalize_generic(data)
    
    def _normalize_generic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generic normalizer for platforms without specific normalizers"""
        return {
            "schema_version": self.schema_version,
            "platform": data.get("platform", "unknown"),
            "link_type": data.get("link_type", "unknown"),
            "url": data.get("url"),
            "username": data.get("username"),
            "extraction_date": self._get_current_timestamp(),
            "extraction_successful": data.get("extraction_successful", True),
            "raw_data": data
        }
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        return datetime.now().isoformat()
    
    def _calculate_original_repos_ratio(self, repo_stats: Dict[str, Any]) -> float:
        """Calculate ratio of original repositories to total repositories"""
        total_repos = repo_stats.get("total_repositories", 0)
        forked_repos = repo_stats.get("forked_repos", 0)
        
        if total_repos == 0:
            return 0.0
            
        original_repos = total_repos - forked_repos
        return round(original_repos / total_repos, 2)
    
    def _parse_connections(self, connections_str: str) -> int:
        """Parse LinkedIn connections string to integer"""
        if not connections_str:
            return 0
            
        # Handle "500+" format
        if "+" in connections_str:
            return int(connections_str.replace("+", ""))
            
        # Try to extract number
        import re
        match = re.search(r'\d+', connections_str)
        if match:
            return int(match.group())
            
        return 0


class SchemaValidator:
    """
    Validates that normalized data conforms to the expected schema.
    Ensures data consistency for downstream components.
    """
    
    def __init__(self):
        self.schema_version = "1.0.0"
        
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate that data conforms to the expected schema"""
        # Check required top-level fields
        required_fields = [
            "schema_version", "platform", "link_type", "url", 
            "extraction_date", "extraction_successful"
        ]
        
        for field in required_fields:
            if field not in data:
                return False
                
        # Check schema version
        if data["schema_version"] != self.schema_version:
            return False
            
        # Validate based on platform and link_type
        platform = data.get("platform", "").lower()
        link_type = data.get("link_type", "").lower()
        
        if platform == "github":
            if link_type == "profile":
                return self._validate_github_profile(data)
            elif link_type == "repository":
                return self._validate_github_repository(data)
        elif platform == "leetcode":
            return self._validate_leetcode_profile(data)
        elif platform == "linkedin":
            return self._validate_linkedin_profile(data)
            
        # For other platforms or generic data, just check basic structure
        return True
    
    def _validate_github_profile(self, data: Dict[str, Any]) -> bool:
        """Validate GitHub profile schema"""
        required_sections = ["basic_info", "engagement_metrics", "skill_indicators", "credibility_signals"]
        
        for section in required_sections:
            if section not in data:
                return False
                
        return True
    
    def _validate_github_repository(self, data: Dict[str, Any]) -> bool:
        """Validate GitHub repository schema"""
        required_sections = ["basic_info", "engagement_metrics", "skill_indicators", "credibility_signals"]
        
        for section in required_sections:
            if section not in data:
                return False
                
        # Check repository-specific fields
        if "repository_name" not in data:
            return False
            
        return True
    
    def _validate_leetcode_profile(self, data: Dict[str, Any]) -> bool:
        """Validate LeetCode profile schema"""
        required_sections = ["basic_info", "engagement_metrics", "skill_indicators", "credibility_signals"]
        
        for section in required_sections:
            if section not in data:
                return False
                
        return True
    
    def _validate_linkedin_profile(self, data: Dict[str, Any]) -> bool:
        """Validate LinkedIn profile schema"""
        required_sections = ["basic_info", "engagement_metrics", "skill_indicators", "credibility_signals"]
        
        for section in required_sections:
            if section not in data:
                return False
                
        return True