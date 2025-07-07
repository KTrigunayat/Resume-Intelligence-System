import re
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from collections import Counter

from resume_intelligence.platform_extractors import BaseExtractor

class GitHubExtractor(BaseExtractor):
    """
    GitHub-specific data extractor.
    Extracts repository data, star counts, language distribution, and contribution metrics.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.api_base = "https://api.github.com"
        self.web_base = "https://github.com"
        
    def extract_data(self, url: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract comprehensive data from a GitHub profile or repository
        
        Args:
            url: GitHub URL (profile or repository)
            metadata: Optional metadata about the URL from the link crawler
            
        Returns:
            Dictionary containing extracted GitHub data
        """
        # Extract username from URL or metadata
        username = metadata.get("username") if metadata else None
        if not username:
            username = self._extract_username_from_url(url)
            
        if not username:
            self.logger.error(f"Could not extract GitHub username from {url}")
            return self._empty_result(url)
            
        # Determine if this is a profile or repository URL
        is_repo = self._is_repository_url(url)
        
        if is_repo:
            return self._extract_repository_data(url, username)
        else:
            return self._extract_profile_data(username)
    
    def _extract_username_from_url(self, url: str) -> Optional[str]:
        """Extract GitHub username from URL"""
        # Profile pattern: github.com/username
        profile_match = re.search(r'github\.com/([\w-]+)(?:/|$)', url)
        if profile_match:
            return profile_match.group(1)
        return None
    
    def _is_repository_url(self, url: str) -> bool:
        """Determine if URL is a repository URL"""
        # Repository pattern: github.com/username/repository
        repo_match = re.search(r'github\.com/[\w-]+/[\w-]+(?:/|$)', url)
        return bool(repo_match)
    
    def _extract_repository_data(self, url: str, username: str) -> Dict[str, Any]:
        """Extract data for a specific repository"""
        # Extract repository name from URL
        repo_match = re.search(r'github\.com/[\w-]+/([\w-]+)(?:/|$)', url)
        if not repo_match:
            self.logger.error(f"Could not extract repository name from {url}")
            return self._empty_result(url)
            
        repo_name = repo_match.group(1)
        
        # Get repository data from GitHub API
        api_url = f"{self.api_base}/repos/{username}/{repo_name}"
        repo_data = self._extract_from_api(api_url)
        
        if not repo_data:
            self.logger.warning(f"Could not fetch repository data for {username}/{repo_name}")
            return self._empty_result(url)
        
        # Get repository languages
        languages_url = f"{self.api_base}/repos/{username}/{repo_name}/languages"
        languages_data = self._extract_from_api(languages_url) or {}
        
        # Get commit activity
        commits_url = f"{self.api_base}/repos/{username}/{repo_name}/stats/commit_activity"
        commits_data = self._extract_from_api(commits_url) or []
        
        # Calculate recent commit count (last 4 weeks)
        recent_commits = sum(week.get("total", 0) for week in commits_data[-4:] if isinstance(week, dict))
        
        # Extract contributors
        contributors_url = f"{self.api_base}/repos/{username}/{repo_name}/contributors"
        contributors_data = self._extract_from_api(contributors_url) or []
        contributors_count = len(contributors_data)
        
        # Calculate language percentages
        total_bytes = sum(languages_data.values()) if languages_data else 0
        language_percentages = {}
        if total_bytes > 0:
            language_percentages = {lang: round((bytes / total_bytes) * 100, 2) 
                                  for lang, bytes in languages_data.items()}
        
        # Build result
        return {
            "platform": "github",
            "link_type": "repository",
            "url": url,
            "username": username,
            "repository_name": repo_name,
            "repository_data": {
                "name": repo_data.get("name"),
                "full_name": repo_data.get("full_name"),
                "description": repo_data.get("description"),
                "stars": repo_data.get("stargazers_count", 0),
                "forks": repo_data.get("forks_count", 0),
                "watchers": repo_data.get("watchers_count", 0),
                "open_issues": repo_data.get("open_issues_count", 0),
                "created_at": repo_data.get("created_at"),
                "updated_at": repo_data.get("updated_at"),
                "pushed_at": repo_data.get("pushed_at"),
                "language": repo_data.get("language"),
                "languages": language_percentages,
                "recent_commits": recent_commits,
                "contributors_count": contributors_count,
                "is_fork": repo_data.get("fork", False),
                "topics": repo_data.get("topics", [])
            }
        }
    
    def _extract_profile_data(self, username: str) -> Dict[str, Any]:
        """Extract data for a GitHub user profile"""
        # Get user data from GitHub API
        api_url = f"{self.api_base}/users/{username}"
        user_data = self._extract_from_api(api_url)
        
        if not user_data:
            self.logger.warning(f"Could not fetch user data for {username}")
            return self._empty_result(f"{self.web_base}/{username}")
        
        # Get repositories
        repos_url = f"{self.api_base}/users/{username}/repos"
        repos_data = self._extract_from_api(repos_url) or []
        
        # Get user events (activity)
        events_url = f"{self.api_base}/users/{username}/events"
        events_data = self._extract_from_api(events_url) or []
        
        # Calculate repository statistics
        total_stars = sum(repo.get("stargazers_count", 0) for repo in repos_data)
        total_forks = sum(repo.get("forks_count", 0) for repo in repos_data)
        total_watchers = sum(repo.get("watchers_count", 0) for repo in repos_data)
        
        # Count repositories by language
        languages = [repo.get("language") for repo in repos_data if repo.get("language")]
        language_counts = Counter(languages)
        
        # Calculate recent activity (last 30 days)
        now = datetime.now()
        cutoff_date = now - timedelta(days=30)
        
        recent_events = []
        for event in events_data:
            if "created_at" in event:
                try:
                    event_date = datetime.strptime(event["created_at"], "%Y-%m-%dT%H:%M:%SZ")
                    if event_date > cutoff_date:
                        recent_events.append(event)
                except (ValueError, TypeError):
                    pass
        
        # Count event types
        event_types = [event.get("type") for event in recent_events]
        event_type_counts = Counter(event_types)
        
        # Build result
        return {
            "platform": "github",
            "link_type": "profile",
            "url": f"{self.web_base}/{username}",
            "username": username,
            "profile_data": {
                "name": user_data.get("name"),
                "login": user_data.get("login"),
                "bio": user_data.get("bio"),
                "company": user_data.get("company"),
                "blog": user_data.get("blog"),
                "location": user_data.get("location"),
                "email": user_data.get("email"),
                "public_repos": user_data.get("public_repos", 0),
                "public_gists": user_data.get("public_gists", 0),
                "followers": user_data.get("followers", 0),
                "following": user_data.get("following", 0),
                "created_at": user_data.get("created_at"),
                "updated_at": user_data.get("updated_at"),
                "account_age_days": self._calculate_account_age(user_data.get("created_at"))
            },
            "repository_stats": {
                "total_repositories": len(repos_data),
                "total_stars": total_stars,
                "total_forks": total_forks,
                "total_watchers": total_watchers,
                "languages": dict(language_counts),
                "top_language": language_counts.most_common(1)[0][0] if language_counts else None,
                "forked_repos": sum(1 for repo in repos_data if repo.get("fork", False))
            },
            "activity_stats": {
                "recent_events": len(recent_events),
                "event_types": dict(event_type_counts),
                "most_active_event": event_type_counts.most_common(1)[0][0] if event_type_counts else None
            }
        }
    
    def _calculate_account_age(self, created_at: str) -> Optional[int]:
        """Calculate account age in days"""
        if not created_at:
            return None
            
        try:
            creation_date = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
            return (datetime.now() - creation_date).days
        except (ValueError, TypeError):
            return None
    
    def _empty_result(self, url: str) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            "platform": "github",
            "link_type": "unknown",
            "url": url,
            "username": None,
            "extraction_successful": False,
            "error": "Could not extract data"
        }