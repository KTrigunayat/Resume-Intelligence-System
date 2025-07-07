import re
import json
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup

from resume_intelligence.platform_extractors import BaseExtractor

class LeetCodeExtractor(BaseExtractor):
    """
    LeetCode-specific data extractor.
    Extracts problem-solving statistics, contest ratings, and badges.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.api_base = "https://leetcode.com/graphql"
        self.web_base = "https://leetcode.com"
        
    def extract_data(self, url: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract comprehensive data from a LeetCode profile
        
        Args:
            url: LeetCode profile URL
            metadata: Optional metadata about the URL from the link crawler
            
        Returns:
            Dictionary containing extracted LeetCode data
        """
        # Extract username from URL or metadata
        username = metadata.get("username") if metadata else None
        if not username:
            username = self._extract_username_from_url(url)
            
        if not username:
            self.logger.error(f"Could not extract LeetCode username from {url}")
            return self._empty_result(url)
        
        # Get profile data using GraphQL API
        profile_data = self._extract_profile_data(username)
        
        # Get submission stats
        submission_stats = self._extract_submission_stats(username)
        
        # Get contest data
        contest_data = self._extract_contest_data(username)
        
        # Combine all data
        result = {
            "platform": "leetcode",
            "link_type": "profile",
            "url": url,
            "username": username,
            "profile_data": profile_data,
            "submission_stats": submission_stats,
            "contest_data": contest_data
        }
        
        return result
    
    def _extract_username_from_url(self, url: str) -> Optional[str]:
        """Extract LeetCode username from URL"""
        # Profile pattern: leetcode.com/username
        profile_match = re.search(r'leetcode\.com/([\w-]+)(?:/|$)', url)
        if profile_match:
            return profile_match.group(1)
        return None
    
    def _extract_profile_data(self, username: str) -> Dict[str, Any]:
        """Extract profile data using GraphQL API"""
        query = """
        query getUserProfile($username: String!) {
          matchedUser(username: $username) {
            username
            profile {
              realName
              aboutMe
              userAvatar
              location
              ranking
              reputation
              starRating
              skillTags
            }
            badges {
              id
              displayName
              icon
              creationDate
            }
            socialAccounts {
              provider
              profileUrl
            }
          }
        }
        """
        
        variables = {"username": username}
        
        response = self._make_request(
            self.api_base,
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"query": query, "variables": variables})
        )
        
        if not response or not response.get("data") or not response.get("data").get("matchedUser"):
            self.logger.warning(f"Could not fetch profile data for LeetCode user {username}")
            return {}
        
        user_data = response["data"]["matchedUser"]
        profile = user_data.get("profile", {})
        badges = user_data.get("badges", [])
        social_accounts = user_data.get("socialAccounts", [])
        
        return {
            "username": user_data.get("username"),
            "real_name": profile.get("realName"),
            "about_me": profile.get("aboutMe"),
            "location": profile.get("location"),
            "ranking": profile.get("ranking"),
            "reputation": profile.get("reputation"),
            "star_rating": profile.get("starRating"),
            "skill_tags": profile.get("skillTags", []),
            "badges": [{
                "id": badge.get("id"),
                "name": badge.get("displayName"),
                "creation_date": badge.get("creationDate")
            } for badge in badges],
            "social_accounts": [{
                "provider": account.get("provider"),
                "url": account.get("profileUrl")
            } for account in social_accounts]
        }
    
    def _extract_submission_stats(self, username: str) -> Dict[str, Any]:
        """Extract submission statistics using GraphQL API"""
        query = """
        query getUserProfileStats($username: String!) {
          matchedUser(username: $username) {
            submitStats {
              acSubmissionNum {
                difficulty
                count
                submissions
              }
              totalSubmissionNum {
                difficulty
                count
                submissions
              }
            }
            problemsSolvedBeatsStats {
              difficulty
              percentage
            }
          }
        }
        """
        
        variables = {"username": username}
        
        response = self._make_request(
            self.api_base,
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"query": query, "variables": variables})
        )
        
        if not response or not response.get("data") or not response.get("data").get("matchedUser"):
            self.logger.warning(f"Could not fetch submission stats for LeetCode user {username}")
            return {}
        
        user_data = response["data"]["matchedUser"]
        submit_stats = user_data.get("submitStats", {})
        beats_stats = user_data.get("problemsSolvedBeatsStats", [])
        
        # Process AC submission numbers by difficulty
        ac_submissions = {}
        for item in submit_stats.get("acSubmissionNum", []):
            difficulty = item.get("difficulty")
            if difficulty:
                ac_submissions[difficulty.lower()] = {
                    "count": item.get("count", 0),
                    "submissions": item.get("submissions", 0)
                }
        
        # Process total submission numbers by difficulty
        total_submissions = {}
        for item in submit_stats.get("totalSubmissionNum", []):
            difficulty = item.get("difficulty")
            if difficulty:
                total_submissions[difficulty.lower()] = {
                    "count": item.get("count", 0),
                    "submissions": item.get("submissions", 0)
                }
        
        # Process beats stats
        beats_percentages = {}
        for item in beats_stats:
            difficulty = item.get("difficulty")
            if difficulty:
                beats_percentages[difficulty.lower()] = item.get("percentage", 0)
        
        # Calculate acceptance rate
        total_ac = sum(item.get("count", 0) for item in submit_stats.get("acSubmissionNum", []))
        total_submitted = sum(item.get("count", 0) for item in submit_stats.get("totalSubmissionNum", []))
        acceptance_rate = round((total_ac / total_submitted) * 100, 2) if total_submitted > 0 else 0
        
        return {
            "solved_problems": {
                "total": total_ac,
                "by_difficulty": ac_submissions
            },
            "total_submissions": {
                "total": total_submitted,
                "by_difficulty": total_submissions
            },
            "acceptance_rate": acceptance_rate,
            "beats_percentages": beats_percentages
        }
    
    def _extract_contest_data(self, username: str) -> Dict[str, Any]:
        """Extract contest data using GraphQL API"""
        query = """
        query getUserContestInfo($username: String!) {
          userContestRanking(username: $username) {
            attendedContestsCount
            rating
            globalRanking
            totalParticipants
            topPercentage
            badge {
              name
            }
          }
          userContestRankingHistory(username: $username) {
            attended
            trendDirection
            problemsSolved
            totalProblems
            finishTimeInSeconds
            rating
            ranking
            contest {
              title
              startTime
            }
          }
        }
        """
        
        variables = {"username": username}
        
        response = self._make_request(
            self.api_base,
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"query": query, "variables": variables})
        )
        
        if not response or not response.get("data"):
            self.logger.warning(f"Could not fetch contest data for LeetCode user {username}")
            return {}
        
        ranking = response["data"].get("userContestRanking", {})
        history = response["data"].get("userContestRankingHistory", [])
        
        # Process contest history
        contest_history = []
        for contest in history:
            if contest.get("attended"):
                contest_info = {
                    "title": contest.get("contest", {}).get("title"),
                    "start_time": contest.get("contest", {}).get("startTime"),
                    "problems_solved": contest.get("problemsSolved", 0),
                    "total_problems": contest.get("totalProblems", 0),
                    "finish_time_seconds": contest.get("finishTimeInSeconds", 0),
                    "rating": contest.get("rating", 0),
                    "ranking": contest.get("ranking", 0),
                    "trend": contest.get("trendDirection")
                }
                contest_history.append(contest_info)
        
        return {
            "rating": ranking.get("rating", 0),
            "global_ranking": ranking.get("globalRanking", 0),
            "total_participants": ranking.get("totalParticipants", 0),
            "top_percentage": ranking.get("topPercentage", 0),
            "badge": ranking.get("badge", {}).get("name"),
            "contests_attended": ranking.get("attendedContestsCount", 0),
            "contest_history": contest_history
        }
    
    def _empty_result(self, url: str) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            "platform": "leetcode",
            "link_type": "unknown",
            "url": url,
            "username": None,
            "extraction_successful": False,
            "error": "Could not extract data"
        }