import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

from resume_intelligence.link_crawler import UniversalLinkCrawler
from resume_intelligence.platform_extractors import BaseExtractor
from resume_intelligence.platform_extractors.github_extractor import GitHubExtractor
from resume_intelligence.platform_extractors.leetcode_extractor import LeetCodeExtractor
from resume_intelligence.platform_extractors.linkedin_extractor import LinkedInExtractor
from resume_intelligence.platform_extractors.schema_normalizer import MetadataNormalizer, SchemaValidator

class LinkExtractionSystem:
    """
    Multi-Platform Link Crawling and Data Extraction System.
    
    Integrates the Universal Link Crawler, Platform-Specific Data Extractors,
    and Metadata Normalizer to provide a comprehensive solution for extracting
    and analyzing external links from resumes.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize the link crawler
        self.crawler = UniversalLinkCrawler(self.config.get("crawler_config", {}))
        
        # Initialize the metadata normalizer
        self.normalizer = MetadataNormalizer(self.config.get("normalizer_config", {}))
        
        # Initialize the schema validator
        self.validator = SchemaValidator()
        
        # Initialize platform-specific extractors
        self.extractors = self._initialize_extractors()
        
        # Configure maximum concurrent extractions
        self.max_concurrent_extractions = self.config.get("max_concurrent_extractions", 5)
        
        # Configure caching
        self.enable_caching = self.config.get("enable_caching", True)
        self.cache_dir = self.config.get("cache_dir", "./cache")
        if self.enable_caching and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _initialize_extractors(self) -> Dict[str, BaseExtractor]:
        """Initialize platform-specific extractors"""
        extractors = {}
        
        # GitHub extractor
        github_config = self.config.get("github_extractor_config", {})
        extractors["github"] = GitHubExtractor(github_config)
        
        # LeetCode extractor
        leetcode_config = self.config.get("leetcode_extractor_config", {})
        extractors["leetcode"] = LeetCodeExtractor(leetcode_config)
        
        # LinkedIn extractor
        linkedin_config = self.config.get("linkedin_extractor_config", {})
        extractors["linkedin"] = LinkedInExtractor(linkedin_config)
        
        # Additional extractors can be added here
        # For example:
        # extractors["stackoverflow"] = StackOverflowExtractor(self.config.get("stackoverflow_extractor_config", {}))
        
        return extractors
    
    def process_resume(self, resume_text: str, contact_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a resume to extract, classify, and analyze external links.
        
        Args:
            resume_text: The text content of the resume
            contact_info: Optional dictionary containing contact information
                         (email, phone, etc.) extracted from the resume
            
        Returns:
            Dictionary containing processed link data with the following structure:
            {
                "links": List of all extracted links,
                "platforms": Dictionary of platform-specific normalized data,
                "summary": Summary statistics and insights
            }
        """
        self.logger.info("Starting resume link extraction and analysis process")
        
        # Step 1: Extract and classify links from the resume
        sections = {}  # We don't have sections here, but the discover_links method requires it
        extracted_links = self.crawler.discover_links(resume_text, sections)
        self.logger.info(f"Extracted {extracted_links['total_links_discovered']} links from resume")
        
        # Step 2: Process each link with the appropriate extractor
        processed_data = self._process_links(extracted_links)
        
        # Step 3: Generate summary statistics and insights
        summary = self._generate_summary(processed_data)
        
        # Step 4: Prepare the final result
        result = {
            "links": extracted_links,
            "platforms": processed_data,
            "summary": summary
        }
        
        self.logger.info("Completed resume link extraction and analysis process")
        return result
    
    def _process_links(self, extracted_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Process extracted links with appropriate platform extractors"""
        processed_data = {}
        extraction_tasks = []
        
        # Get the link profiles from the extracted data
        link_profiles = extracted_data.get("link_profiles", {})
        
        # Prepare extraction tasks for each platform
        for platform, links in link_profiles.items():
            for link in links:
                url = link.get("url")
                
                if not url:
                    continue
                    
                # Check if we have an extractor for this platform
                if platform.lower() in self.extractors:
                    # Check cache first if enabled
                    cached_data = self._get_from_cache(url) if self.enable_caching else None
                    
                    if cached_data:
                        # Use cached data if available
                        if platform not in processed_data:
                            processed_data[platform] = []
                        processed_data[platform].append(cached_data)
                    else:
                        # Add to extraction tasks if not cached
                        extraction_tasks.append((platform, url, link))
                else:
                    # For unsupported platforms, store basic link info
                    if platform not in processed_data:
                        processed_data[platform] = []
                    processed_data[platform].append({
                        "platform": platform,
                        "link_type": link.get("link_type", "unknown"),
                        "url": url,
                        "extraction_successful": False,
                        "error": "Unsupported platform"
                    })
        
        # Process extraction tasks concurrently
        if extraction_tasks:
            with ThreadPoolExecutor(max_workers=self.max_concurrent_extractions) as executor:
                future_to_task = {executor.submit(self._extract_data, platform, url, link): (platform, url) 
                                 for platform, url, link in extraction_tasks}
                
                for future in as_completed(future_to_task):
                    platform, url = future_to_task[future]
                    try:
                        result = future.result()
                        if result:
                            if platform not in processed_data:
                                processed_data[platform] = []
                            processed_data[platform].append(result)
                            
                            # Cache the result if enabled
                            if self.enable_caching:
                                self._save_to_cache(url, result)
                    except Exception as e:
                        self.logger.error(f"Error processing {url}: {str(e)}")
        
        return processed_data
    
    def _extract_data(self, platform: str, url: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract data from a single link using the appropriate extractor"""
        try:
            extractor = self.extractors.get(platform.lower())
            if not extractor:
                return None
                
            # Extract platform-specific data
            raw_data = extractor.extract_data(url, metadata)
            
            # Normalize the data to a common schema
            normalized_data = self.normalizer.normalize(raw_data)
            
            # Validate the normalized data
            if not self.validator.validate(normalized_data):
                self.logger.warning(f"Normalized data for {url} failed validation")
                
            return normalized_data
        except Exception as e:
            self.logger.error(f"Error extracting data from {url}: {str(e)}")
            return None
    
    def _generate_summary(self, processed_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate summary statistics and insights from processed data"""
        summary = {
            "total_platforms": len(processed_data),
            "total_links": sum(len(links) for links in processed_data.values()),
            "platforms_found": list(processed_data.keys()),
            "extraction_success_rate": 0,
            "platform_counts": {},
            "skill_indicators": {},
            "credibility_score": 0
        }
        
        # Calculate platform counts
        for platform, links in processed_data.items():
            summary["platform_counts"][platform] = len(links)
        
        # Calculate extraction success rate
        total_links = summary["total_links"]
        successful_extractions = sum(
            sum(1 for link in links if link.get("extraction_successful", False))
            for links in processed_data.values()
        )
        
        if total_links > 0:
            summary["extraction_success_rate"] = round(successful_extractions / total_links, 2)
        
        # Extract skill indicators from GitHub
        if "github" in processed_data:
            github_profiles = [link for link in processed_data["github"] 
                              if link.get("link_type") == "profile" and link.get("extraction_successful", False)]
            
            if github_profiles:
                # Use the first GitHub profile for skill indicators
                profile = github_profiles[0]
                summary["skill_indicators"]["github"] = {
                    "languages": profile.get("skill_indicators", {}).get("languages", {}),
                    "top_language": profile.get("skill_indicators", {}).get("top_language"),
                    "repositories": profile.get("engagement_metrics", {}).get("total_repositories", 0)
                }
        
        # Extract skill indicators from LeetCode
        if "leetcode" in processed_data:
            leetcode_profiles = [link for link in processed_data["leetcode"] 
                               if link.get("link_type") == "profile" and link.get("extraction_successful", False)]
            
            if leetcode_profiles:
                # Use the first LeetCode profile for skill indicators
                profile = leetcode_profiles[0]
                summary["skill_indicators"]["leetcode"] = {
                    "problems_solved": profile.get("skill_indicators", {}).get("total_solved", 0),
                    "contest_rating": profile.get("skill_indicators", {}).get("contest_rating", 0),
                    "solved_by_difficulty": profile.get("skill_indicators", {}).get("solved_by_difficulty", {})
                }
        
        # Extract skill indicators from LinkedIn
        if "linkedin" in processed_data:
            linkedin_profiles = [link for link in processed_data["linkedin"] 
                               if link.get("link_type") == "profile" and link.get("extraction_successful", False)]
            
            if linkedin_profiles:
                # Use the first LinkedIn profile for skill indicators
                profile = linkedin_profiles[0]
                summary["skill_indicators"]["linkedin"] = {
                    "skills": profile.get("skill_indicators", {}).get("skills", []),
                    "experience_count": profile.get("engagement_metrics", {}).get("experience_count", 0),
                    "total_experience_months": profile.get("skill_indicators", {}).get("total_experience_months", 0)
                }
        
        # Calculate a simple credibility score based on platform presence and data quality
        credibility_score = 0
        
        # Base score from number of platforms
        platform_weight = 10
        credibility_score += min(len(processed_data), 5) * platform_weight
        
        # Additional score from GitHub metrics if available
        if "github" in summary["skill_indicators"]:
            github_metrics = summary["skill_indicators"]["github"]
            repos = github_metrics.get("repositories", 0)
            credibility_score += min(repos, 10) * 2  # Up to 20 points for repositories
        
        # Additional score from LeetCode metrics if available
        if "leetcode" in summary["skill_indicators"]:
            leetcode_metrics = summary["skill_indicators"]["leetcode"]
            problems = leetcode_metrics.get("problems_solved", 0)
            credibility_score += min(problems // 50, 5) * 5  # Up to 25 points for problems
        
        # Additional score from LinkedIn metrics if available
        if "linkedin" in summary["skill_indicators"]:
            linkedin_metrics = summary["skill_indicators"]["linkedin"]
            experience = linkedin_metrics.get("experience_count", 0)
            credibility_score += min(experience, 5) * 5  # Up to 25 points for experience
        
        # Normalize to 0-100 scale
        summary["credibility_score"] = min(credibility_score, 100)
        
        return summary
    
    def _get_cache_key(self, url: str) -> str:
        """Generate a cache key from a URL"""
        parsed_url = urlparse(url)
        return f"{parsed_url.netloc}_{parsed_url.path.replace('/', '_')}"
    
    def _get_from_cache(self, url: str) -> Optional[Dict[str, Any]]:
        """Retrieve data from cache if available and not expired"""
        cache_key = self._get_cache_key(url)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if not os.path.exists(cache_file):
            return None
            
        # Check if cache is expired (default: 7 days)
        cache_ttl = self.config.get("cache_ttl_days", 7) * 86400  # Convert days to seconds
        import time as time_module
        if os.path.getmtime(cache_file) + cache_ttl < time_module.time():
            return None
            
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error reading cache file {cache_file}: {str(e)}")
            return None
    
    def _save_to_cache(self, url: str, data: Dict[str, Any]) -> bool:
        """Save data to cache"""
        if not self.enable_caching:
            return False
            
        cache_key = self._get_cache_key(url)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            return True
        except Exception as e:
            self.logger.error(f"Error writing to cache file {cache_file}: {str(e)}")
            return False


def integrate_with_resume_intelligence(resume_text: str, contact_info: Dict[str, Any] = None, 
                                      config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Integration function to use the Link Extraction System with the Resume Intelligence System.
    
    Args:
        resume_text: The text content of the resume
        contact_info: Optional dictionary containing contact information
        config: Optional configuration dictionary
        
    Returns:
        Dictionary containing link extraction results for integration with other components
    """
    # Initialize the Link Extraction System
    extraction_system = LinkExtractionSystem(config)
    
    # Process the resume
    result = extraction_system.process_resume(resume_text, contact_info)
    
    # Return the result for integration with other components
    return result