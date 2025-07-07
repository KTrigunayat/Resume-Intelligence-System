import re
import logging
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
from pathlib import Path
import json
import time

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logging.warning("Playwright not available. Falling back to basic link extraction.")

class UniversalLinkCrawler:
    """
    Universal Link Crawler Engine: Automatically discover and fetch every external link 
    embedded in a résumé to build a comprehensive engagement profile.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        self.platform_patterns = self._compile_platform_patterns()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration for link crawling"""
        return {
            "timeout": 30,  # seconds
            "max_depth": 1,  # How many levels of links to follow
            "rate_limit_delay": 2,  # seconds between requests
            "max_links_per_platform": 5,  # Maximum links to process per platform
            "headless": True,  # Run browser in headless mode
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "platforms": {
                "github": {
                    "patterns": [r"github\.com/([\w-]+)(?:/([\w-]+))?"],
                    "indicators": ["github.com", "repositories", "contribution-graph"]
                },
                "linkedin": {
                    "patterns": [r"linkedin\.com/in/([\w-]+)"],
                    "indicators": ["linkedin.com/in", "experience", "education"]
                },
                "leetcode": {
                    "patterns": [r"leetcode\.com/([\w-]+)"],
                    "indicators": ["leetcode.com", "problems-solved", "ranking"]
                },
                "kaggle": {
                    "patterns": [r"kaggle\.com/([\w-]+)"],
                    "indicators": ["kaggle.com", "notebooks", "competitions"]
                },
                "medium": {
                    "patterns": [r"medium\.com/@?([\w-]+)"],
                    "indicators": ["medium.com", "articles", "claps"]
                },
                "behance": {
                    "patterns": [r"behance\.net/([\w-]+)"],
                    "indicators": ["behance.net", "projects", "appreciations"]
                },
                "dribbble": {
                    "patterns": [r"dribbble\.com/([\w-]+)"],
                    "indicators": ["dribbble.com", "shots", "likes"]
                },
                "figma": {
                    "patterns": [r"figma\.com/@([\w-]+)"],
                    "indicators": ["figma.com", "designs", "community"]
                },
                "stackoverflow": {
                    "patterns": [r"stackoverflow\.com/users/(\d+)/([\w-]+)"],
                    "indicators": ["stackoverflow.com/users", "reputation", "badges"]
                },
                "personal_website": {
                    "patterns": [r"https?://([\w-]+\.[\w-]+)(?:\.[\w-]+)+"],
                    "indicators": ["portfolio", "about", "contact"]
                }
            }
        }
    
    def _compile_platform_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for platform detection"""
        compiled_patterns = {}
        for platform, config in self.config["platforms"].items():
            compiled_patterns[platform] = [re.compile(pattern, re.IGNORECASE) for pattern in config["patterns"]]
        return compiled_patterns
    
    def discover_links(self, resume_text: str, sections: Dict[str, Any]) -> Dict[str, Any]:
        """
        Discover all links in a resume and classify them by platform
        
        Args:
            resume_text: Full text of the resume
            sections: Parsed resume sections
            
        Returns:
            Dictionary containing discovered links and metadata
        """
        self.logger.info(f"Starting link discovery in resume text of length {len(resume_text)}")
        
        # Log the first 100 characters of the resume text for debugging
        self.logger.debug(f"Resume text preview: {resume_text[:100]}...")
        
        # Extract all URLs from text using regex
        basic_links = self._extract_basic_links(resume_text)
        self.logger.info(f"Extracted {len(basic_links)} basic links from resume text")
        
        # Also try to extract links from each section if available
        section_links = []
        if sections and isinstance(sections, dict):
            for section_name, section_content in sections.items():
                if isinstance(section_content, str):
                    section_basic_links = self._extract_basic_links(section_content)
                    section_links.extend(section_basic_links)
                    self.logger.info(f"Extracted {len(section_basic_links)} links from section '{section_name}'")
        
        # Combine all links and remove duplicates
        all_links = list(set(basic_links + section_links))
        self.logger.info(f"Combined total of {len(all_links)} unique links")
        
        # Classify links by platform
        classified_links = self._classify_links(all_links)
        
        # If Playwright is available, use it for enhanced link discovery
        enhanced_links = {}
        if PLAYWRIGHT_AVAILABLE and self.config.get("use_playwright", True):
            try:
                enhanced_links = self._discover_links_with_playwright(all_links)
                # Merge basic and enhanced links
                for platform, links in enhanced_links.items():
                    if platform in classified_links:
                        classified_links[platform].extend(links)
                    else:
                        classified_links[platform] = links
            except Exception as e:
                self.logger.error(f"Error using Playwright for link discovery: {e}")
        
        # Extract basic metadata for each link
        link_profiles = self._extract_basic_metadata(classified_links)
        
        result = {
            "total_links_discovered": sum(len(links) for links in classified_links.values()),
            "platforms_found": list(classified_links.keys()),
            "link_profiles": link_profiles
        }
        
        self.logger.info(f"Link discovery complete. Found {result['total_links_discovered']} links across {len(result['platforms_found'])} platforms")
        return result
    
    def _extract_basic_links(self, text: str) -> List[str]:
        """Extract all URLs from text using regex"""
        # Basic URL pattern - enhanced to catch more variations
        url_pattern = re.compile(
            r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
            r'(?:/(?:[-\w%!$&\'()*+,;=]|(?:%[\da-fA-F]{2}))*)*'
            r'(?:\?(?:[\w%!$&\'()*+,;=:/?@]|(?:%[\da-fA-F]{2}))*)?'
            r'(?:#(?:[\w%!$&\'()*+,;=:/?@]|(?:%[\da-fA-F]{2}))*)?',
            re.IGNORECASE
        )
        
        # Find all URLs in text
        urls = url_pattern.findall(text)
        
        # Also look for www. links that might not have http/https
        www_pattern = re.compile(r'www\.(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[\w\-./?%&=]*)?', re.IGNORECASE)
        www_urls = www_pattern.findall(text)
        
        # Look for common platform domains without http/https or www
        domain_pattern = re.compile(
            r'(?<![\w@-])'
            r'(?:github\.com|linkedin\.com|leetcode\.com|kaggle\.com|medium\.com|behance\.net|dribbble\.com|figma\.com|stackoverflow\.com)'
            r'/[\w\-./]+',
            re.IGNORECASE
        )
        domain_urls = domain_pattern.findall(text)
        
        # Add http:// to www. links
        urls.extend([f"http://{url}" for url in www_urls if not any(url in u for u in urls)])
        
        # Add http:// to domain links
        urls.extend([f"http://{url}" for url in domain_urls if not any(url in u for u in urls)])
        
        # Look for email addresses that might contain links
        email_pattern = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+', re.IGNORECASE)
        emails = email_pattern.findall(text)
        
        # Extract potential domains from emails
        for email in emails:
            domain = email.split('@')[1]
            if domain not in ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']:
                urls.append(f"http://{domain}")
        
        # Clean and normalize URLs
        cleaned_urls = []
        for url in urls:
            # Remove trailing punctuation
            while url and url[-1] in '.,;:"()[]{}':  # Added missing colon
                url = url[:-1]
            if url:
                cleaned_urls.append(url)
        
        return list(set(cleaned_urls))  # Remove duplicates
    
    def _classify_links(self, urls: List[str]) -> Dict[str, List[str]]:
        """Classify URLs by platform"""
        classified = {}
        
        for url in urls:
            platform_found = False
            
            for platform, patterns in self.platform_patterns.items():
                for pattern in patterns:
                    if pattern.search(url):
                        if platform not in classified:
                            classified[platform] = []
                        classified[platform].append(url)
                        platform_found = True
                        break
                        
                if platform_found:
                    break
            
            # If no specific platform matched, classify as "other"
            if not platform_found:
                if "other" not in classified:
                    classified["other"] = []
                classified["other"].append(url)
        
        return classified
    
    def _discover_links_with_playwright(self, seed_urls: List[str]) -> Dict[str, List[str]]:
        """Use Playwright to discover additional links by rendering pages"""
        if not PLAYWRIGHT_AVAILABLE:
            return {}
            
        discovered_links = {}
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.config["headless"])
            context = browser.new_context(
                user_agent=self.config["user_agent"],
                viewport={"width": 1280, "height": 800}
            )
            
            for url in seed_urls:
                try:
                    # Rate limiting
                    time.sleep(self.config["rate_limit_delay"])
                    
                    page = context.new_page()
                    page.goto(url, timeout=self.config["timeout"]*1000, wait_until="networkidle")
                    
                    # Extract all links from the page
                    links = page.eval_on_selector_all("a[href]", """
                        elements => elements.map(el => el.href)
                    """)
                    
                    # Classify the new links
                    new_classified = self._classify_links(links)
                    
                    # Merge with discovered links
                    for platform, platform_links in new_classified.items():
                        if platform not in discovered_links:
                            discovered_links[platform] = []
                        # Add only new links
                        discovered_links[platform].extend([link for link in platform_links if link not in discovered_links[platform]])
                    
                    page.close()
                    
                except Exception as e:
                    self.logger.error(f"Error processing URL {url} with Playwright: {e}")
            
            browser.close()
        
        return discovered_links
    
    def _extract_basic_metadata(self, classified_links: Dict[str, List[str]]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract basic metadata for each link"""
        link_profiles = {}
        
        for platform, urls in classified_links.items():
            # Limit the number of links per platform
            limited_urls = urls[:self.config["max_links_per_platform"]]
            
            link_profiles[platform] = []
            
            for url in limited_urls:
                # Parse URL to extract components
                parsed_url = urlparse(url)
                
                # Basic metadata that doesn't require rendering
                metadata = {
                    "url": url,
                    "domain": parsed_url.netloc,
                    "path": parsed_url.path,
                    "username": self._extract_username(url, platform),
                    "link_type": self._determine_link_type(url, platform),
                    "platform": platform
                }
                
                link_profiles[platform].append(metadata)
        
        return link_profiles
    
    def _extract_username(self, url: str, platform: str) -> Optional[str]:
        """Extract username from URL based on platform patterns"""
        patterns = self.platform_patterns.get(platform, [])
        
        for pattern in patterns:
            match = pattern.search(url)
            if match and match.groups():
                return match.group(1)
        
        return None
    
    def _determine_link_type(self, url: str, platform: str) -> str:
        """Determine the type of link based on URL and platform"""
        # Default link types by platform
        platform_link_types = {
            "github": "repository",
            "linkedin": "profile",
            "leetcode": "profile",
            "kaggle": "profile",
            "medium": "article",
            "behance": "portfolio",
            "dribbble": "portfolio",
            "figma": "design",
            "stackoverflow": "profile",
            "personal_website": "website"
        }
        
        # More specific detection based on URL patterns
        if platform == "github":
            if "/repositories" in url:
                return "repositories_list"
            elif "/stars" in url:
                return "starred_repositories"
            elif re.search(r"/[\w-]+/[\w-]+", url):
                return "repository"
        elif platform == "medium":
            if re.search(r"/[\w-]+/[\w-]+", url):
                return "article"
        
        # Return default type for platform or generic "link"
        return platform_link_types.get(platform, "link")