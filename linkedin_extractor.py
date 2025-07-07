import re
import json
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup

from resume_intelligence.platform_extractors import BaseExtractor
from resume_intelligence.utils.rate_limiter import RateLimiter

class LinkedInExtractor(BaseExtractor):
    """
    LinkedIn-specific data extractor.
    Extracts profile information, experience, education, and skills.
    
    Note: LinkedIn has strict scraping policies and rate limits.
    This extractor uses a combination of public profile scraping and
    optional API integration when credentials are provided.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.web_base = "https://www.linkedin.com"
        
        # Initialize rate limiter with LinkedIn-appropriate defaults
        # LinkedIn is sensitive to scraping, so use conservative defaults
        self._rate_limiter = RateLimiter(default_delay=2.0, jitter=1.0)
        
        # Check if API credentials are provided
        self.use_api = False
        if config and 'linkedin_api_key' in config and 'linkedin_api_secret' in config:
            self.use_api = True
            self.api_key = config['linkedin_api_key']
            self.api_secret = config['linkedin_api_secret']
    
    def extract_data(self, url: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract comprehensive data from a LinkedIn profile
        
        Args:
            url: LinkedIn profile URL
            metadata: Optional metadata about the URL from the link crawler
            
        Returns:
            Dictionary containing extracted LinkedIn data
        """
        # Extract username/vanity name from URL or metadata
        username = metadata.get("username") if metadata else None
        if not username:
            username = self._extract_username_from_url(url)
            
        if not username:
            self.logger.error(f"Could not extract LinkedIn username from {url}")
            return self._empty_result(url)
        
        # If API credentials are provided, use LinkedIn API
        if self.use_api:
            return self._extract_via_api(username)
        else:
            # Otherwise, use web scraping with proper rate limiting
            return self._extract_via_scraping(url, username)
    
    def _extract_username_from_url(self, url: str) -> Optional[str]:
        """Extract LinkedIn username/vanity name from URL"""
        # Profile patterns:
        # - linkedin.com/in/username
        # - linkedin.com/pub/username
        profile_match = re.search(r'linkedin\.com/(?:in|pub)/([\w-]+)(?:/|$)', url)
        if profile_match:
            return profile_match.group(1)
        return None
    
    def _extract_via_api(self, username: str) -> Dict[str, Any]:
        """Extract LinkedIn data using the LinkedIn API"""
        # This would use the LinkedIn API with proper authentication
        # For demonstration purposes, we'll return a placeholder
        # In a real implementation, this would make authenticated API calls
        
        self.logger.info(f"Using LinkedIn API to extract data for {username}")
        
        # Placeholder for API response
        # In a real implementation, this would contain actual API calls
        return {
            "platform": "linkedin",
            "link_type": "profile",
            "url": f"{self.web_base}/in/{username}",
            "username": username,
            "extraction_method": "api",
            "profile_data": {
                "name": "[API extraction would provide actual name]",
                "headline": "[API extraction would provide actual headline]",
                "location": "[API extraction would provide actual location]",
                "connections": "[API extraction would provide connection count]",
            },
            "experience": ["[API extraction would provide experience details]"],
            "education": ["[API extraction would provide education details]"],
            "skills": ["[API extraction would provide skills]"],
            "recommendations": ["[API extraction would provide recommendations]"],
            "extraction_successful": True,
            "note": "This is a placeholder. Actual implementation requires LinkedIn API credentials and proper API calls."
        }
    
    def _extract_via_scraping(self, url: str, username: str) -> Dict[str, Any]:
        """Extract LinkedIn data via web scraping with proper rate limiting"""
        # Note: LinkedIn actively prevents scraping, so this approach has limitations
        # and should be used with caution and respect for LinkedIn's terms of service
        
        self.logger.info(f"Using web scraping to extract data for LinkedIn profile: {username}")
        
        # Apply strict rate limiting to avoid being blocked
        self._apply_rate_limit(extended_delay=True)
        
        # Make request to the profile page
        response_html = self._make_request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
            }
        )
        
        # Check if we got a valid response
        if not response_html:
            self.logger.warning(f"Could not fetch LinkedIn profile for {username}")
            return self._empty_result(url)
        
        # Parse the HTML response
        try:
            soup = BeautifulSoup(response_html, 'html.parser')
            
            # Extract basic profile information
            profile_data = self._extract_profile_info(soup)
            
            # Extract experience information
            experience_data = self._extract_experience(soup)
            
            # Extract education information
            education_data = self._extract_education(soup)
            
            # Extract skills
            skills_data = self._extract_skills(soup)
            
            return {
                "platform": "linkedin",
                "link_type": "profile",
                "url": url,
                "username": username,
                "extraction_method": "scraping",
                "profile_data": profile_data,
                "experience": experience_data,
                "education": education_data,
                "skills": skills_data,
                "extraction_successful": True
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing LinkedIn profile: {str(e)}")
            return self._empty_result(url)
    
    def _extract_profile_info(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract basic profile information from LinkedIn page"""
        # Note: These selectors are examples and may need to be updated
        # as LinkedIn frequently changes their HTML structure
        
        profile_data = {}
        
        # Try to extract name
        try:
            name_elem = soup.select_one('.pv-top-card--list .inline.t-24.t-black.t-normal.break-words')
            if name_elem:
                profile_data["name"] = name_elem.get_text().strip()
        except Exception:
            pass
            
        # Try to extract headline
        try:
            headline_elem = soup.select_one('.pv-top-card--list .mt1.t-18.t-black.t-normal.break-words')
            if headline_elem:
                profile_data["headline"] = headline_elem.get_text().strip()
        except Exception:
            pass
            
        # Try to extract location
        try:
            location_elem = soup.select_one('.pv-top-card--list .t-16.t-black.t-normal.inline-block')
            if location_elem:
                profile_data["location"] = location_elem.get_text().strip()
        except Exception:
            pass
            
        # Try to extract connection count
        try:
            connection_elem = soup.select_one('.pv-top-card--list-container .t-16.t-black.t-normal')
            if connection_elem and 'connections' in connection_elem.get_text().lower():
                connections_text = connection_elem.get_text().strip()
                # Extract number from text like "500+ connections"
                connections_match = re.search(r'(\d+)\+?\s*connections', connections_text, re.IGNORECASE)
                if connections_match:
                    profile_data["connections"] = connections_match.group(1) + '+'
        except Exception:
            pass
            
        # Try to extract about section
        try:
            about_elem = soup.select_one('.pv-about-section .pv-about__summary-text')
            if about_elem:
                profile_data["about"] = about_elem.get_text().strip()
        except Exception:
            pass
            
        return profile_data
    
    def _extract_experience(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract experience information from LinkedIn page"""
        experience_list = []
        
        # Try to extract experience section
        try:
            experience_section = soup.select_one('#experience-section')
            if not experience_section:
                return experience_list
                
            experience_items = experience_section.select('li.pv-entity__position-group-pager')
            
            for item in experience_items:
                experience_data = {}
                
                # Try to extract company name
                try:
                    company_elem = item.select_one('.pv-entity__secondary-title')
                    if company_elem:
                        experience_data["company"] = company_elem.get_text().strip()
                except Exception:
                    pass
                    
                # Try to extract title
                try:
                    title_elem = item.select_one('.pv-entity__summary-info h3')
                    if title_elem:
                        experience_data["title"] = title_elem.get_text().strip()
                except Exception:
                    pass
                    
                # Try to extract date range
                try:
                    date_elem = item.select_one('.pv-entity__date-range span:nth-child(2)')
                    if date_elem:
                        experience_data["date_range"] = date_elem.get_text().strip()
                except Exception:
                    pass
                    
                # Try to extract location
                try:
                    location_elem = item.select_one('.pv-entity__location span:nth-child(2)')
                    if location_elem:
                        experience_data["location"] = location_elem.get_text().strip()
                except Exception:
                    pass
                    
                # Try to extract description
                try:
                    description_elem = item.select_one('.pv-entity__description')
                    if description_elem:
                        experience_data["description"] = description_elem.get_text().strip()
                except Exception:
                    pass
                    
                if experience_data:
                    experience_list.append(experience_data)
                    
        except Exception as e:
            self.logger.error(f"Error extracting experience: {str(e)}")
            
        return experience_list
    
    def _extract_education(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract education information from LinkedIn page"""
        education_list = []
        
        # Try to extract education section
        try:
            education_section = soup.select_one('#education-section')
            if not education_section:
                return education_list
                
            education_items = education_section.select('li.pv-education-entity')
            
            for item in education_items:
                education_data = {}
                
                # Try to extract school name
                try:
                    school_elem = item.select_one('.pv-entity__school-name')
                    if school_elem:
                        education_data["school"] = school_elem.get_text().strip()
                except Exception:
                    pass
                    
                # Try to extract degree
                try:
                    degree_elem = item.select_one('.pv-entity__degree-name .pv-entity__comma-item')
                    if degree_elem:
                        education_data["degree"] = degree_elem.get_text().strip()
                except Exception:
                    pass
                    
                # Try to extract field of study
                try:
                    field_elem = item.select_one('.pv-entity__fos .pv-entity__comma-item')
                    if field_elem:
                        education_data["field_of_study"] = field_elem.get_text().strip()
                except Exception:
                    pass
                    
                # Try to extract date range
                try:
                    date_elem = item.select_one('.pv-entity__dates span:nth-child(2)')
                    if date_elem:
                        education_data["date_range"] = date_elem.get_text().strip()
                except Exception:
                    pass
                    
                if education_data:
                    education_list.append(education_data)
                    
        except Exception as e:
            self.logger.error(f"Error extracting education: {str(e)}")
            
        return education_list
    
    def _extract_skills(self, soup: BeautifulSoup) -> List[str]:
        """Extract skills from LinkedIn page"""
        skills_list = []
        
        # Try to extract skills section
        try:
            skills_section = soup.select_one('.pv-skill-categories-section')
            if not skills_section:
                return skills_list
                
            skill_items = skills_section.select('.pv-skill-category-entity__name-text')
            
            for item in skill_items:
                skill_text = item.get_text().strip()
                if skill_text:
                    skills_list.append(skill_text)
                    
        except Exception as e:
            self.logger.error(f"Error extracting skills: {str(e)}")
            
        return skills_list
    
    def _apply_rate_limit(self, extended_delay: bool = False):
        """Apply strict rate limiting for LinkedIn scraping"""
        # LinkedIn is particularly sensitive to scraping, so we use extended delays
        if extended_delay:
            self._rate_limiter.delay(minimum_delay=5.0, jitter=2.0)
        else:
            self._rate_limiter.delay()
    
    def _empty_result(self, url: str) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            "platform": "linkedin",
            "link_type": "unknown",
            "url": url,
            "username": None,
            "extraction_successful": False,
            "error": "Could not extract data"
        }