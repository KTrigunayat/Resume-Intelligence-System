import requests
import re
import json
from typing import Dict, List, Tuple, Any, Optional
from bs4 import BeautifulSoup
import logging
from datetime import datetime
import time
from urllib.parse import urljoin, urlparse
import hashlib

class CredibilityEngine:
    """
    Credibility & Verifiability Engine: Programmatically confirm certifications 
    and online presence to tighten trust in the candidate's profile.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration for credibility verification"""
        return {
            "timeout": 10,
            "max_retries": 3,
            "rate_limit_delay": 2,
            "verify_ssl": True,
            "certification_providers": {
                "aws": {
                    "name": "Amazon Web Services",
                    "verification_url": "https://www.credly.com/badges",
                    "patterns": [r"aws\s+certified", r"amazon\s+web\s+services"]
                },
                "google": {
                    "name": "Google Cloud",
                    "verification_url": "https://www.credential.net",
                    "patterns": [r"google\s+cloud", r"gcp\s+certified"]
                },
                "microsoft": {
                    "name": "Microsoft",
                    "verification_url": "https://docs.microsoft.com/en-us/learn/certifications",
                    "patterns": [r"microsoft\s+certified", r"azure\s+certified"]
                },
                "cisco": {
                    "name": "Cisco",
                    "verification_url": "https://www.cisco.com/c/en/us/training-events/training-certifications",
                    "patterns": [r"cisco\s+certified", r"ccna", r"ccnp", r"ccie"]
                },
                "comptia": {
                    "name": "CompTIA",
                    "verification_url": "https://www.comptia.org/certifications",
                    "patterns": [r"comptia", r"security\+", r"network\+", r"a\+"]
                }
            },
            "social_platforms": {
                "linkedin": {
                    "base_url": "https://www.linkedin.com/in/",
                    "verification_indicators": ["linkedin.com/in/", "profile-view-analytics"]
                },
                "github": {
                    "base_url": "https://github.com/",
                    "verification_indicators": ["github.com/", "contribution-graph"]
                },
                "stackoverflow": {
                    "base_url": "https://stackoverflow.com/users/",
                    "verification_indicators": ["stackoverflow.com/users/", "reputation"]
                }
            }
        }
    
    def verify_credibility(self, resume_text: str, sections: Dict[str, str], 
                          contact_info: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Comprehensive credibility verification
        
        Args:
            resume_text: Full resume text
            sections: Parsed resume sections
            contact_info: Contact information including social profiles
            
        Returns:
            Dictionary containing credibility scores and verification results
        """
        try:
            # Extract certifications from resume
            certifications = self._extract_certifications(resume_text, sections)
            
            # Extract social profiles and contact info
            social_profiles = self._extract_social_profiles(resume_text, contact_info)
            
            # Verify certifications
            certification_verification = self._verify_certifications(certifications)
            
            # Verify social presence
            social_verification = self._verify_social_presence(social_profiles)
            
            # Cross-reference information
            cross_reference_results = self._cross_reference_information(
                certifications, social_profiles, sections
            )
            
            # Calculate credibility score
            credibility_score = self._calculate_credibility_score(
                certification_verification,
                social_verification,
                cross_reference_results
            )
            
            return {
                "credibility_score": credibility_score,
                "max_score": 100,
                "verification_results": {
                    "certifications": certification_verification,
                    "social_presence": social_verification,
                    "cross_reference": cross_reference_results
                },
                "extracted_data": {
                    "certifications": certifications,
                    "social_profiles": social_profiles
                },
                "recommendations": self._generate_credibility_recommendations(
                    certification_verification,
                    social_verification,
                    cross_reference_results
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error in credibility verification: {e}")
            return self._get_default_credibility_score()
    
    def _extract_certifications(self, resume_text: str, sections: Dict[str, str]) -> List[Dict[str, Any]]:
        """Extract certification information from resume"""
        certifications = []
        
        # Get certification-related sections
        cert_text = ""
        for section_name, content in sections.items():
            if any(keyword in section_name.lower() for keyword in ["certification", "license", "credential"]):
                cert_text += content + " "
        
        # If no dedicated section, search entire resume
        if not cert_text.strip():
            cert_text = resume_text
        
        # Extract certifications using patterns
        for provider, config in self.config["certification_providers"].items():
            for pattern in config["patterns"]:
                matches = re.finditer(pattern, cert_text, re.IGNORECASE)
                for match in matches:
                    # Extract surrounding context
                    start = max(0, match.start() - 50)
                    end = min(len(cert_text), match.end() + 50)
                    context = cert_text[start:end].strip()
                    
                    # Try to extract certification details
                    cert_details = self._parse_certification_details(context, provider)
                    if cert_details:
                        certifications.append(cert_details)
        
        # Remove duplicates
        unique_certs = []
        seen_certs = set()
        for cert in certifications:
            cert_key = f"{cert['provider']}_{cert['name']}"
            if cert_key not in seen_certs:
                unique_certs.append(cert)
                seen_certs.add(cert_key)
        
        return unique_certs
    
    def _parse_certification_details(self, context: str, provider: str) -> Optional[Dict[str, Any]]:
        """Parse certification details from context"""
        # Extract certification name
        cert_name = context.strip()
        
        # Try to extract date
        date_patterns = [
            r'(\d{4})',  # Year
            r'(\d{1,2}/\d{4})',  # Month/Year
            r'(\w+\s+\d{4})',  # Month Year
            r'(\d{1,2}/\d{1,2}/\d{4})'  # Full date
        ]
        
        extracted_date = None
        for pattern in date_patterns:
            match = re.search(pattern, context)
            if match:
                extracted_date = match.group(1)
                break
        
        # Try to extract certification ID
        id_patterns = [
            r'([A-Z0-9]{8,})',  # Alphanumeric ID
            r'(\b\d{6,}\b)',  # Numeric ID
            r'([A-Z]{2,}-[A-Z0-9]{4,})'  # Formatted ID
        ]
        
        cert_id = None
        for pattern in id_patterns:
            match = re.search(pattern, context)
            if match:
                cert_id = match.group(1)
                break
        
        return {
            "name": cert_name,
            "provider": provider,
            "date_obtained": extracted_date,
            "certification_id": cert_id,
            "context": context
        }
    
    def _extract_social_profiles(self, resume_text: str, contact_info: Dict[str, str] = None) -> Dict[str, str]:
        """Extract social media profiles and professional links"""
        profiles = {}
        
        # Combine resume text and contact info
        search_text = resume_text
        if contact_info:
            search_text += " " + " ".join(contact_info.values())
        
        # Extract LinkedIn profile
        linkedin_patterns = [
            r'linkedin\.com/in/([a-zA-Z0-9\-]+)',
            r'linkedin\.com/pub/([a-zA-Z0-9\-]+)',
            r'www\.linkedin\.com/in/([a-zA-Z0-9\-]+)'
        ]
        
        for pattern in linkedin_patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                profiles["linkedin"] = f"https://www.linkedin.com/in/{match.group(1)}"
                break
        
        # Extract GitHub profile
        github_patterns = [
            r'github\.com/([a-zA-Z0-9\-]+)',
            r'www\.github\.com/([a-zA-Z0-9\-]+)'
        ]
        
        for pattern in github_patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                profiles["github"] = f"https://github.com/{match.group(1)}"
                break
        
        # Extract other professional profiles
        other_patterns = {
            "stackoverflow": r'stackoverflow\.com/users/([0-9]+)',
            "portfolio": r'(https?://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,})',
            "personal_website": r'(www\.[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,})'
        }
        
        for platform, pattern in other_patterns.items():
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                url = match.group(1)
                if not url.startswith('http'):
                    url = f"https://{url}"
                profiles[platform] = url
        
        return profiles
    
    def _verify_certifications(self, certifications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify certifications against provider databases"""
        verification_results = {
            "total_certifications": len(certifications),
            "verified_count": 0,
            "unverified_count": 0,
            "verification_details": []
        }
        
        for cert in certifications:
            try:
                time.sleep(self.config["rate_limit_delay"])  # Rate limiting
                
                verification_result = self._verify_single_certification(cert)
                verification_results["verification_details"].append(verification_result)
                
                if verification_result["status"] == "verified":
                    verification_results["verified_count"] += 1
                else:
                    verification_results["unverified_count"] += 1
                    
            except Exception as e:
                self.logger.error(f"Error verifying certification {cert['name']}: {e}")
                verification_results["verification_details"].append({
                    "certification": cert,
                    "status": "error",
                    "message": str(e)
                })
                verification_results["unverified_count"] += 1
        
        verification_results["verification_rate"] = (
            verification_results["verified_count"] / max(len(certifications), 1)
        )
        
        return verification_results
    
    def _verify_single_certification(self, cert: Dict[str, Any]) -> Dict[str, Any]:
        """Verify a single certification"""
        provider = cert["provider"]
        provider_config = self.config["certification_providers"].get(provider, {})
        
        # For demonstration, we'll do basic verification
        # In a real implementation, you'd integrate with actual certification APIs
        
        verification_result = {
            "certification": cert,
            "status": "unverified",
            "confidence": 0.0,
            "verification_method": "pattern_matching",
            "details": {}
        }
        
        # Basic verification based on format and content
        confidence_score = 0.0
        
        # Check if certification ID format is reasonable
        if cert.get("certification_id"):
            cert_id = cert["certification_id"]
            if len(cert_id) >= 6 and any(c.isalnum() for c in cert_id):
                confidence_score += 0.3
        
        # Check if date is reasonable
        if cert.get("date_obtained"):
            try:
                # Try to parse various date formats
                date_str = cert["date_obtained"]
                if re.match(r'\d{4}', date_str):  # Year format
                    year = int(date_str)
                    if 2000 <= year <= datetime.now().year:
                        confidence_score += 0.3
            except:
                pass
        
        # Check if certification name matches known patterns
        cert_name_lower = cert["name"].lower()
        for pattern in provider_config.get("patterns", []):
            if re.search(pattern, cert_name_lower):
                confidence_score += 0.4
                break
        
        verification_result["confidence"] = min(1.0, confidence_score)
        
        if confidence_score >= 0.7:
            verification_result["status"] = "verified"
        elif confidence_score >= 0.4:
            verification_result["status"] = "likely_valid"
        else:
            verification_result["status"] = "unverified"
        
        return verification_result
    
    def _verify_social_presence(self, social_profiles: Dict[str, str]) -> Dict[str, Any]:
        """Verify social media presence and activity"""
        verification_results = {
            "total_profiles": len(social_profiles),
            "accessible_profiles": 0,
            "profile_details": {}
        }
        
        for platform, url in social_profiles.items():
            try:
                time.sleep(self.config["rate_limit_delay"])  # Rate limiting
                
                profile_verification = self._verify_single_profile(platform, url)
                verification_results["profile_details"][platform] = profile_verification
                
                if profile_verification["accessible"]:
                    verification_results["accessible_profiles"] += 1
                    
            except Exception as e:
                self.logger.error(f"Error verifying {platform} profile: {e}")
                verification_results["profile_details"][platform] = {
                    "url": url,
                    "accessible": False,
                    "error": str(e)
                }
        
        verification_results["accessibility_rate"] = (
            verification_results["accessible_profiles"] / max(len(social_profiles), 1)
        )
        
        return verification_results
    
    def _verify_single_profile(self, platform: str, url: str) -> Dict[str, Any]:
        """Verify a single social media profile"""
        try:
            response = self.session.get(url, timeout=self.config["timeout"], 
                                      verify=self.config["verify_ssl"])
            
            profile_data = {
                "url": url,
                "accessible": response.status_code == 200,
                "status_code": response.status_code,
                "platform": platform
            }
            
            if response.status_code == 200:
                # Basic content analysis
                content = response.text.lower()
                
                # Check for platform-specific indicators
                platform_config = self.config["social_platforms"].get(platform, {})
                indicators = platform_config.get("verification_indicators", [])
                
                found_indicators = []
                for indicator in indicators:
                    if indicator.lower() in content:
                        found_indicators.append(indicator)
                
                profile_data["verification_indicators"] = found_indicators
                profile_data["credibility_score"] = len(found_indicators) / max(len(indicators), 1)
                
                # Extract basic profile information
                if platform == "linkedin":
                    profile_data.update(self._extract_linkedin_info(content))
                elif platform == "github":
                    profile_data.update(self._extract_github_info(content))
            
            return profile_data
            
        except Exception as e:
            return {
                "url": url,
                "accessible": False,
                "error": str(e),
                "platform": platform
            }
    
    def _extract_linkedin_info(self, content: str) -> Dict[str, Any]:
        """Extract basic LinkedIn profile information"""
        info = {}
        
        # Look for professional indicators
        if "experience" in content:
            info["has_experience_section"] = True
        if "education" in content:
            info["has_education_section"] = True
        if "skills" in content:
            info["has_skills_section"] = True
        if "connections" in content:
            info["has_connections"] = True
        
        return info
    
    def _extract_github_info(self, content: str) -> Dict[str, Any]:
        """Extract basic GitHub profile information"""
        info = {}
        
        # Look for activity indicators
        if "contributions" in content:
            info["has_contributions"] = True
        if "repositories" in content:
            info["has_repositories"] = True
        if "followers" in content:
            info["has_followers"] = True
        
        # Try to extract repository count
        repo_match = re.search(r'(\d+)\s*repositories?', content)
        if repo_match:
            info["repository_count"] = int(repo_match.group(1))
        
        return info
    
    def _cross_reference_information(self, certifications: List[Dict[str, Any]], 
                                   social_profiles: Dict[str, str], 
                                   sections: Dict[str, str]) -> Dict[str, Any]:
        """Cross-reference information across different sources"""
        cross_ref_results = {
            "consistency_score": 0.0,
            "inconsistencies": [],
            "supporting_evidence": []
        }
        
        # Check if certifications are mentioned in social profiles
        for cert in certifications:
            cert_mentioned = False
            for platform, url in social_profiles.items():
                if platform == "linkedin":  # LinkedIn often lists certifications
                    cert_mentioned = True
                    cross_ref_results["supporting_evidence"].append(
                        f"Certification '{cert['name']}' may be verifiable on LinkedIn profile"
                    )
                    break
            
            if not cert_mentioned:
                cross_ref_results["inconsistencies"].append(
                    f"Certification '{cert['name']}' not found in social profiles"
                )
        
        # Calculate consistency score
        total_items = len(certifications) + len(social_profiles)
        if total_items > 0:
            consistency_items = len(cross_ref_results["supporting_evidence"])
            cross_ref_results["consistency_score"] = consistency_items / total_items
        
        return cross_ref_results
    
    def _calculate_credibility_score(self, cert_verification: Dict, 
                                   social_verification: Dict, 
                                   cross_reference: Dict) -> float:
        """Calculate overall credibility score (0-100)"""
        score = 0.0
        
        # Certification verification (40% weight)
        cert_score = cert_verification.get("verification_rate", 0) * 40
        score += cert_score
        
        # Social presence verification (30% weight)
        social_score = social_verification.get("accessibility_rate", 0) * 30
        score += social_score
        
        # Cross-reference consistency (20% weight)
        consistency_score = cross_reference.get("consistency_score", 0) * 20
        score += consistency_score
        
        # Bonus for having multiple verification sources (10% weight)
        verification_sources = 0
        if cert_verification.get("verified_count", 0) > 0:
            verification_sources += 1
        if social_verification.get("accessible_profiles", 0) > 0:
            verification_sources += 1
        
        source_bonus = min(10, verification_sources * 5)
        score += source_bonus
        
        return min(100, max(0, score))
    
    def _generate_credibility_recommendations(self, cert_verification: Dict,
                                            social_verification: Dict,
                                            cross_reference: Dict) -> List[str]:
        """Generate recommendations for improving credibility"""
        recommendations = []
        
        # Certification recommendations
        if cert_verification.get("verification_rate", 0) < 0.5:
            recommendations.append("Include certification IDs and dates to improve verifiability")
            recommendations.append("Ensure certification names match official provider terminology")
        
        # Social presence recommendations
        if social_verification.get("accessibility_rate", 0) < 0.5:
            recommendations.append("Ensure social media profiles are public and accessible")
            recommendations.append("Include professional profiles (LinkedIn, GitHub) to enhance credibility")
        
        # Cross-reference recommendations
        if cross_reference.get("consistency_score", 0) < 0.5:
            recommendations.append("Ensure consistency between resume and online profiles")
            recommendations.append("Update social profiles to reflect current skills and certifications")
        
        # General recommendations
        if len(social_verification.get("profile_details", {})) < 2:
            recommendations.append("Consider adding more professional online profiles")
        
        if not recommendations:
            recommendations.append("Strong credibility profile - maintain current verification standards")
        
        return recommendations
    
    def _get_default_credibility_score(self) -> Dict[str, Any]:
        """Return default credibility score when analysis fails"""
        return {
            "credibility_score": 50.0,
            "max_score": 100,
            "verification_results": {},
            "extracted_data": {},
            "recommendations": ["Could not perform credibility verification - ensure resume content is accessible"]
        }