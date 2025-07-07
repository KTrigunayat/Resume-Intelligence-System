import json
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import requests
from datetime import datetime
import logging


class PlatformType(Enum):
    """Enumeration of supported platforms"""
    GITHUB = "github"
    LEETCODE = "leetcode"
    KAGGLE = "kaggle"
    FIGMA = "figma"
    DRIBBBLE = "dribbble"
    LINKEDIN = "linkedin"
    RESUME = "resume"
    CERTIFICATION = "certification"


@dataclass
class PlatformScore:
    """Individual platform score with metadata"""
    platform: PlatformType
    raw_score: float
    normalized_score: float  # 0-100
    confidence: float  # 0-1
    metadata: Dict[str, Any]
    last_updated: datetime


@dataclass
class WeightedScoreConfig:
    """Configuration for weighted scoring"""
    github_weight: float = 0.3
    leetcode_weight: float = 0.2
    certification_weight: float = 0.2
    design_weight: float = 0.1
    resume_weight: float = 0.1
    linkedin_weight: float = 0.1
    
    def __post_init__(self):
        """Validate that weights sum to 1.0"""
        total = (self.github_weight + self.leetcode_weight + self.certification_weight + 
                self.design_weight + self.resume_weight + self.linkedin_weight)
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass
class FinalScore:
    """Final aggregated score with breakdown"""
    final_score: float  # 0-100
    grade: str  # A-F
    platform_scores: Dict[str, PlatformScore]
    score_breakdown: Dict[str, float]
    confidence_score: float
    recommendations: List[str]
    analysis_metadata: Dict[str, Any]


class MultiPlatformWeightedScoreEngine:
    """
    Normalize and combine domain-specific scores from various platforms 
    into a single, interpretable final score.
    """
    
    def __init__(self, config: Optional[WeightedScoreConfig] = None):
        """
        Initialize the Multi-Platform Weighted Score Engine.
        
        Args:
            config: Scoring configuration with platform weights
        """
        self.config = config or WeightedScoreConfig()
        self.logger = logging.getLogger(__name__)
        
        # Platform-specific scoring functions
        self.platform_scorers = {
            PlatformType.GITHUB: self._score_github,
            PlatformType.LEETCODE: self._score_leetcode,
            PlatformType.KAGGLE: self._score_kaggle,
            PlatformType.FIGMA: self._score_figma,
            PlatformType.DRIBBBLE: self._score_dribbble,
            PlatformType.LINKEDIN: self._score_linkedin,
            PlatformType.RESUME: self._score_resume,
            PlatformType.CERTIFICATION: self._score_certification
        }
    
    def calculate_final_score(self, platform_data: Dict[str, Any]) -> FinalScore:
        """
        Calculate final weighted score from all platform data.
        
        Args:
            platform_data: Dictionary containing data from all platforms
            
        Returns:
            FinalScore object with comprehensive scoring results
        """
        platform_scores = {}
        score_breakdown = {}
        total_weighted_score = 0.0
        total_confidence = 0.0
        
        # Process each platform
        for platform_type in PlatformType:
            platform_key = platform_type.value
            
            if platform_key in platform_data:
                try:
                    # Calculate platform-specific score
                    platform_score = self.platform_scorers[platform_type](
                        platform_data[platform_key]
                    )
                    platform_scores[platform_key] = platform_score
                    
                    # Get weight for this platform
                    weight = self._get_platform_weight(platform_type)
                    
                    # Add to weighted total
                    weighted_contribution = platform_score.normalized_score * weight
                    total_weighted_score += weighted_contribution
                    total_confidence += platform_score.confidence * weight
                    
                    score_breakdown[platform_key] = {
                        'raw_score': platform_score.raw_score,
                        'normalized_score': platform_score.normalized_score,
                        'weight': weight,
                        'weighted_contribution': weighted_contribution,
                        'confidence': platform_score.confidence
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error scoring platform {platform_key}: {e}")
                    # Create default score for failed platforms
                    platform_scores[platform_key] = PlatformScore(
                        platform=platform_type,
                        raw_score=0.0,
                        normalized_score=0.0,
                        confidence=0.0,
                        metadata={'error': str(e)},
                        last_updated=datetime.now()
                    )
        
        # Calculate final metrics
        final_score = min(max(total_weighted_score, 0.0), 100.0)
        grade = self._calculate_grade(final_score)
        recommendations = self._generate_recommendations(platform_scores, score_breakdown)
        
        return FinalScore(
            final_score=final_score,
            grade=grade,
            platform_scores=platform_scores,
            score_breakdown=score_breakdown,
            confidence_score=total_confidence,
            recommendations=recommendations,
            analysis_metadata={
                'analysis_timestamp': datetime.now().isoformat(),
                'platforms_analyzed': list(platform_scores.keys()),
                'config_used': asdict(self.config)
            }
        )
    
    def _get_platform_weight(self, platform: PlatformType) -> float:
        """Get weight for specific platform"""
        weight_mapping = {
            PlatformType.GITHUB: self.config.github_weight,
            PlatformType.LEETCODE: self.config.leetcode_weight,
            PlatformType.CERTIFICATION: self.config.certification_weight,
            PlatformType.FIGMA: self.config.design_weight,
            PlatformType.DRIBBBLE: self.config.design_weight,
            PlatformType.RESUME: self.config.resume_weight,
            PlatformType.LINKEDIN: self.config.linkedin_weight,
            PlatformType.KAGGLE: self.config.leetcode_weight  # Use leetcode weight for data science
        }
        return weight_mapping.get(platform, 0.0)
    
    def _score_github(self, github_data: Dict[str, Any]) -> PlatformScore:
        """
        Score GitHub platform data.
        
        Args:
            github_data: GitHub analysis results
            
        Returns:
            PlatformScore for GitHub
        """
        try:
            # Extract key metrics
            technical_score = github_data.get('technical_skill_score', 0)
            originality = github_data.get('originality_percentage', 0)
            
            # Calculate raw score (combination of technical skill and originality)
            raw_score = (technical_score * 0.7) + (originality * 0.3)
            
            # Normalize to 0-100
            normalized_score = min(max(raw_score, 0.0), 100.0)
            
            # Calculate confidence based on data completeness
            repo_count = len(github_data.get('repository_analysis', {}).get('repository_details', []))
            confidence = min(repo_count / 5.0, 1.0)  # Full confidence with 5+ repos
            
            return PlatformScore(
                platform=PlatformType.GITHUB,
                raw_score=raw_score,
                normalized_score=normalized_score,
                confidence=confidence,
                metadata={
                    'repositories_analyzed': repo_count,
                    'code_quality_grade': github_data.get('code_quality_grade', 'N/A'),
                    'languages': github_data.get('repository_analysis', {}).get('languages_used', {})
                },
                last_updated=datetime.now()
            )
            
        except Exception as e:
            return self._create_error_score(PlatformType.GITHUB, str(e))
    
    def _score_leetcode(self, leetcode_data: Dict[str, Any]) -> PlatformScore:
        """
        Score LeetCode platform data.
        
        Args:
            leetcode_data: LeetCode profile data
            
        Returns:
            PlatformScore for LeetCode
        """
        try:
            # Extract metrics
            problems_solved = leetcode_data.get('problems_solved', 0)
            acceptance_rate = leetcode_data.get('acceptance_rate', 0)
            contest_rating = leetcode_data.get('contest_rating', 0)
            
            # Calculate algorithmic score
            # Base score from problems solved (logarithmic scaling)
            problems_score = min(np.log10(max(problems_solved, 1)) * 25, 60)
            
            # Bonus from acceptance rate
            acceptance_bonus = acceptance_rate * 0.2
            
            # Bonus from contest performance
            contest_bonus = min(contest_rating / 50, 20) if contest_rating > 0 else 0
            
            raw_score = problems_score + acceptance_bonus + contest_bonus
            normalized_score = min(max(raw_score, 0.0), 100.0)
            
            # Confidence based on activity level
            confidence = min(problems_solved / 100.0, 1.0)
            
            return PlatformScore(
                platform=PlatformType.LEETCODE,
                raw_score=raw_score,
                normalized_score=normalized_score,
                confidence=confidence,
                metadata={
                    'problems_solved': problems_solved,
                    'acceptance_rate': acceptance_rate,
                    'contest_rating': contest_rating
                },
                last_updated=datetime.now()
            )
            
        except Exception as e:
            return self._create_error_score(PlatformType.LEETCODE, str(e))
    
    def _score_kaggle(self, kaggle_data: Dict[str, Any]) -> PlatformScore:
        """
        Score Kaggle platform data for data science depth.
        
        Args:
            kaggle_data: Kaggle profile data
            
        Returns:
            PlatformScore for Kaggle
        """
        try:
            # Extract metrics
            competitions_joined = kaggle_data.get('competitions_joined', 0)
            highest_rank = kaggle_data.get('highest_rank', float('inf'))
            datasets_created = kaggle_data.get('datasets_created', 0)
            notebooks_published = kaggle_data.get('notebooks_published', 0)
            
            # Calculate data science score
            # Competition performance (inverse rank scoring)
            comp_score = 0
            if highest_rank != float('inf') and highest_rank > 0:
                comp_score = max(50 - np.log10(highest_rank) * 10, 0)
            
            # Activity score
            activity_score = min((competitions_joined * 5) + (datasets_created * 3) + (notebooks_published * 2), 50)
            
            raw_score = comp_score + activity_score
            normalized_score = min(max(raw_score, 0.0), 100.0)
            
            # Confidence based on overall activity
            total_activity = competitions_joined + datasets_created + notebooks_published
            confidence = min(total_activity / 10.0, 1.0)
            
            return PlatformScore(
                platform=PlatformType.KAGGLE,
                raw_score=raw_score,
                normalized_score=normalized_score,
                confidence=confidence,
                metadata={
                    'competitions_joined': competitions_joined,
                    'highest_rank': highest_rank,
                    'datasets_created': datasets_created,
                    'notebooks_published': notebooks_published
                },
                last_updated=datetime.now()
            )
            
        except Exception as e:
            return self._create_error_score(PlatformType.KAGGLE, str(e))
    
    def _score_figma(self, figma_data: Dict[str, Any]) -> PlatformScore:
        """
        Score Figma platform data for design skills.
        
        Args:
            figma_data: Figma profile data
            
        Returns:
            PlatformScore for Figma
        """
        try:
            # Extract metrics
            projects_count = figma_data.get('projects_count', 0)
            likes_received = figma_data.get('likes_received', 0)
            views_count = figma_data.get('views_count', 0)
            
            # Calculate creative score
            project_score = min(projects_count * 10, 40)
            engagement_score = min((likes_received * 2) + (views_count / 100), 40)
            quality_score = 20  # Base quality score
            
            raw_score = project_score + engagement_score + quality_score
            normalized_score = min(max(raw_score, 0.0), 100.0)
            
            # Confidence based on portfolio size
            confidence = min(projects_count / 5.0, 1.0)
            
            return PlatformScore(
                platform=PlatformType.FIGMA,
                raw_score=raw_score,
                normalized_score=normalized_score,
                confidence=confidence,
                metadata={
                    'projects_count': projects_count,
                    'likes_received': likes_received,
                    'views_count': views_count
                },
                last_updated=datetime.now()
            )
            
        except Exception as e:
            return self._create_error_score(PlatformType.FIGMA, str(e))
    
    def _score_dribbble(self, dribbble_data: Dict[str, Any]) -> PlatformScore:
        """
        Score Dribbble platform data for design skills.
        
        Args:
            dribbble_data: Dribbble profile data
            
        Returns:
            PlatformScore for Dribbble
        """
        try:
            # Extract metrics
            shots_count = dribbble_data.get('shots_count', 0)
            likes_received = dribbble_data.get('likes_received', 0)
            followers_count = dribbble_data.get('followers_count', 0)
            
            # Calculate creative score
            portfolio_score = min(shots_count * 5, 40)
            popularity_score = min((likes_received / 10) + (followers_count / 20), 40)
            consistency_score = 20  # Base consistency score
            
            raw_score = portfolio_score + popularity_score + consistency_score
            normalized_score = min(max(raw_score, 0.0), 100.0)
            
            # Confidence based on portfolio activity
            confidence = min(shots_count / 10.0, 1.0)
            
            return PlatformScore(
                platform=PlatformType.DRIBBBLE,
                raw_score=raw_score,
                normalized_score=normalized_score,
                confidence=confidence,
                metadata={
                    'shots_count': shots_count,
                    'likes_received': likes_received,
                    'followers_count': followers_count
                },
                last_updated=datetime.now()
            )
            
        except Exception as e:
            return self._create_error_score(PlatformType.DRIBBBLE, str(e))
    
    def _score_linkedin(self, linkedin_data: Dict[str, Any]) -> PlatformScore:
        """
        Score LinkedIn platform data for social trust.
        
        Args:
            linkedin_data: LinkedIn profile data
            
        Returns:
            PlatformScore for LinkedIn
        """
        try:
            # Extract metrics
            connections_count = linkedin_data.get('connections_count', 0)
            endorsements_count = linkedin_data.get('endorsements_count', 0)
            recommendations_count = linkedin_data.get('recommendations_count', 0)
            posts_engagement = linkedin_data.get('posts_engagement', 0)
            
            # Calculate social trust score
            network_score = min(np.log10(max(connections_count, 1)) * 15, 30)
            credibility_score = min((endorsements_count * 2) + (recommendations_count * 5), 40)
            activity_score = min(posts_engagement / 10, 30)
            
            raw_score = network_score + credibility_score + activity_score
            normalized_score = min(max(raw_score, 0.0), 100.0)
            
            # Confidence based on profile completeness
            profile_completeness = linkedin_data.get('profile_completeness', 0.5)
            confidence = profile_completeness
            
            return PlatformScore(
                platform=PlatformType.LINKEDIN,
                raw_score=raw_score,
                normalized_score=normalized_score,
                confidence=confidence,
                metadata={
                    'connections_count': connections_count,
                    'endorsements_count': endorsements_count,
                    'recommendations_count': recommendations_count,
                    'posts_engagement': posts_engagement
                },
                last_updated=datetime.now()
            )
            
        except Exception as e:
            return self._create_error_score(PlatformType.LINKEDIN, str(e))
    
    def _score_resume(self, resume_data: Dict[str, Any]) -> PlatformScore:
        """
        Score resume quality data.
        
        Args:
            resume_data: Resume analysis results
            
        Returns:
            PlatformScore for resume
        """
        try:
            # Extract metrics from existing resume analysis
            overall_score = resume_data.get('overall_score', 0)
            formatting_score = resume_data.get('formatting_score', 0)
            content_score = resume_data.get('content_score', 0)
            
            # Use overall score as base, with adjustments
            raw_score = overall_score
            normalized_score = min(max(raw_score, 0.0), 100.0)
            
            # Confidence based on analysis completeness
            confidence = 0.9  # High confidence in resume analysis
            
            return PlatformScore(
                platform=PlatformType.RESUME,
                raw_score=raw_score,
                normalized_score=normalized_score,
                confidence=confidence,
                metadata={
                    'formatting_score': formatting_score,
                    'content_score': content_score,
                    'analysis_components': list(resume_data.keys())
                },
                last_updated=datetime.now()
            )
            
        except Exception as e:
            return self._create_error_score(PlatformType.RESUME, str(e))
    
    def _score_certification(self, certification_data: Dict[str, Any]) -> PlatformScore:
        """
        Score certification data.
        
        Args:
            certification_data: Certification information
            
        Returns:
            PlatformScore for certifications
        """
        try:
            # Extract metrics
            certifications_count = len(certification_data.get('certifications', []))
            verified_count = len([cert for cert in certification_data.get('certifications', []) 
                                if cert.get('verified', False)])
            
            # Calculate certification score
            quantity_score = min(certifications_count * 15, 60)
            quality_score = min(verified_count * 20, 40)
            
            raw_score = quantity_score + quality_score
            normalized_score = min(max(raw_score, 0.0), 100.0)
            
            # Confidence based on verification rate
            confidence = verified_count / max(certifications_count, 1) if certifications_count > 0 else 0
            
            return PlatformScore(
                platform=PlatformType.CERTIFICATION,
                raw_score=raw_score,
                normalized_score=normalized_score,
                confidence=confidence,
                metadata={
                    'total_certifications': certifications_count,
                    'verified_certifications': verified_count,
                    'certification_details': certification_data.get('certifications', [])
                },
                last_updated=datetime.now()
            )
            
        except Exception as e:
            return self._create_error_score(PlatformType.CERTIFICATION, str(e))
    
    def _create_error_score(self, platform: PlatformType, error_msg: str) -> PlatformScore:
        """Create a default error score for failed platform analysis"""
        return PlatformScore(
            platform=platform,
            raw_score=0.0,
            normalized_score=0.0,
            confidence=0.0,
            metadata={'error': error_msg},
            last_updated=datetime.now()
        )
    
    def _calculate_grade(self, score: float) -> str:
        """Convert numerical score to letter grade"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _generate_recommendations(self, platform_scores: Dict[str, PlatformScore], 
                                score_breakdown: Dict[str, Dict]) -> List[str]:
        """
        Generate actionable recommendations based on platform scores.
        
        Args:
            platform_scores: Dictionary of platform scores
            score_breakdown: Detailed score breakdown
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Analyze each platform for improvement opportunities
        for platform_key, score in platform_scores.items():
            if score.normalized_score < 50:
                if platform_key == 'github':
                    recommendations.append(
                        "Improve GitHub presence: Add more original projects, increase code complexity, and add comprehensive tests"
                    )
                elif platform_key == 'leetcode':
                    recommendations.append(
                        "Enhance algorithmic skills: Solve more LeetCode problems and participate in contests"
                    )
                elif platform_key == 'linkedin':
                    recommendations.append(
                        "Strengthen LinkedIn profile: Increase connections, gather endorsements, and post regularly"
                    )
                elif platform_key == 'resume':
                    recommendations.append(
                        "Improve resume quality: Focus on formatting consistency and content relevance"
                    )
                elif platform_key == 'certification':
                    recommendations.append(
                        "Obtain relevant certifications: Focus on verified, industry-recognized credentials"
                    )
        
        # Add general recommendations based on overall performance
        avg_score = np.mean([score.normalized_score for score in platform_scores.values()])
        
        if avg_score < 60:
            recommendations.append(
                "Focus on building a stronger technical foundation across all platforms"
            )
        elif avg_score < 80:
            recommendations.append(
                "Good foundation - focus on specializing in your strongest areas while addressing weak points"
            )
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def update_weights(self, new_config: WeightedScoreConfig) -> None:
        """
        Update scoring weights configuration.
        
        Args:
            new_config: New weight configuration
        """
        self.config = new_config
    
    def save_results(self, final_score: FinalScore, output_path: str) -> None:
        """
        Save final scoring results to JSON file.
        
        Args:
            final_score: FinalScore object to save
            output_path: Path to save the results
        """
        # Convert to serializable format
        result = {
            'final_score': final_score.final_score,
            'grade': final_score.grade,
            'confidence_score': final_score.confidence_score,
            'platform_scores': {
                platform: {
                    'platform': score.platform.value,
                    'raw_score': score.raw_score,
                    'normalized_score': score.normalized_score,
                    'confidence': score.confidence,
                    'metadata': score.metadata,
                    'last_updated': score.last_updated.isoformat()
                }
                for platform, score in final_score.platform_scores.items()
            },
            'score_breakdown': final_score.score_breakdown,
            'recommendations': final_score.recommendations,
            'analysis_metadata': final_score.analysis_metadata
        }
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)