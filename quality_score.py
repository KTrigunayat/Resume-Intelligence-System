import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
from pathlib import Path

class QualityScoreEngine:
    """
    Final Quality Score & Issue Report: Fuse all sub-scores into a unified 
    résumé score (0–100) and emit a structured JSON report.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the quality score engine.
        
        Args:
            config (Dict[str, Any], optional): Custom configuration. Defaults to None.
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._validate_config(config) if config else self._get_default_config()
        
        # Create output directory if it doesn't exist
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize configuration.
        
        Args:
            config (Dict[str, Any]): Configuration to validate.
            
        Returns:
            Dict[str, Any]: Validated configuration.
        """
        default_config = self._get_default_config()
        
        # Validate weights
        weights = config.get("weights", {})
        if not isinstance(weights, dict):
            self.logger.warning("Invalid weights configuration. Using defaults.")
            weights = default_config["weights"]
        else:
            # Ensure all weights are between 0 and 1
            weights = {
                k: max(0, min(1, float(v)))
                for k, v in weights.items()
                if k in default_config["weights"]
            }
            # Normalize weights to sum to 1
            total = sum(weights.values())
            if total > 0:
                weights = {k: v/total for k, v in weights.items()}
            else:
                self.logger.warning("All weights are 0. Using defaults.")
                weights = default_config["weights"]
        
        # Validate grade thresholds
        thresholds = config.get("grade_thresholds", {})
        if not isinstance(thresholds, dict):
            self.logger.warning("Invalid grade thresholds. Using defaults.")
            thresholds = default_config["grade_thresholds"]
        else:
            # Ensure all thresholds are between 0 and 100
            thresholds = {
                k: max(0, min(100, int(v)))
                for k, v in thresholds.items()
                if k in default_config["grade_thresholds"]
            }
            # Sort thresholds in descending order
            thresholds = dict(sorted(thresholds.items(), key=lambda x: x[1], reverse=True))
        
        return {
            "weights": weights,
            "grade_thresholds": thresholds,
            "critical_issues_threshold": max(1, int(config.get("critical_issues_threshold", 3))),
            "warning_issues_threshold": max(1, int(config.get("warning_issues_threshold", 5)))
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for quality scoring.
        
        Returns:
            Dict[str, Any]: Default configuration.
        """
        return {
            "weights": {
                "skill_alignment": 0.30,  # 30% - Recalculated Skill Alignment
                "project_validation": 0.30,  # 30% - Project Validation
                "formatting": 0.10,  # 10% - Resume Formatting
                "trustworthiness": 0.10,  # 10% - Content Trustworthiness
                "credibility": 0.10,  # 10% - Credential Verification
                "online_presence": 0.10  # 10% - Online Presence
            },
            "grade_thresholds": {
                "A+": 95,
                "A": 90,
                "A-": 85,
                "B+": 80,
                "B": 75,
                "B-": 70,
                "C+": 65,
                "C": 60,
                "C-": 55,
                "D": 50,
                "F": 0
            },
            "critical_issues_threshold": 3,
            "warning_issues_threshold": 5
        }
    
    def calculate_final_quality_score(self, 
                                    skill_alignment_results: Dict[str, Any],
                                    project_validation_results: Dict[str, Any],
                                    formatting_results: Dict[str, Any],
                                    trustworthiness_results: Dict[str, Any],
                                    credibility_results: Dict[str, Any],
                                    resume_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Calculate final quality score and generate comprehensive report.
        
        Args:
            skill_alignment_results: Results from skill matcher
            project_validation_results: Results from project validator
            formatting_results: Results from formatting scorer
            trustworthiness_results: Results from trustworthiness detector
            credibility_results: Results from credibility engine
            resume_metadata: Additional metadata about the resume
            
        Returns:
            Dict[str, Any]: Comprehensive quality report with final score
        """
        try:
            # Validate input data
            self._validate_input_data(
                skill_alignment_results,
                project_validation_results,
                formatting_results,
                trustworthiness_results,
                credibility_results
            )
            
            # Extract individual scores
            scores = self._extract_component_scores(
                skill_alignment_results,
                project_validation_results,
                formatting_results,
                trustworthiness_results,
                credibility_results
            )
            
            # Add online presence score if available in metadata
            if resume_metadata and isinstance(resume_metadata, dict):
                scores["online_presence"] = resume_metadata.get("online_presence_score", 0)
            
            # Calculate weighted final score
            final_score = self._calculate_weighted_score(scores)
            
            # Determine grade
            grade = self._determine_grade(final_score)
            
            # Collect all issues and flags
            issues = self._collect_issues(
                skill_alignment_results,
                project_validation_results,
                formatting_results,
                trustworthiness_results,
                credibility_results
            )
            
            # Generate recommendations
            recommendations = self._generate_comprehensive_recommendations(
                scores, issues, skill_alignment_results, project_validation_results
            )
            
            # Create detailed report
            quality_report = {
                "overall_score": round(final_score, 2),
                "grade": grade,
                "max_score": 100,
                "analysis_timestamp": datetime.now().isoformat(),
                "component_scores": scores,
                "score_breakdown": self._create_score_breakdown(scores),
                "issues_summary": self._summarize_issues(issues),
                "detailed_issues": issues,
                "recommendations": recommendations,
                "strengths": self._identify_strengths(scores, skill_alignment_results, project_validation_results),
                "improvement_areas": self._identify_improvement_areas(scores, issues),
                "metadata": resume_metadata or {}
            }
            
            # Save report
            self.save_quality_report(quality_report)
            
            return quality_report
            
        except Exception as e:
            self.logger.error(f"Error calculating quality score: {e}")
            return self._get_default_quality_report()
    
    def _validate_input_data(self, *args):
        """Validate input data from all components.
        
        Args:
            *args: Results from different components
        """
        for arg in args:
            if not isinstance(arg, dict):
                raise ValueError(f"Invalid input data type: {type(arg)}")
    
    def _extract_component_scores(self, skill_alignment: Dict, project_validation: Dict,
                                formatting: Dict, trustworthiness: Dict, 
                                credibility: Dict) -> Dict[str, float]:
        """Extract normalized scores from each component.
        
        Args:
            skill_alignment: Skill alignment results
            project_validation: Project validation results
            formatting: Formatting results
            trustworthiness: Trustworthiness results
            credibility: Credibility results
            
        Returns:
            Dict[str, float]: Normalized component scores
        """
        scores = {}
        
        # Skill alignment score (already 0-100)
        scores["skill_alignment"] = self._normalize_score(
            skill_alignment.get("overall_alignment_score", 0)
        )
        
        # Project validation score
        project_score = self._extract_project_score(project_validation)
        scores["project_validation"] = self._normalize_score(project_score)
        
        # Formatting score
        formatting_score = self._extract_formatting_score(formatting)
        scores["formatting"] = self._normalize_score(formatting_score)
        
        # Trustworthiness score
        scores["trustworthiness"] = self._normalize_score(
            trustworthiness.get("trust_score", 50)
        )
        
        # Credibility score
        scores["credibility"] = self._normalize_score(
            credibility.get("credibility_score", 50)
        )
        
        return scores
    
    def _normalize_score(self, score: float) -> float:
        """Normalize a score to 0-100 range.
        
        Args:
            score (float): Raw score
            
        Returns:
            float: Normalized score
        """
        try:
            score = float(score)
            return min(100, max(0, score))
        except (ValueError, TypeError):
            return 0.0
    
    def _extract_project_score(self, project_validation: Dict) -> float:
        """Extract and normalize project validation score.
        
        Args:
            project_validation (Dict): Project validation results
            
        Returns:
            float: Normalized project score
        """
        if not isinstance(project_validation, dict):
            return 0.0
            
        # Try to get overall_score directly
        if "overall_score" in project_validation:
            score = project_validation["overall_score"]
            if isinstance(score, dict):
                score = score.get("total", 0)
            return self._normalize_score(score)
            
        # Calculate from project_scores
        if "project_scores" in project_validation and project_validation["project_scores"]:
            try:
                scores = project_validation["project_scores"]
                if isinstance(scores, dict):
                    avg_score = sum(scores.values()) / len(scores)
                    return self._normalize_score(avg_score * 100)
            except (TypeError, ValueError, ZeroDivisionError):
                pass
                
        return 0.0
    
    def _extract_formatting_score(self, formatting: Dict) -> float:
        """Extract and normalize formatting score.
        
        Args:
            formatting (Dict): Formatting results
            
        Returns:
            float: Normalized formatting score
        """
        if not isinstance(formatting, dict):
            return 0.0
            
        score = formatting.get("total_score", 0)
        max_score = formatting.get("max_score", 20)
        
        try:
            return self._normalize_score((score / max(max_score, 1)) * 100)
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.0
    
    def _calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted final score.
        
        Args:
            scores (Dict[str, float]): Component scores
            
        Returns:
            float: Weighted final score
        """
        weighted_score = 0.0
        total_weight = 0.0
        
        for component, score in scores.items():
            weight = self.config["weights"].get(component, 0)
            weighted_score += score * weight
            total_weight += weight
        
        # Normalize if weights don't sum to 1
        if total_weight > 0:
            weighted_score = weighted_score / total_weight
        
        return self._normalize_score(weighted_score)
    
    def _determine_grade(self, score: float) -> str:
        """Determine letter grade based on score.
        
        Args:
            score (float): Final score
            
        Returns:
            str: Letter grade
        """
        for grade, threshold in self.config["grade_thresholds"].items():
            if score >= threshold:
                return grade
        return "F"
    
    def _collect_issues(self, skill_alignment: Dict, project_validation: Dict,
                       formatting: Dict, trustworthiness: Dict, 
                       credibility: Dict) -> List[Dict[str, Any]]:
        """Collect all issues and flags from components.
        
        Args:
            skill_alignment: Skill alignment results
            project_validation: Project validation results
            formatting: Formatting results
            trustworthiness: Trustworthiness results
            credibility: Credibility results
            
        Returns:
            List[Dict[str, Any]]: List of issues
        """
        all_issues = []
        
        # Skill alignment issues
        missing_skills = skill_alignment.get("missing_skills", [])
        for skill in missing_skills:
            all_issues.append({
                "type": "missing_skill",
                "severity": "medium",
                "component": "skill_alignment",
                "message": f"Missing skill: {skill}",
                "recommendation": f"Consider adding experience or projects demonstrating {skill}"
            })
        
        # Project validation issues
        flagged_projects = project_validation.get("flagged_projects", [])
        for project in flagged_projects:
            if isinstance(project, str):
                all_issues.append({
                    "type": "project_issue",
                    "severity": "medium",
                    "component": "project_validation",
                    "message": f"Project issue: {project}",
                    "recommendation": "Improve project description with specific technologies and quantifiable results"
                })
            elif isinstance(project, dict):
                all_issues.append({
                    "type": "project_issue",
                    "severity": "medium",
                    "component": "project_validation",
                    "message": f"Project '{project.get('title', 'Unknown')}' flagged for review",
                    "details": project.get("issues", []),
                    "recommendation": "Improve project description with specific technologies and quantifiable results"
                })
        
        # Formatting issues
        formatting_recommendations = formatting.get("recommendations", [])
        for rec in formatting_recommendations:
            if "excellent" not in rec.lower():
                all_issues.append({
                    "type": "formatting_issue",
                    "severity": "low",
                    "component": "formatting",
                    "message": rec,
                    "recommendation": rec
                })
        
        # Trustworthiness flags
        trust_flags = trustworthiness.get("flags", [])
        for flag in trust_flags:
            if isinstance(flag, str):
                all_issues.append({
                    "type": "trust_issue",
                    "severity": "high",
                    "component": "trustworthiness",
                    "message": flag,
                    "recommendation": "Verify and provide evidence for claims"
                })
            elif isinstance(flag, dict):
                all_issues.append({
                    "type": "trust_issue",
                    "severity": flag.get("severity", "high"),
                    "component": "trustworthiness",
                    "message": flag.get("message", "Unknown trust issue"),
                    "recommendation": flag.get("recommendation", "Verify and provide evidence for claims")
                })
        
        # Credibility issues
        cred_issues = credibility.get("issues", [])
        for issue in cred_issues:
            if isinstance(issue, str):
                all_issues.append({
                    "type": "credibility_issue",
                    "severity": "high",
                    "component": "credibility",
                    "message": issue,
                    "recommendation": "Provide verification for credentials"
                })
            elif isinstance(issue, dict):
                all_issues.append({
                    "type": "credibility_issue",
                    "severity": issue.get("severity", "high"),
                    "component": "credibility",
                    "message": issue.get("message", "Unknown credibility issue"),
                    "recommendation": issue.get("recommendation", "Provide verification for credentials")
                })
        
        return all_issues
    
    def _summarize_issues(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize issues by type and severity.
        
        Args:
            issues (List[Dict[str, Any]]): List of issues
            
        Returns:
            Dict[str, Any]: Issue summary
        """
        summary = {
            "total_issues": len(issues),
            "by_severity": {
                "high": 0,
                "medium": 0,
                "low": 0
            },
            "by_type": {},
            "by_component": {}
        }
        
        for issue in issues:
            # Count by severity
            severity = issue.get("severity", "medium")
            summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
            
            # Count by type
            issue_type = issue.get("type", "unknown")
            summary["by_type"][issue_type] = summary["by_type"].get(issue_type, 0) + 1
            
            # Count by component
            component = issue.get("component", "unknown")
            summary["by_component"][component] = summary["by_component"].get(component, 0) + 1
        
        return summary
    
    def _create_score_breakdown(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """Create detailed score breakdown.
        
        Args:
            scores (Dict[str, float]): Component scores
            
        Returns:
            Dict[str, Any]: Score breakdown
        """
        breakdown = {
            "components": {},
            "weighted_contributions": {}
        }
        
        for component, score in scores.items():
            weight = self.config["weights"].get(component, 0)
            weighted_score = score * weight
            
            breakdown["components"][component] = {
                "raw_score": score,
                "weight": weight,
                "weighted_score": weighted_score
            }
            
            breakdown["weighted_contributions"][component] = weighted_score
        
        return breakdown
    
    def _get_default_quality_report(self) -> Dict[str, Any]:
        """Get default quality report for error cases.
        
        Returns:
            Dict[str, Any]: Default report
        """
        return {
            "overall_score": 0,
            "grade": "F",
            "max_score": 100,
            "analysis_timestamp": datetime.now().isoformat(),
            "component_scores": {},
            "score_breakdown": {},
            "issues_summary": {
                "total_issues": 0,
                "by_severity": {"high": 0, "medium": 0, "low": 0},
                "by_type": {},
                "by_component": {}
            },
            "detailed_issues": [],
            "recommendations": [],
            "strengths": [],
            "improvement_areas": [],
            "metadata": {},
            "error": "Error calculating quality score"
        }
    
    def save_quality_report(self, quality_report: Dict[str, Any], 
                          output_path: str = None) -> bool:
        """Save quality report to file.
        
        Args:
            quality_report (Dict[str, Any]): Quality report to save
            output_path (str, optional): Output file path. Defaults to None.
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = self.output_dir / f"quality_report_{timestamp}.json"
            
            # Convert numpy types to Python types
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            # Convert and save
            report_data = convert_numpy_types(quality_report)
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving quality report: {e}")
            return False
    
    def _generate_comprehensive_recommendations(self, scores: Dict[str, float], 
                                              issues: List[Dict[str, Any]],
                                              skill_alignment: Dict[str, Any],
                                              project_validation: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate comprehensive recommendations"""
        recommendations = {
            "immediate_actions": [],
            "skill_development": [],
            "content_improvement": [],
            "presentation_enhancement": [],
            "credibility_building": []
        }
        
        # Safe access to nested dictionaries
        def safe_get(d, *keys, default=None):
            """Safely get a value from nested dictionaries"""
            if not isinstance(d, dict):
                return default
            
            current = d
            for key in keys:
                if not isinstance(current, dict) or key not in current:
                    return default
                current = current[key]
            
            return current
        
        # Immediate actions based on critical issues
        high_severity_issues = [issue for issue in issues if issue.get("severity") == "high"]
        for issue in high_severity_issues:
            recommendations["immediate_actions"].append(issue.get("recommendation", issue.get("message")))
        
        # Skill development recommendations
        missing_skills = skill_alignment.get("missing_skills", [])
        if missing_skills:
            recommendations["skill_development"].extend([
                f"Develop proficiency in {skill}" for skill in missing_skills[:5]  # Top 5
            ])
        
        # Content improvement based on project validation
        flagged_projects = project_validation.get("flagged_projects", [])
        if flagged_projects:
            recommendations["content_improvement"].extend([
                "Add quantifiable results to project descriptions",
                "Include specific technologies and methodologies used",
                "Highlight problem-solving approaches and outcomes"
            ])
        
        # Presentation enhancement based on formatting score
        if scores.get("formatting", 100) < 70:
            recommendations["presentation_enhancement"].extend([
                "Improve resume formatting and visual consistency",
                "Use professional fonts and appropriate spacing",
                "Ensure consistent bullet point usage"
            ])
        
        # Credibility building based on scores
        if scores.get("credibility", 100) < 60:
            recommendations["credibility_building"].extend([
                "Add professional certifications with verification details",
                "Include links to professional profiles (LinkedIn, GitHub)",
                "Ensure consistency across all professional platforms"
            ])
        
        # Remove duplicates and empty categories
        for category in recommendations:
            recommendations[category] = list(set(recommendations[category]))
            if not recommendations[category]:
                recommendations[category] = ["No specific recommendations - maintain current standards"]
        
        return recommendations
    
    def _identify_strengths(self, scores: Dict[str, float], 
                          skill_alignment: Dict[str, Any],
                          project_validation: Dict[str, Any]) -> List[str]:
        """Identify resume strengths"""
        strengths = []
        
        # Safe access to nested dictionaries
        def safe_get(d, *keys, default=None):
            """Safely get a value from nested dictionaries"""
            if not isinstance(d, dict):
                return default
            
            current = d
            for key in keys:
                if not isinstance(current, dict) or key not in current:
                    return default
                current = current[key]
            
            return current
        
        # Score-based strengths
        for component, score in scores.items():
            if score >= 85:
                component_name = component.replace("_", " ").title()
                strengths.append(f"Excellent {component_name} (Score: {score:.1f}/100)")
            elif score >= 75:
                component_name = component.replace("_", " ").title()
                strengths.append(f"Strong {component_name} (Score: {score:.1f}/100)")
        
        # Skill-based strengths
        matched_skills = safe_get(skill_alignment, "matched_skills", default=[])
        if matched_skills and len(matched_skills) >= 3:
            strengths.append(f"Strong alignment with {len(matched_skills)} job requirements")
        
        # Project-based strengths
        top_projects = safe_get(project_validation, "top_projects", default=[])
        if top_projects:
            strengths.append(f"Has {len(top_projects)} well-documented projects")
        
        return strengths
    
    def _identify_improvement_areas(self, scores: Dict[str, float], 
                                   issues: List[Dict[str, Any]]) -> List[str]:
        """Identify areas for improvement"""
        improvement_areas = []
        
        # Safe access to nested dictionaries
        def safe_get(d, *keys, default=None):
            """Safely get a value from nested dictionaries"""
            if not isinstance(d, dict):
                return default
            
            current = d
            for key in keys:
                if not isinstance(current, dict) or key not in current:
                    return default
                current = current[key]
            
            return current
        
        # Score-based improvement areas
        for component, score in scores.items():
            if 65 <= score < 75:
                component_name = component.replace("_", " ").title()
                improvement_areas.append(f"{component_name} needs enhancement (Score: {score:.1f}/100)")
        
        # Issue-based improvement areas (medium severity)
        medium_severity_issues = [issue for issue in issues if issue.get("severity") == "medium"]
        if medium_severity_issues:
            for issue in medium_severity_issues[:3]:  # Limit to top 3
                improvement_areas.append(issue.get("description", "Unknown issue"))
        
        return improvement_areas