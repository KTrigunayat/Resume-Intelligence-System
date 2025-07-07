#!/usr/bin/env python3
"""
Resume Intelligence System - End-to-End Pipeline

This script implements a comprehensive 4-step analysis pipeline for the Resume Intelligence System:
1. Initial Screening
2. In-Depth Analysis
3. Overall Quality Score Calculation
4. Comprehensive Report Generation
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import argparse
from datetime import datetime
import traceback

# Add the resume_intelligence package to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import components from the Resume Intelligence System
from resume_intelligence.section_detector import SectionDetector
from resume_intelligence.skill_matcher import SkillMatcher
from resume_intelligence.project_validator import ProjectValidator
from resume_intelligence.formatting_scorer import FormattingScorer
from resume_intelligence.trustworthiness_detector import TrustworthinessDetector
from resume_intelligence.credibility_engine import CredibilityEngine
from resume_intelligence.quality_score import QualityScoreEngine
from resume_intelligence.utils.document_parser import DocumentParser
from resume_intelligence.visualizer import Visualizer
from resume_intelligence.link_extraction_system import LinkExtractionSystem
from resume_intelligence.explainability_layer import ExplainabilityReportingLayer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ResumePipeline:
    """
    End-to-End Pipeline for Resume Intelligence System
    """
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize all components
        self.logger.info("Initializing Resume Intelligence System components...")
        self.section_detector = SectionDetector()
        self.skill_matcher = SkillMatcher()
        self.project_validator = ProjectValidator()
        self.formatting_scorer = FormattingScorer()
        self.trustworthiness_detector = TrustworthinessDetector()
        self.credibility_engine = CredibilityEngine()
        self.quality_engine = QualityScoreEngine()
        self.document_parser = DocumentParser()
        self.visualizer = Visualizer()
        self.explainability_layer = ExplainabilityReportingLayer(output_dir=str(self.output_dir))
        
        # Initialize the Link Extraction System
        self.link_extraction_system = LinkExtractionSystem({
            "enable_caching": True,
            "cache_dir": str(self.output_dir / "link_cache"),
            "max_concurrent_extractions": 3
        })
        
        # Define the alignment score threshold
        self.alignment_threshold = 15  # Lowered threshold to allow more resumes to pass
        
        self.logger.info("All components initialized successfully")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'analysis.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_pipeline(self, resume_path: str, job_description_path: str, 
                    github_username: str = None, 
                    contact_info: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Run the complete 4-step analysis pipeline
        
        Args:
            resume_path: Path to the resume file
            job_description_path: Path to the job description file
            github_username: Optional GitHub username for verification
            contact_info: Optional contact information for credibility checks
            
        Returns:
            Complete analysis results
        """
        self.logger.info(f"Starting 4-step analysis pipeline for {resume_path}")
        
        try:
            # Validate input files
            if not os.path.exists(resume_path):
                raise FileNotFoundError(f"Resume file not found: {resume_path}")
            if not os.path.exists(job_description_path):
                raise FileNotFoundError(f"Job description file not found: {job_description_path}")
            
            # Parse documents
            try:
                resume_text = self.document_parser.parse(resume_path)
                if not resume_text.strip():
                    raise ValueError(f"Resume file is empty: {resume_path}")
            except Exception as e:
                self.logger.error(f"Error parsing resume: {str(e)}")
                raise
            
            try:
                job_description = self.document_parser.parse(job_description_path)
                if not job_description.strip():
                    raise ValueError(f"Job description file is empty: {job_description_path}")
            except Exception as e:
                self.logger.error(f"Error parsing job description: {str(e)}")
                raise
            
            # STEP 1: INITIAL SCREENING
            try:
                initial_screening_results = self._run_initial_screening(resume_path, resume_text, job_description)
            except Exception as e:
                self.logger.error(f"Error in initial screening: {str(e)}")
                raise
            
            # Check if candidate passes the threshold
            skill_alignment_score = initial_screening_results["skill_alignment"].get("overall_alignment_score", 0)
            self.logger.info(f"Skill alignment score: {skill_alignment_score}")
            
            if skill_alignment_score < self.alignment_threshold:
                self.logger.info(f"Candidate did not meet threshold ({self.alignment_threshold}). Stopping analysis.")
                return self._generate_rejection_report(initial_screening_results)
            
            # STEP 2: IN-DEPTH ANALYSIS
            try:
                in_depth_results = self._run_in_depth_analysis(
                    resume_path, resume_text, job_description, 
                    initial_screening_results["sections"],
                    initial_screening_results["skill_alignment"],
                    github_username, contact_info
                )
            except Exception as e:
                self.logger.error(f"Error in in-depth analysis: {str(e)}")
                raise
            
            # STEP 3: CALCULATE OVERALL QUALITY SCORE
            try:
                quality_score_results = self._calculate_quality_score(
                    initial_screening_results, in_depth_results, resume_path, job_description_path
                )
            except Exception as e:
                self.logger.error(f"Error calculating quality score: {str(e)}")
                raise
            
            # STEP 4: GENERATE COMPREHENSIVE REPORT
            try:
                report_results = self._generate_comprehensive_report(
                    initial_screening_results, in_depth_results, quality_score_results
                )
            except Exception as e:
                self.logger.error(f"Error generating comprehensive report: {str(e)}")
                raise
            
            # Combine all results
            final_results = {
                "status": "success",
                "overall_score": quality_score_results.get("overall_score", 0),
                "grade": quality_score_results.get("grade", "F"),
                "analysis_summary": {
                    "skill_alignment": initial_screening_results["skill_alignment"].get("overall_alignment_score", 0),
                    "project_validation": initial_screening_results["project_validation"].get("overall_score", 0),
                    "formatting": in_depth_results["formatting"].get("total_score", 0),
                    "trustworthiness": in_depth_results["trustworthiness"].get("trust_score", 0),
                    "credibility": in_depth_results["credibility"].get("credibility_score", 0),
                    "online_presence": in_depth_results["link_extraction"].get("summary", {}).get("credibility_score", 0)
                },
                "output_files": report_results["output_files"],
                "recommendations": quality_score_results.get("recommendations", [])
            }
            
            self.logger.info("4-step analysis pipeline completed successfully")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error during pipeline execution: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "error_message": str(e),
                "error_traceback": traceback.format_exc(),
                "overall_score": 0,
                "grade": "F"
            }
    
    def _run_initial_screening(self, resume_path: str, resume_text: str, job_description: str) -> Dict[str, Any]:
        """
        Run Step 1: Initial Screening
        - Document Parsing (already done)
        - Section Detection
        - Project Validation
        - Skill Alignment
        - Visualization
        
        Returns:
            Results of initial screening
        """
        self.logger.info("STEP 1: INITIAL SCREENING")
        
        # 1.1: Document Parsing (already done in run_pipeline)
        self.logger.info("1.1: Document parsing completed")
        
        # 1.2: Section detection
        self.logger.info("1.2: Detecting resume sections...")
        sections = self.section_detector.detect_sections(resume_text)
        
        # 1.3: Project validation
        self.logger.info("1.3: Validating projects...")
        projects_text = self._extract_section_text(sections, [
            "Projects", "Project", "Portfolio", "Technical Projects", "Work Experience", "Experience"
        ])
        
        skills_text = self._extract_section_text(sections, [
            "Skills", "Technical Skills", "Technologies", "Core Competencies", "Expertise", "Proficiencies"
        ])
        
        project_validation_results = self.project_validator.validate_projects(projects_text, skills_text)
        
        # 1.4: Skill alignment analysis
        self.logger.info("1.4: Analyzing skill alignment...")
        skill_alignment_results = self.skill_matcher.compute_alignment(
            resume_text, job_description, sections
        )
        
        # 1.5: Initial Visualization
        self.logger.info("1.5: Generating initial visualizations...")
        self._generate_initial_visualizations(skill_alignment_results, project_validation_results)
        
        # Save results
        self._save_results({
            "sections": sections,
            "skill_alignment": skill_alignment_results,
            "project_validation": project_validation_results
        })
        
        return {
            "sections": sections,
            "skill_alignment": skill_alignment_results,
            "project_validation": project_validation_results
        }
    
    def _run_in_depth_analysis(self, resume_path: str, resume_text: str, job_description: str, 
                             sections: Dict[str, str], skill_alignment_results: Dict[str, Any],
                             github_username: str = None, contact_info: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Run Step 2: In-Depth Analysis
        - Formatting Analysis
        - Link Analysis
        - Trustworthiness Analysis
        - Credibility Verification
        
        Returns:
            Results of in-depth analysis
        """
        self.logger.info("STEP 2: IN-DEPTH ANALYSIS")
        
        # 2.1: Formatting analysis
        self.logger.info("2.1: Analyzing formatting...")
        formatting_results = self.formatting_scorer.analyze_formatting(
            resume_path, sections
        )
        
        # 2.2: Link analysis
        self.logger.info("2.2: Analyzing external links...")
        try:
            link_extraction_results = self.link_extraction_system.process_resume(
                resume_text, contact_info
            )
            total_links = link_extraction_results.get('summary', {}).get('total_links', 0)
            self.logger.info(f"Link extraction completed. Found {total_links} links.")
        except Exception as e:
            self.logger.error(f"Error in link extraction: {str(e)}")
            link_extraction_results = {"summary": {"credibility_score": 0, "platforms_found": [], "platform_counts": {}, "skill_indicators": {}}}
        
        # 2.3: Trustworthiness analysis
        self.logger.info("2.3: Analyzing trustworthiness...")
        trustworthiness_results = self.trustworthiness_detector.analyze_trustworthiness(
            resume_text, sections, 
            skill_alignment_results.get("matched_skills", []),
            github_username
        )
        
        # 2.4: Credibility verification
        self.logger.info("2.4: Verifying credibility...")
        credibility_results = self.credibility_engine.verify_credibility(
            resume_text, sections, contact_info
        )
        
        # Save results
        results = {
            "formatting": formatting_results,
            "link_extraction": link_extraction_results,
            "trustworthiness": trustworthiness_results,
            "credibility": credibility_results
        }
        
        self._save_results(results)
        
        return results
    
    def _calculate_quality_score(self, initial_screening_results: Dict[str, Any], 
                               in_depth_results: Dict[str, Any],
                               resume_path: str, job_description_path: str) -> Dict[str, Any]:
        """
        Run Step 3: Calculate Overall Quality Score
        - Component Breakdown:
          (i) Skill Alignment (30.0% weight)
          (ii) Project Validation (30.0% weight)
          (iii) Resume Formatting (10.0% weight)
          (iv) Content Trustworthiness (10.0% weight)
          (v) Credential Verification (10.0% weight)
          (vi) Online Presence (10.0% weight)
        
        Returns:
            Quality score results
        """
        self.logger.info("STEP 3: CALCULATING OVERALL QUALITY SCORE")
        
        # Extract online presence score from link extraction results
        online_presence_score = 0
        link_extraction_results = in_depth_results.get("link_extraction", {})
        
        if isinstance(link_extraction_results, dict):
            summary = link_extraction_results.get("summary", {})
            if isinstance(summary, dict):
                online_presence_score = summary.get("credibility_score", 0)
        
        # Calculate quality score
        quality_results = self.quality_engine.calculate_final_quality_score(
            initial_screening_results["skill_alignment"],
            initial_screening_results["project_validation"],
            in_depth_results["formatting"],
            in_depth_results["trustworthiness"],
            in_depth_results["credibility"],
            {
                "resume_file": os.path.basename(resume_path),
                "job_description_file": os.path.basename(job_description_path),
                "analysis_version": "1.0",
                "link_extraction": link_extraction_results.get("summary", {}),
                "online_presence_score": online_presence_score
            }
        )
        
        # Generate quality score visualization
        self._generate_quality_visualization(quality_results)
        
        # Save results
        self._save_results({"quality_report": quality_results})
        
        return quality_results
    
    def _generate_comprehensive_report(self, initial_screening_results: Dict[str, Any],
                                     in_depth_results: Dict[str, Any],
                                     quality_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Step 4: Generate Comprehensive Report
        
        Returns:
            Report generation results
        """
        self.logger.info("STEP 4: GENERATING COMPREHENSIVE REPORT")
        
        try:
            # Generate report using explainability layer
            report_path = self.output_dir / "comprehensive_report.html"
            
            report_data = {
                "skill_alignment": initial_screening_results["skill_alignment"],
                "project_validation": initial_screening_results["project_validation"],
                "formatting": in_depth_results["formatting"],
                "trustworthiness": in_depth_results["trustworthiness"],
                "credibility": in_depth_results["credibility"],
                "link_extraction": in_depth_results["link_extraction"],
                "quality_report": quality_results
            }
            
            # Create candidate_data structure
            candidate_data = {
                "name": "Candidate",  # Default name
                "resume_file": quality_results.get("metadata", {}).get("resume_file", "Unknown"),
                "job_description_file": quality_results.get("metadata", {}).get("job_description_file", "Unknown"),
                "analysis_version": quality_results.get("metadata", {}).get("analysis_version", "1.0")
            }
            
            # Create prediction_result structure
            prediction_result_data = {
                "overall_recommendation": quality_results.get("grade", "F"),
                "confidence_score": 0.8,  # Default confidence
                "technical_round_probability": 0.7,  # Default probability
                "culture_fit_probability": 0.7,  # Default probability
                "learning_adaptability_probability": 0.7,  # Default probability
                "top_strengths": quality_results.get("strengths", []),
                "top_weaknesses": quality_results.get("improvement_areas", []),
                "actionable_recommendations": quality_results.get("recommendations", {}).get("immediate_actions", []),
                "feature_importance": {}
            }
            
            # Create a simple object with attribute access for prediction_result
            class PredictionResult:
                def __init__(self, data):
                    for key, value in data.items():
                        setattr(self, key, value)
            
            prediction_result = PredictionResult(prediction_result_data)
            
            # Create platform_scores structure
            platform_scores = {}
            component_scores = quality_results.get("component_scores", {})
            for component, score in component_scores.items():
                platform_scores[component] = {
                    "normalized_score": score,
                    "raw_score": score,
                    "weight": quality_results.get("score_breakdown", {}).get(component, {}).get("weight", 0.1),
                    "confidence": 0.8  # Default confidence
                }
            
            self.explainability_layer.generate_comprehensive_report(
                candidate_data, prediction_result, platform_scores
            )
            
            # Create a summary text file
            summary_path = self.output_dir / "analysis_summary.txt"
            self._generate_summary_file(report_data, summary_path)
            
            return {
                "status": "success",
                "output_files": {
                    "comprehensive_report": str(report_path),
                    "summary": str(summary_path),
                    "quality_report": str(self.output_dir / "comprehensive_quality_report.json"),
                    "skill_alignment": str(self.output_dir / "skill_alignment.json"),
                    "project_validation": str(self.output_dir / "project_validation.json"),
                    "formatting": str(self.output_dir / "formatting.json"),
                    "trustworthiness": str(self.output_dir / "trustworthiness.json"),
                    "credibility": str(self.output_dir / "credibility.json"),
                    "link_extraction": str(self.output_dir / "link_extraction.json")
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            return {
                "status": "error",
                "error_message": str(e),
                "output_files": {}
            }
    
    def _generate_rejection_report(self, initial_screening_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a rejection report for candidates who don't meet the threshold."""
        skill_alignment = initial_screening_results.get("skill_alignment", {})
        
        # Ensure missing_skills is a list
        missing_skills = []
        if isinstance(skill_alignment.get("missing_skills"), list):
            missing_skills = skill_alignment.get("missing_skills", [])
        elif isinstance(skill_alignment.get("missing_skills"), dict):
            missing_skills = list(skill_alignment.get("missing_skills", {}).keys())
        
        # Generate report
        report_path = self.output_dir / "rejection_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Resume Analysis - Rejection Report\n\n")
            f.write("## Analysis Results\n\n")
            f.write(f"Skill Alignment Score: {skill_alignment.get('overall_alignment_score', 0):.2f}%\n")
            f.write(f"Required Threshold: {self.alignment_threshold}%\n\n")
            
            # Add missing skills
            if missing_skills:
                f.write("#### Missing Critical Skills\n\n")
                for skill in missing_skills:
                    f.write(f"- {skill}\n")
        
        return {
            "status": "rejected",
            "reason": f"Skill alignment score ({skill_alignment.get('overall_alignment_score', 0):.2f}%) below threshold ({self.alignment_threshold}%)",
            "skill_alignment_score": skill_alignment.get('overall_alignment_score', 0),
            "missing_skills": missing_skills,
            "output_files": {
                "rejection_report": str(report_path),
                "skill_alignment": str(self.output_dir / "skill_alignment.json"),
                "project_validation": str(self.output_dir / "project_validation.json")
            }
        }
    
    def _extract_section_text(self, sections: Dict[str, str], section_names: List[str]) -> str:
        """
        Extract text from a section, trying multiple possible section names
        
        Args:
            sections: Dictionary of sections
            section_names: List of possible section names to try
            
        Returns:
            Text from the section, or empty string if not found
        """
        # First try exact matches
        for name in section_names:
            if name in sections:
                return sections[name]
        
        # If no exact match, try case-insensitive partial matches
        for key in sections.keys():
            if any(name.lower() in key.lower() for name in section_names):
                return sections[key]
        
        return ""
    
    def _generate_initial_visualizations(self, skill_alignment_results: Dict[str, Any], 
                                        project_validation_results: Dict[str, Any]):
        """
        Generate visualizations for initial screening results
        """
        # Generate skill alignment visualization
        self.visualizer.visualize_skill_alignment(
            skill_alignment_results,
            output_path=str(self.output_dir / "skill_alignment.png")
        )
        
        # Generate project validation visualization
        self.visualizer.visualize_project_validation(
            project_validation_results,
            output_path=str(self.output_dir / "project_validation.png")
        )
    
    def _generate_quality_visualization(self, quality_results: Dict[str, Any]):
        """
        Generate visualization for quality score results
        """
        # Generate quality score visualization
        self.visualizer.visualize_quality_scores(
            quality_results,
            output_path=str(self.output_dir / "comprehensive_quality.png")
        )
    
    def _generate_summary_file(self, report_data: Dict[str, Any], output_path: Path):
        """
        Generate a summary text file with key metrics
        """
        with open(output_path, 'w') as f:
            f.write("Resume Intelligence System - Analysis Summary\n")
            f.write("==============================================\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall score and grade
            quality_report = report_data.get("quality_report", {})
            overall_score = quality_report.get("overall_score", 0)
            grade = quality_report.get("grade", "F")
            
            f.write(f"Overall Quality Score: {overall_score:.2f}%\n")
            f.write(f"Grade: {grade}\n\n")
            
            # Component scores
            f.write("Component Scores:\n")
            f.write("----------------\n")
            f.write(f"Skill Alignment: {report_data.get('skill_alignment', {}).get('overall_alignment_score', 0):.2f}%\n")
            f.write(f"Project Validation: {report_data.get('project_validation', {}).get('overall_score', 0):.2f}%\n")
            f.write(f"Resume Formatting: {report_data.get('formatting', {}).get('total_score', 0):.2f}%\n")
            f.write(f"Content Trustworthiness: {report_data.get('trustworthiness', {}).get('trust_score', 0):.2f}%\n")
            f.write(f"Credential Verification: {report_data.get('credibility', {}).get('credibility_score', 0):.2f}%\n")
            
            # Online presence
            link_extraction = report_data.get("link_extraction", {})
            if isinstance(link_extraction, dict):
                summary = link_extraction.get("summary", {})
                if isinstance(summary, dict):
                    online_score = summary.get("credibility_score", 0)
                    f.write(f"Online Presence: {online_score:.2f}%\n")
            
            # Key strengths and weaknesses
            f.write("\nKey Strengths:\n")
            f.write("-------------\n")
            strengths = quality_report.get("strengths", [])
            for strength in strengths[:5]:  # Top 5 strengths
                f.write(f"- {strength}\n")
            
            f.write("\nAreas for Improvement:\n")
            f.write("---------------------\n")
            weaknesses = quality_report.get("weaknesses", [])
            for weakness in weaknesses[:5]:  # Top 5 weaknesses
                f.write(f"- {weakness}\n")
            
            # Recommendations
            f.write("\nRecommendations:\n")
            f.write("----------------\n")
            recommendations = quality_report.get("recommendations", [])
            for recommendation in recommendations:
                f.write(f"- {recommendation}\n")
            
            # Output files
            f.write("\nOutput Files:\n")
            f.write("-------------\n")
            f.write(f"Comprehensive Report: {self.output_dir / 'comprehensive_report.html'}\n")
            f.write(f"Quality Report: {self.output_dir / 'comprehensive_quality_report.json'}\n")
            f.write(f"Skill Alignment: {self.output_dir / 'skill_alignment.json'}\n")
            f.write(f"Project Validation: {self.output_dir / 'project_validation.json'}\n")
            f.write(f"Link Extraction: {self.output_dir / 'link_extraction.json'}\n")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save analysis results to files"""
        # Save individual component results
        for component, data in results.items():
            output_path = self.output_dir / f"{component}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                # Convert NumPy types to native Python types for JSON serialization
                json_data = self._convert_numpy_types(data)
                json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    def _convert_numpy_types(self, obj):
        """Convert NumPy types to native Python types for JSON serialization"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_numpy_types(obj.tolist())
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def analyze_skill_alignment(self, candidate_skills: List[str], job_requirements: List[str]) -> Dict[str, Any]:
        """Analyze skill alignment between candidate skills and job requirements."""
        return self.skill_matcher.compute_alignment(candidate_skills, job_requirements)

    def _generate_assessment(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Generate assessment based on resume and job description."""
        # ... existing code ...
        assessment = {
            "skill_match": self.analyze_skill_alignment(candidate_skills, job_requirements),
            "experience_match": self.analyze_experience(resume_text, job_description),
            "education_match": self.analyze_education(resume_text, job_description),
            "overall_assessment": self.analyze_overall(resume_text, job_description),
            "next_steps": ["Review skill gaps", "Enhance relevant experience", "Update education details"]  # Example next steps
        }
        return assessment


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Resume Intelligence System Pipeline")
    parser.add_argument("--resume", type=str, required=True, help="Path to resume file")
    parser.add_argument("--job-description", type=str, required=True, help="Path to job description file")
    parser.add_argument("--github-username", type=str, help="GitHub username for verification")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = ResumePipeline(output_dir=args.output_dir)
    
    # Run pipeline
    contact_info = {}
    if args.github_username:
        contact_info["github"] = args.github_username
    
    results = pipeline.run_pipeline(
        resume_path=args.resume,
        job_description_path=args.job_description,
        github_username=args.github_username,
        contact_info=contact_info
    )
    
    # Print summary
    print("\nResume Analysis Complete")
    print("========================\n")
    print(f"Status: {results['status']}")
    
    if results['status'] == 'success':
        print(f"Overall Score: {results['overall_score']:.2f}%")
        print(f"Grade: {results['grade']}")
        print("\nComponent Scores:")
        for component, score in results['analysis_summary'].items():
            print(f"  {component.replace('_', ' ').title()}: {score:.2f}%")
        
        print("\nOutput Files:")
        for name, path in results['output_files'].items():
            print(f"  {name.replace('_', ' ').title()}: {path}")
    else:
        print(f"Error: {results.get('error_message', 'Unknown error')}")


if __name__ == "__main__":
    main()