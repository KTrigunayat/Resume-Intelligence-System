#!/usr/bin/env python3
"""
Resume Intelligence System - Analysis Runner

This script provides a command-line interface to run the Resume Intelligence System pipeline.
It allows users to analyze resumes against job descriptions and generate comprehensive reports.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the pipeline
from resume_pipeline import ResumePipeline


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Resume Intelligence System - Analyze resumes against job descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
        python run_analysis.py --resume path/to/resume.pdf --job-description path/to/job.txt
        python run_analysis.py --resumes path/to/resume1.pdf path/to/resume2.pdf --job-description path/to/job.txt --github-username johndoe
        """
    )
    
    # Resume analysis arguments
    resume_group = parser.add_mutually_exclusive_group(required=True)
    resume_group.add_argument("--resume", type=str,
                            help="Path to a single resume file (PDF, DOCX, or TXT)")
    resume_group.add_argument("--resumes", type=str, nargs="+",
                            help="Paths to multiple resume files")
    
    parser.add_argument("--job-description", type=str, required=True,
                       help="Path to the job description file")
    parser.add_argument("--github-username", type=str,
                       help="GitHub username for additional verification")
    parser.add_argument("--linkedin-url", type=str,
                       help="LinkedIn URL for additional verification")
    parser.add_argument("--output-dir", type=str, default="output",
                       help="Directory to store analysis results")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Maximum number of parallel workers")
    
    # If no arguments are provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    return parser.parse_args()


def validate_files(args):
    """Validate that the specified files exist"""
    # Validate job description
    job_desc_path = Path(args.job_description)
    if not job_desc_path.exists():
        print(f"Error: Job description file not found: {job_desc_path}")
        sys.exit(1)
    
    # Validate resume(s)
    resume_paths = [args.resume] if args.resume else args.resumes
    for resume_path in resume_paths:
        path = Path(resume_path)
        if not path.exists():
            print(f"Error: Resume file not found: {path}")
            sys.exit(1)


def analyze_single_resume(resume_path: str, job_description_path: str, output_dir: str,
                         github_username: str = None, linkedin_url: str = None) -> dict:
    """Analyze a single resume"""
    # Create output directory for this resume
    resume_name = Path(resume_path).stem
    resume_output_dir = os.path.join(output_dir, resume_name)
    os.makedirs(resume_output_dir, exist_ok=True)
    
    # Prepare contact info
    contact_info = {}
    if github_username:
        contact_info["github"] = github_username
    if linkedin_url:
        contact_info["linkedin"] = linkedin_url
    
    # Create and run pipeline
    pipeline = ResumePipeline(output_dir=resume_output_dir)
    
    return pipeline.run_pipeline(
        resume_path=resume_path,
        job_description_path=job_description_path,
        github_username=github_username,
        contact_info=contact_info
    )


def run_analysis(args):
    """Run analysis on one or more resumes"""
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get resume paths
    resume_paths = [args.resume] if args.resume else args.resumes
    
    print(f"\nAnalyzing {len(resume_paths)} resume(s)")
    print(f"Against job description: {args.job_description}")
    print(f"Output directory: {args.output_dir}\n")
    
    results = {}
    
    if len(resume_paths) == 1:
        # Single resume analysis
        results[resume_paths[0]] = analyze_single_resume(
            resume_paths[0],
            args.job_description,
            str(output_dir),
            args.github_username,
            args.linkedin_url
        )
    else:
        # Parallel processing for multiple resumes
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all jobs
            future_to_resume = {
                executor.submit(
                    analyze_single_resume,
                    resume_path,
                    args.job_description,
                    str(output_dir),
                    args.github_username,
                    args.linkedin_url
                ): resume_path
                for resume_path in resume_paths
            }
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_resume), total=len(resume_paths),
                             desc="Analyzing resumes"):
                resume_path = future_to_resume[future]
                try:
                    results[resume_path] = future.result()
                except Exception as e:
                    print(f"Error processing {resume_path}: {str(e)}")
                    results[resume_path] = {
                        "status": "error",
                        "error_message": str(e),
                        "overall_score": 0,
                        "grade": "F"
                    }
    
    # Print summary
    print("\nAnalysis Complete")
    print("================\n")
    
    # Sort results by overall score
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get("overall_score", 0),
        reverse=True
    )
    
    for resume_path, result in sorted_results:
        print(f"\nResume: {resume_path}")
        print("-" * (len(resume_path) + 8))
        
        if result["status"] == "success":
            print(f"Overall Score: {result['overall_score']:.2f}%")
            print(f"Grade: {result['grade']}")
            
            print("\nComponent Scores:")
            for component, score in result["analysis_summary"].items():
                print(f"  {component.replace('_', ' ').title()}: {score:.2f}%")
            
            print("\nOutput Files:")
            for name, path in result["output_files"].items():
                print(f"  {name.replace('_', ' ').title()}: {path}")
        
        elif result["status"] == "rejected":
            print(f"Result: REJECTED")
            print(f"Reason: {result.get('reason', 'Unknown')}")
            print(f"Skill Alignment Score: {result.get('skill_alignment_score', 0):.2f}%")
            
            if "missing_skills" in result and result["missing_skills"]:
                print("\nMissing Critical Skills:")
                for skill in result["missing_skills"]:
                    print(f"  - {skill}")
            
            print("\nOutput Files:")
            for name, path in result["output_files"].items():
                print(f"  {name.replace('_', ' ').title()}: {path}")
        
        else:  # Error
            print(f"Error: {result.get('error_message', 'Unknown error')}")
    
    return results


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Validate files
    validate_files(args)
    
    # Run analysis
    run_analysis(args)


if __name__ == "__main__":
    main()