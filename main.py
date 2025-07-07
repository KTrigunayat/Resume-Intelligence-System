import os
import sys
import logging
import argparse
from pathlib import Path
import uvicorn
import threading
import time
from dotenv import load_dotenv
from . import candidate_db

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

# Import system components
from resume_intelligence.api_layer import app as api_app
from resume_intelligence.workflow_orchestration import WorkflowOrchestrator


def start_api_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the API server"""
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run("resume_intelligence.api_layer:app", host=host, port=port, reload=reload)


def start_workflow_orchestrator(output_dir: str = "./workflow_output"):
    """Start the workflow orchestrator"""
    logger.info("Starting workflow orchestrator")
    orchestrator = WorkflowOrchestrator(output_dir=output_dir)
    orchestrator.start()
    return orchestrator


def extract_pros_cons(analysis):
    """Extracts pros and cons from the analysis JSON."""
    pros = []
    cons = []
    # Example logic, can be improved based on your analysis structure
    if analysis.get('skill_alignment', 0) > 0.8:
        pros.append('Strong skill alignment')
    elif analysis.get('skill_alignment', 0) < 0.5:
        cons.append('Weak skill alignment')
    if analysis.get('formatting_score', 0) > 0.8:
        pros.append('Well-formatted resume')
    elif analysis.get('formatting_score', 0) < 0.5:
        cons.append('Poor formatting')
    if analysis.get('credibility_score', 0) > 0.8:
        pros.append('High credibility')
    elif analysis.get('credibility_score', 0) < 0.5:
        cons.append('Low credibility')
    # Add more rules as needed
    return pros, cons


def display_top_candidates(job_id):
    top_candidates = candidate_db.get_top_candidates(job_id, top_n=10)
    print(f"Top 10 Candidates for Job: {job_id}\n")
    for idx, cand in enumerate(top_candidates, 1):
        name = cand['candidate_name']
        score = cand['score']
        analysis = cand['analysis']
        pros, cons = extract_pros_cons(analysis)
        print(f"{idx}. {name} (Score: {score:.2f})")
        print(f"   Pros: {', '.join(pros) if pros else 'None'}")
        print(f"   Cons: {', '.join(cons) if cons else 'None'}\n")


def main():
    """Main entry point for the Resume Intelligence System"""
    parser = argparse.ArgumentParser(description="Resume Intelligence System")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # API server command
    api_parser = subparsers.add_parser("api", help="Start the API server")
    api_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # Workflow orchestrator command
    workflow_parser = subparsers.add_parser("workflow", help="Start the workflow orchestrator")
    workflow_parser.add_argument("--output-dir", type=str, default="./workflow_output", 
                               help="Output directory for workflow results")
    
    # Full system command
    full_parser = subparsers.add_parser("full", help="Start the full system (API + workflow)")
    full_parser.add_argument("--api-host", type=str, default="0.0.0.0", help="API host to bind to")
    full_parser.add_argument("--api-port", type=int, default=8000, help="API port to bind to")
    full_parser.add_argument("--workflow-output-dir", type=str, default="./workflow_output", 
                           help="Output directory for workflow results")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the appropriate command
    if args.command == "api":
        start_api_server(host=args.host, port=args.port, reload=args.reload)
    
    elif args.command == "workflow":
        orchestrator = start_workflow_orchestrator(output_dir=args.output_dir)
        
        try:
            # Keep running until interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping workflow orchestrator")
            orchestrator.stop()
    
    elif args.command == "full":
        # Start workflow orchestrator in a separate thread
        orchestrator = start_workflow_orchestrator(output_dir=args.workflow_output_dir)
        
        # Start API server in main thread
        try:
            start_api_server(host=args.api_host, port=args.api_port)
        except KeyboardInterrupt:
            logger.info("Stopping system")
            orchestrator.stop()
    
    else:
        parser.print_help()

    # Assume job_id is the job description filename (without extension)
    job_desc_path = 'job_description.txt'  # or from args
    job_id = os.path.splitext(os.path.basename(job_desc_path))[0]
    candidate_db.init_db()
    # ... existing code ...
    resume_folder = 'resumes'  # or wherever resumes are uploaded
    for resume_file in os.listdir(resume_folder):
        if not resume_file.lower().endswith('.txt'):
            continue
        resume_path = os.path.join(resume_folder, resume_file)
        # ... existing code to analyze resume ...
        # Assume analysis_result is the output dict, and candidate_name is parsed from resume_file
        candidate_name = os.path.splitext(resume_file)[0]
        score = analysis_result.get('overall_score', 0)
        candidate_db.insert_candidate_analysis(job_id, candidate_name, score, analysis_result)
    display_top_candidates(job_id)


if __name__ == "__main__":
    main()