import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import system components
from resume_intelligence.github_skill_predictor import GitHubTechnicalSkillPredictor
from resume_intelligence.multi_platform_scorer import MultiPlatformWeightedScoreEngine
from resume_intelligence.ai_prediction_engine import AIPredictionEngine
from resume_intelligence.explainability_layer import ExplainabilityReportingLayer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Resume Intelligence System API",
    description="API for analyzing resumes and predicting candidate outcomes",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define data models
class GitHubProfile(BaseModel):
    username: str
    access_token: Optional[str] = None

class LeetCodeProfile(BaseModel):
    username: str

class KaggleProfile(BaseModel):
    username: str
    access_token: Optional[str] = None

class DesignProfile(BaseModel):
    platform: str = Field(..., description="Design platform (e.g., 'figma', 'dribbble')")
    username: str
    portfolio_url: Optional[str] = None

class LinkedInProfile(BaseModel):
    profile_url: str
    public_id: Optional[str] = None

class CandidateProfiles(BaseModel):
    github: Optional[GitHubProfile] = None
    leetcode: Optional[LeetCodeProfile] = None
    kaggle: Optional[KaggleProfile] = None
    design: Optional[DesignProfile] = None
    linkedin: Optional[LinkedInProfile] = None

class AnalysisRequest(BaseModel):
    candidate_id: Optional[str] = None
    candidate_name: Optional[str] = None
    profiles: CandidateProfiles
    job_description: Optional[str] = None
    weight_config: Optional[Dict[str, float]] = None

class AnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    message: str
    timestamp: datetime

class AnalysisResult(BaseModel):
    analysis_id: str
    candidate_id: Optional[str] = None
    candidate_name: Optional[str] = None
    overall_score: float
    grade: str
    platform_scores: Dict[str, Dict[str, Any]]
    predictions: Dict[str, float]
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    report_url: Optional[str] = None
    timestamp: datetime

# In-memory storage for analysis results (replace with database in production)
analysis_storage: Dict[str, Dict[str, Any]] = {}

# Create output directory for reports
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize components
github_predictor = GitHubTechnicalSkillPredictor()
multi_platform_scorer = MultiPlatformWeightedScoreEngine()
ai_prediction_engine = AIPredictionEngine()
explainability_layer = ExplainabilityReportingLayer(output_dir=str(OUTPUT_DIR))

@app.get("/")
async def root():
    """Root endpoint to check if API is running"""
    return {"message": "Resume Intelligence System API is running"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_candidate(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Submit a candidate for analysis"""
    try:
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Store initial analysis state
        analysis_storage[analysis_id] = {
            "status": "processing",
            "request": request.dict(),
            "timestamp": datetime.now(),
            "result": None
        }
        
        # Process analysis in background
        background_tasks.add_task(process_analysis, analysis_id, request)
        
        return AnalysisResponse(
            analysis_id=analysis_id,
            status="processing",
            message="Analysis submitted successfully and is being processed",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error submitting analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error submitting analysis: {str(e)}")

@app.get("/analysis/{analysis_id}", response_model=Union[AnalysisResponse, AnalysisResult])
async def get_analysis_status(analysis_id: str):
    """Get the status or result of an analysis"""
    if analysis_id not in analysis_storage:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = analysis_storage[analysis_id]
    
    if analysis["status"] == "processing":
        return AnalysisResponse(
            analysis_id=analysis_id,
            status="processing",
            message="Analysis is still being processed",
            timestamp=datetime.now()
        )
    elif analysis["status"] == "completed":
        return analysis["result"]
    else:  # error
        return AnalysisResponse(
            analysis_id=analysis_id,
            status="error",
            message=analysis.get("error_message", "Unknown error occurred"),
            timestamp=datetime.now()
        )

@app.get("/report/{analysis_id}")
async def get_analysis_report(analysis_id: str):
    """Get the HTML report for an analysis"""
    if analysis_id not in analysis_storage:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = analysis_storage[analysis_id]
    
    if analysis["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis is not completed yet")
    
    report_path = analysis.get("report_path")
    if not report_path or not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(report_path)

@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """Upload a resume file for processing"""
    try:
        # Save the uploaded file
        file_path = OUTPUT_DIR / f"resume_{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return {"message": "Resume uploaded successfully", "file_path": str(file_path)}
    
    except Exception as e:
        logger.error(f"Error uploading resume: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading resume: {str(e)}")

@app.get("/platforms")
async def get_supported_platforms():
    """Get list of supported platforms for analysis"""
    return {
        "platforms": [
            {"id": "github", "name": "GitHub", "description": "Code repositories and contributions"},
            {"id": "leetcode", "name": "LeetCode", "description": "Algorithmic problem solving"},
            {"id": "kaggle", "name": "Kaggle", "description": "Data science competitions"},
            {"id": "figma", "name": "Figma", "description": "UI/UX design"},
            {"id": "dribbble", "name": "Dribbble", "description": "Design portfolio"},
            {"id": "linkedin", "name": "LinkedIn", "description": "Professional profile"}
        ]
    }

@app.post("/weight-config")
async def update_weight_config(config: Dict[str, float] = Body(...)):
    """Update the default weight configuration for scoring"""
    try:
        # Validate weights sum to 1.0
        total_weight = sum(config.values())
        if not (0.99 <= total_weight <= 1.01):  # Allow small floating point errors
            raise HTTPException(status_code=400, detail="Weights must sum to 1.0")
        
        # Update multi-platform scorer config
        multi_platform_scorer.update_weight_config(config)
        
        return {"message": "Weight configuration updated successfully", "config": config}
    
    except Exception as e:
        if not isinstance(e, HTTPException):
            logger.error(f"Error updating weight config: {e}")
            raise HTTPException(status_code=500, detail=f"Error updating weight config: {str(e)}")
        raise e

async def process_analysis(analysis_id: str, request: AnalysisRequest):
    """Process the analysis in the background"""
    try:
        # Extract profiles
        github_profile = request.profiles.github
        leetcode_profile = request.profiles.leetcode
        kaggle_profile = request.profiles.kaggle
        design_profile = request.profiles.design
        linkedin_profile = request.profiles.linkedin
        
        # Initialize platform scores
        platform_scores = {}
        
        # Process GitHub profile if available
        if github_profile:
            github_result = github_predictor.analyze_github_profile(
                github_profile.username,
                access_token=github_profile.access_token
            )
            platform_scores["github"] = {
                "normalized_score": github_result.technical_skill_score,
                "raw_score": github_result.technical_skill_score,
                "grade": github_result.grade,
                "metrics": {
                    "code_quality": github_result.code_quality_score,
                    "complexity": github_result.complexity_score,
                    "test_coverage": github_result.test_coverage_score,
                    "originality": github_result.originality_percentage
                }
            }
        
        # Process other platforms (placeholder for now)
        # In a real implementation, these would call their respective analyzers
        if leetcode_profile:
            platform_scores["leetcode"] = {
                "normalized_score": 70.0,  # Placeholder
                "raw_score": 70.0,
                "metrics": {
                    "problems_solved": 150,
                    "contest_rating": 1800
                }
            }
        
        if kaggle_profile:
            platform_scores["kaggle"] = {
                "normalized_score": 65.0,  # Placeholder
                "raw_score": 65.0,
                "metrics": {
                    "competitions": 5,
                    "medals": 2
                }
            }
        
        if design_profile:
            platform_scores["design"] = {
                "normalized_score": 75.0,  # Placeholder
                "raw_score": 75.0,
                "metrics": {
                    "projects": 10,
                    "likes": 250
                }
            }
        
        if linkedin_profile:
            platform_scores["linkedin"] = {
                "normalized_score": 80.0,  # Placeholder
                "raw_score": 80.0,
                "metrics": {
                    "connections": 500,
                    "endorsements": 50
                }
            }
        
        # Add resume score if available (placeholder)
        platform_scores["resume"] = {
            "normalized_score": 60.0,  # Placeholder
            "raw_score": 60.0,
            "metrics": {
                "formatting": 70.0,
                "content": 65.0,
                "skills_match": 45.0
            }
        }
        
        # Apply weight configuration if provided
        if request.weight_config:
            multi_platform_scorer.update_weight_config(request.weight_config)
        
        # Calculate final score
        score_result = multi_platform_scorer.calculate_final_score(platform_scores)
        
        # Generate predictions
        prediction_result = ai_prediction_engine.predict_candidate_outcomes(
            platform_scores=platform_scores,
            final_score=score_result.final_score,
            job_description=request.job_description
        )
        
        # Generate report
        candidate_data = {
            "name": request.candidate_name or "Candidate",
            "id": request.candidate_id or analysis_id
        }
        
        report_path = explainability_layer.generate_comprehensive_report(
            candidate_data=candidate_data,
            prediction_result=prediction_result,
            platform_scores=platform_scores
        )
        
        # Create analysis result
        result = AnalysisResult(
            analysis_id=analysis_id,
            candidate_id=request.candidate_id,
            candidate_name=request.candidate_name,
            overall_score=score_result.final_score,
            grade=score_result.grade,
            platform_scores=platform_scores,
            predictions={
                "technical_round": prediction_result.technical_round_probability,
                "culture_fit": prediction_result.culture_fit_probability,
                "learning_adaptability": prediction_result.learning_adaptability_probability
            },
            strengths=prediction_result.top_strengths,
            weaknesses=prediction_result.top_weaknesses,
            recommendations=prediction_result.actionable_recommendations,
            report_url=f"/report/{analysis_id}",
            timestamp=datetime.now()
        )
        
        # Update analysis storage
        analysis_storage[analysis_id] = {
            "status": "completed",
            "request": request.dict(),
            "timestamp": datetime.now(),
            "result": result,
            "report_path": report_path
        }
        
    except Exception as e:
        logger.error(f"Error processing analysis {analysis_id}: {e}")
        analysis_storage[analysis_id] = {
            "status": "error",
            "request": request.dict(),
            "timestamp": datetime.now(),
            "error_message": str(e)
        }

# Run with: uvicorn resume_intelligence.api_layer:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)