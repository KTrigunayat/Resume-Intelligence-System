"""Resume Intelligence System - A comprehensive system for analyzing resumes and predicting candidate outcomes."""

__version__ = "1.0.0"
__author__ = "Resume Intelligence Team"

# Import main components for easier access
from resume_intelligence.github_skill_predictor import GitHubTechnicalSkillPredictor
from resume_intelligence.multi_platform_scorer import MultiPlatformWeightedScoreEngine, WeightedScoreConfig
from resume_intelligence.ai_prediction_engine import AIPredictionEngine, PredictionResult
from resume_intelligence.explainability_layer import ExplainabilityReportingLayer
from resume_intelligence.workflow_orchestration import WorkflowOrchestrator
from . import candidate_db

# Define what's available when importing *
__all__ = [
    'GitHubTechnicalSkillPredictor',
    'MultiPlatformWeightedScoreEngine',
    'WeightedScoreConfig',
    'AIPredictionEngine',
    'PredictionResult',
    'ExplainabilityReportingLayer',
    'WorkflowOrchestrator',
]