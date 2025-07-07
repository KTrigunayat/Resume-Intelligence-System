import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from resume_intelligence.skill_matcher import SkillMatcher

def test_skill_matcher():
    """Test the skill matcher functionality"""
    logger.info("Initializing skill matcher...")
    matcher = SkillMatcher()
    
    # Test resume text
    resume_text = """
    Professional Summary:
    Experienced software developer with expertise in Python, JavaScript, and React.
    Strong background in machine learning and data analysis.
    Proficient in SQL and database management.
    """
    
    # Test job description
    job_description = """
    We are looking for a developer skilled in Python, React, AWS, and Machine Learning. Experience with SQL and data analysis is a plus.
    """
    
    logger.info("Testing skill alignment...")
    results = matcher.compute_alignment(resume_text, job_description, sections={})
    logger.info(f"Alignment results: {results}")
    print("Alignment Results:\n", results)

if __name__ == "__main__":
    test_skill_matcher() 