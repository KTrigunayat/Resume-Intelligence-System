# Resume Intelligence System

An advanced AI-powered system for resume analysis and job matching. This system combines traditional NLP techniques with state-of-the-art LLM capabilities to provide comprehensive resume analysis, skill matching, and interview preparation.
Download the ZIP File to Run this on your own System avoid cloning the repo.
## Features

- **Resume Analysis**: Deep analysis of resume content, formatting, and structure
- **Skill Matching**: Advanced skill alignment with job requirements
- **Project Validation**: Verification of project claims and technical skills
- **Credibility Check**: Verification of credentials and online presence
- **Interview Preparation**: Generation of targeted HR questions
- **Comprehensive Reports**: Detailed analysis with actionable recommendations

## Prerequisites

- Python 3.9 or higher
- OpenAI API key (for enhanced analysis)
- Git

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd resume-intelligence
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

5. Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Basic Usage

```bash
python run_analysis.py --resume path/to/resume.pdf --job-description path/to/job.txt
```

### Advanced Usage

```bash
python ai_agent.py --resumes path/to/resume1.pdf path/to/resume2.pdf --jd path/to/job.txt --github_usernames user1 user2 --output_dir output
```

### Command Line Arguments

- `--resume` or `--resumes`: Path to resume file(s) (PDF, DOCX, or TXT)
- `--job-description` or `--jd`: Path to job description file
- `--github-username` or `--github_usernames`: GitHub username(s) for verification
- `--output-dir`: Directory for analysis results (default: "output")
- `--verbose`: Enable detailed logging

## Project Structure

```
resume-intelligence/
├── ai_agent.py              # Main AI agent for enhanced analysis
├── resume_pipeline.py       # Core analysis pipeline
├── run_analysis.py          # CLI interface
├── requirements.txt         # Project dependencies
├── .env                     # Environment variables
├── output/                  # Analysis results
└── resume_intelligence/     # Core package
    ├── __init__.py
    ├── section_detector.py
    ├── skill_matcher.py
    ├── project_validator.py
    ├── formatting_scorer.py
    ├── trustworthiness_detector.py
    ├── credibility_engine.py
    ├── quality_score.py
    ├── visualizer.py
    └── utils/
```

## Output

The system generates several output files in the specified output directory:

- `analysis_[resume_name].json`: Comprehensive analysis report
- `skill_alignment.png`: Skill matching visualization
- `project_validation.png`: Project validation results
- `comprehensive_quality.png`: Overall quality assessment
- `analysis.log`: Detailed processing log

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT-3.5 Turbo API
- spaCy for NLP capabilities
- Sentence Transformers for semantic matching
- All other open-source contributors

## Support

For support, please open an issue in the GitHub repository or contact the maintainers. 
