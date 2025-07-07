import ast
import os
import json
import subprocess
import tempfile
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import requests
import numpy as np
from git import Repo
import radon.complexity as radon_cc
import radon.metrics as radon_metrics
from radon.visitors import ComplexityVisitor
import coverage


@dataclass
class CodeMetrics:
    """Data class to store code analysis metrics"""
    function_count: int
    class_count: int
    cyclomatic_complexity: float
    lines_of_code: int
    test_coverage: float
    maintainability_index: float
    halstead_volume: float
    originality_score: float


@dataclass
class GitHubSkillScore:
    """Data class to store final GitHub skill assessment"""
    technical_skill_score: float  # 0-100
    code_quality_grade: str  # A-F
    originality_percentage: float  # 0-100
    detailed_metrics: CodeMetrics
    repository_analysis: Dict[str, Any]


class GitHubTechnicalSkillPredictor:
    """
    Deep-dive analysis of a candidate's GitHub footprint to quantify coding ability,
    code quality, and originality using AST parsing, complexity analysis, and plagiarism detection.
    """
    
    def __init__(self, github_token: Optional[str] = None, plagiarism_api_key: Optional[str] = None):
        """
        Initialize the GitHub Technical Skill Predictor.
        
        Args:
            github_token: GitHub API token for accessing repositories
            plagiarism_api_key: API key for plagiarism detection service
        """
        self.github_token = github_token
        self.plagiarism_api_key = plagiarism_api_key
        self.headers = {'Authorization': f'token {github_token}'} if github_token else {}
        
        # Scoring weights for different metrics
        self.weights = {
            'complexity': 0.25,
            'test_coverage': 0.20,
            'code_structure': 0.20,
            'maintainability': 0.15,
            'originality': 0.20
        }
    
    def analyze_github_profile(self, username: str) -> GitHubSkillScore:
        """
        Perform comprehensive analysis of a GitHub profile.
        
        Args:
            username: GitHub username to analyze
            
        Returns:
            GitHubSkillScore with detailed technical assessment
        """
        try:
            # Get user repositories
            repos = self._get_user_repositories(username)
            
            # Filter for significant repositories (exclude forks, focus on original work)
            significant_repos = self._filter_significant_repositories(repos)
            
            # Analyze each repository
            repo_analyses = []
            total_metrics = CodeMetrics(0, 0, 0.0, 0, 0.0, 0.0, 0.0, 0.0)
            
            for repo in significant_repos[:10]:  # Analyze top 10 repositories
                try:
                    analysis = self._analyze_repository(username, repo['name'])
                    repo_analyses.append(analysis)
                    total_metrics = self._aggregate_metrics(total_metrics, analysis['metrics'])
                except Exception as e:
                    print(f"Error analyzing repository {repo['name']}: {e}")
                    continue
            
            # Calculate final scores
            technical_score = self._calculate_technical_score(total_metrics)
            quality_grade = self._calculate_quality_grade(technical_score)
            
            return GitHubSkillScore(
                technical_skill_score=technical_score,
                code_quality_grade=quality_grade,
                originality_percentage=total_metrics.originality_score,
                detailed_metrics=total_metrics,
                repository_analysis={
                    'total_repositories_analyzed': len(repo_analyses),
                    'repository_details': repo_analyses,
                    'languages_used': self._extract_languages(repos),
                    'contribution_patterns': self._analyze_contribution_patterns(username)
                }
            )
            
        except Exception as e:
            print(f"Error analyzing GitHub profile {username}: {e}")
            return self._create_default_score()
    
    def _get_user_repositories(self, username: str) -> List[Dict]:
        """
        Fetch user repositories from GitHub API.
        
        Args:
            username: GitHub username
            
        Returns:
            List of repository information
        """
        url = f"https://api.github.com/users/{username}/repos"
        params = {'per_page': 100, 'sort': 'updated', 'direction': 'desc'}
        
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching repositories: {response.status_code}")
            return []
    
    def _filter_significant_repositories(self, repos: List[Dict]) -> List[Dict]:
        """
        Filter repositories to focus on significant, original work.
        
        Args:
            repos: List of repository information
            
        Returns:
            Filtered list of significant repositories
        """
        significant = []
        
        for repo in repos:
            # Skip forks and very small repositories
            if repo.get('fork', False):
                continue
            
            # Consider repositories with meaningful size and activity
            if (repo.get('size', 0) > 10 and  # At least 10KB
                repo.get('stargazers_count', 0) >= 0):  # Any star count
                significant.append(repo)
        
        # Sort by a combination of stars, size, and recency
        significant.sort(
            key=lambda x: (x.get('stargazers_count', 0) * 2 + 
                          x.get('size', 0) / 1000 + 
                          (1 if x.get('updated_at', '') > '2023-01-01' else 0)),
            reverse=True
        )
        
        return significant
    
    def _analyze_repository(self, username: str, repo_name: str) -> Dict[str, Any]:
        """
        Perform detailed analysis of a single repository.
        
        Args:
            username: GitHub username
            repo_name: Repository name
            
        Returns:
            Dictionary containing repository analysis results
        """
        # Clone repository to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir) / repo_name
            
            try:
                # Clone the repository
                repo_url = f"https://github.com/{username}/{repo_name}.git"
                Repo.clone_from(repo_url, repo_path)
                
                # Analyze code structure and complexity
                metrics = self._analyze_code_structure(repo_path)
                
                # Calculate test coverage if tests exist
                coverage_score = self._calculate_test_coverage(repo_path)
                metrics.test_coverage = coverage_score
                
                # Check originality
                originality = self._check_originality(repo_path)
                metrics.originality_score = originality
                
                return {
                    'repository_name': repo_name,
                    'metrics': metrics,
                    'analysis_successful': True
                }
                
            except Exception as e:
                print(f"Error cloning/analyzing repository {repo_name}: {e}")
                return {
                    'repository_name': repo_name,
                    'metrics': CodeMetrics(0, 0, 0.0, 0, 0.0, 0.0, 0.0, 0.0),
                    'analysis_successful': False,
                    'error': str(e)
                }
    
    def _analyze_code_structure(self, repo_path: Path) -> CodeMetrics:
        """
        Analyze code structure using AST parsing and complexity metrics.
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            CodeMetrics object with analysis results
        """
        total_functions = 0
        total_classes = 0
        total_complexity = 0.0
        total_loc = 0
        total_halstead = 0.0
        total_maintainability = 0.0
        file_count = 0
        
        # Find all Python files
        python_files = list(repo_path.rglob('*.py'))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # AST-based analysis
                tree = ast.parse(content)
                
                # Count functions and classes
                functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                
                total_functions += len(functions)
                total_classes += len(classes)
                
                # Complexity analysis using Radon
                complexity_visitor = ComplexityVisitor.from_code(content)
                file_complexity = sum(block.complexity for block in complexity_visitor.blocks)
                total_complexity += file_complexity
                
                # Lines of code
                loc = len([line for line in content.split('\n') if line.strip() and not line.strip().startswith('#')])
                total_loc += loc
                
                # Halstead metrics
                try:
                    halstead = radon_metrics.h_visit(content)
                    if halstead:
                        total_halstead += halstead.volume
                except:
                    pass
                
                # Maintainability index
                try:
                    mi = radon_metrics.mi_visit(content, multi=True)
                    if mi:
                        total_maintainability += mi
                except:
                    pass
                
                file_count += 1
                
            except Exception as e:
                print(f"Error analyzing file {py_file}: {e}")
                continue
        
        # Calculate averages
        avg_complexity = total_complexity / max(file_count, 1)
        avg_halstead = total_halstead / max(file_count, 1)
        avg_maintainability = total_maintainability / max(file_count, 1)
        
        return CodeMetrics(
            function_count=total_functions,
            class_count=total_classes,
            cyclomatic_complexity=avg_complexity,
            lines_of_code=total_loc,
            test_coverage=0.0,  # Will be calculated separately
            maintainability_index=avg_maintainability,
            halstead_volume=avg_halstead,
            originality_score=0.0  # Will be calculated separately
        )
    
    def _calculate_test_coverage(self, repo_path: Path) -> float:
        """
        Calculate test coverage for the repository.
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            Test coverage percentage (0-100)
        """
        try:
            # Look for test files
            test_files = list(repo_path.rglob('test_*.py')) + list(repo_path.rglob('*_test.py'))
            test_dirs = list(repo_path.rglob('tests/'))
            
            if not test_files and not test_dirs:
                return 0.0
            
            # Simple heuristic: ratio of test files to source files
            python_files = list(repo_path.rglob('*.py'))
            source_files = [f for f in python_files if not any(test_pattern in str(f) for test_pattern in ['test_', '_test', '/tests/'])]
            
            if not source_files:
                return 0.0
            
            test_ratio = len(test_files) / len(source_files)
            
            # Convert to percentage with a cap at 100%
            coverage_estimate = min(test_ratio * 100, 100.0)
            
            return coverage_estimate
            
        except Exception as e:
            print(f"Error calculating test coverage: {e}")
            return 0.0
    
    def _check_originality(self, repo_path: Path) -> float:
        """
        Check code originality using simple heuristics.
        Note: In production, this would integrate with plagiarism detection APIs.
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            Originality score (0-100)
        """
        try:
            # Simple heuristics for originality
            python_files = list(repo_path.rglob('*.py'))
            
            if not python_files:
                return 50.0  # Neutral score for non-Python repos
            
            # Check for common indicators of original work
            originality_indicators = 0
            total_checks = 0
            
            for py_file in python_files[:10]:  # Check first 10 files
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    total_checks += 1
                    
                    # Check for custom function names (not just main, init, etc.)
                    tree = ast.parse(content)
                    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                    
                    common_names = {'main', '__init__', 'test', 'setup', 'run'}
                    custom_functions = [f for f in functions if f not in common_names]
                    
                    if len(custom_functions) > 2:
                        originality_indicators += 1
                    
                    # Check for meaningful comments
                    comment_lines = [line for line in content.split('\n') if line.strip().startswith('#')]
                    if len(comment_lines) > 5:
                        originality_indicators += 0.5
                    
                    # Check for custom classes
                    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                    if len(classes) > 0:
                        originality_indicators += 0.5
                        
                except Exception:
                    continue
            
            if total_checks == 0:
                return 50.0
            
            originality_score = (originality_indicators / total_checks) * 100
            return min(max(originality_score, 10.0), 95.0)  # Clamp between 10-95%
            
        except Exception as e:
            print(f"Error checking originality: {e}")
            return 50.0  # Default neutral score
    
    def _aggregate_metrics(self, total: CodeMetrics, new: CodeMetrics) -> CodeMetrics:
        """
        Aggregate metrics from multiple repositories.
        
        Args:
            total: Current aggregated metrics
            new: New metrics to add
            
        Returns:
            Updated aggregated metrics
        """
        return CodeMetrics(
            function_count=total.function_count + new.function_count,
            class_count=total.class_count + new.class_count,
            cyclomatic_complexity=(total.cyclomatic_complexity + new.cyclomatic_complexity) / 2,
            lines_of_code=total.lines_of_code + new.lines_of_code,
            test_coverage=(total.test_coverage + new.test_coverage) / 2,
            maintainability_index=(total.maintainability_index + new.maintainability_index) / 2,
            halstead_volume=(total.halstead_volume + new.halstead_volume) / 2,
            originality_score=(total.originality_score + new.originality_score) / 2
        )
    
    def _calculate_technical_score(self, metrics: CodeMetrics) -> float:
        """
        Calculate overall technical skill score based on metrics.
        
        Args:
            metrics: Aggregated code metrics
            
        Returns:
            Technical skill score (0-100)
        """
        # Normalize individual metrics to 0-100 scale
        complexity_score = min(metrics.cyclomatic_complexity * 10, 100)  # Higher complexity = higher score (to a point)
        if complexity_score > 50:
            complexity_score = 100 - complexity_score  # Penalize overly complex code
        
        coverage_score = metrics.test_coverage
        
        structure_score = min((metrics.function_count + metrics.class_count) * 2, 100)
        
        maintainability_score = min(metrics.maintainability_index * 5, 100)
        
        originality_score = metrics.originality_score
        
        # Weighted combination
        total_score = (
            self.weights['complexity'] * complexity_score +
            self.weights['test_coverage'] * coverage_score +
            self.weights['code_structure'] * structure_score +
            self.weights['maintainability'] * maintainability_score +
            self.weights['originality'] * originality_score
        )
        
        return min(max(total_score, 0.0), 100.0)
    
    def _calculate_quality_grade(self, technical_score: float) -> str:
        """
        Convert technical score to letter grade.
        
        Args:
            technical_score: Technical skill score (0-100)
            
        Returns:
            Letter grade (A-F)
        """
        if technical_score >= 90:
            return 'A'
        elif technical_score >= 80:
            return 'B'
        elif technical_score >= 70:
            return 'C'
        elif technical_score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _extract_languages(self, repos: List[Dict]) -> Dict[str, int]:
        """
        Extract programming languages used across repositories.
        
        Args:
            repos: List of repository information
            
        Returns:
            Dictionary of languages and their usage count
        """
        languages = {}
        
        for repo in repos:
            if repo.get('language'):
                lang = repo['language']
                languages[lang] = languages.get(lang, 0) + 1
        
        return languages
    
    def _analyze_contribution_patterns(self, username: str) -> Dict[str, Any]:
        """
        Analyze contribution patterns from GitHub API.
        
        Args:
            username: GitHub username
            
        Returns:
            Dictionary containing contribution analysis
        """
        try:
            # Get user events (recent activity)
            url = f"https://api.github.com/users/{username}/events"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                events = response.json()
                
                # Analyze event types
                event_types = {}
                for event in events[:100]:  # Last 100 events
                    event_type = event.get('type', 'Unknown')
                    event_types[event_type] = event_types.get(event_type, 0) + 1
                
                return {
                    'recent_activity_count': len(events),
                    'activity_types': event_types,
                    'most_common_activity': max(event_types.items(), key=lambda x: x[1])[0] if event_types else None
                }
            else:
                return {'error': f'Failed to fetch events: {response.status_code}'}
                
        except Exception as e:
            return {'error': str(e)}
    
    def _create_default_score(self) -> GitHubSkillScore:
        """
        Create a default score when analysis fails.
        
        Returns:
            Default GitHubSkillScore
        """
        return GitHubSkillScore(
            technical_skill_score=0.0,
            code_quality_grade='F',
            originality_percentage=0.0,
            detailed_metrics=CodeMetrics(0, 0, 0.0, 0, 0.0, 0.0, 0.0, 0.0),
            repository_analysis={'error': 'Analysis failed'}
        )
    
    def save_analysis(self, username: str, score: GitHubSkillScore, output_path: str) -> None:
        """
        Save analysis results to JSON file.
        
        Args:
            username: GitHub username
            score: GitHubSkillScore object
            output_path: Path to save the results
        """
        result = {
            'username': username,
            'analysis_timestamp': str(np.datetime64('now')),
            'technical_skill_score': score.technical_skill_score,
            'code_quality_grade': score.code_quality_grade,
            'originality_percentage': score.originality_percentage,
            'detailed_metrics': {
                'function_count': score.detailed_metrics.function_count,
                'class_count': score.detailed_metrics.class_count,
                'cyclomatic_complexity': score.detailed_metrics.cyclomatic_complexity,
                'lines_of_code': score.detailed_metrics.lines_of_code,
                'test_coverage': score.detailed_metrics.test_coverage,
                'maintainability_index': score.detailed_metrics.maintainability_index,
                'halstead_volume': score.detailed_metrics.halstead_volume,
                'originality_score': score.detailed_metrics.originality_score
            },
            'repository_analysis': score.repository_analysis
        }
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)