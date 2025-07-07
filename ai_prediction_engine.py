import numpy as np
import pandas as pd
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from pathlib import Path

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.neural_network import MLPClassifier
import joblib

# For feature importance and explainability
import shap
from lime.lime_tabular import LimeTabularExplainer


@dataclass
class CandidateFeatures:
    """Structured candidate features for ML model"""
    # Platform scores
    github_score: float
    leetcode_score: float
    kaggle_score: float
    linkedin_score: float
    resume_score: float
    certification_score: float
    
    # Metadata features
    years_experience: float
    education_level: int  # 0=High School, 1=Bachelor, 2=Master, 3=PhD
    total_projects: int
    programming_languages_count: int
    
    # Derived features
    overall_activity_score: float
    technical_depth_score: float
    social_presence_score: float


@dataclass
class PredictionResult:
    """ML prediction results with explanations"""
    technical_round_probability: float
    culture_fit_probability: float
    learning_adaptability_probability: float
    
    # Overall recommendation
    overall_recommendation: str  # "Strong Hire", "Hire", "Maybe", "No Hire"
    confidence_score: float
    
    # Explanations
    feature_importance: Dict[str, float]
    top_strengths: List[str]
    top_weaknesses: List[str]
    actionable_recommendations: List[str]
    
    # Model metadata
    model_version: str
    prediction_timestamp: datetime


class AIPredictionEngine:
    """
    Leverage historical hiring data to forecast candidate outcomes beyond raw scores
    using machine learning models with explainability features.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the AI-Based Prediction Engine.
        
        Args:
            model_path: Path to pre-trained models directory
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = Path(model_path) if model_path else Path("models")
        self.model_path.mkdir(exist_ok=True)
        
        # Initialize models
        self.models = {
            'technical_round': None,
            'culture_fit': None,
            'learning_adaptability': None
        }
        
        # Feature scaler
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Explainability tools
        self.shap_explainers = {}
        self.lime_explainer = None
        
        # Load pre-trained models if available
        self._load_models()
    
    def train_models(self, training_data: pd.DataFrame, 
                    target_columns: Dict[str, str]) -> Dict[str, float]:
        """
        Train ML models on historical hiring data.
        
        Args:
            training_data: DataFrame with candidate features and outcomes
            target_columns: Dictionary mapping prediction type to column name
            
        Returns:
            Dictionary of model performance scores
        """
        performance_scores = {}
        
        # Prepare features
        feature_columns = self._get_feature_columns(training_data)
        X = training_data[feature_columns]
        self.feature_names = feature_columns
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train each prediction model
        for prediction_type, target_col in target_columns.items():
            if target_col not in training_data.columns:
                self.logger.warning(f"Target column {target_col} not found in training data")
                continue
            
            y = training_data[target_col]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model with hyperparameter tuning
            model, score = self._train_single_model(
                X_train, X_test, y_train, y_test, prediction_type
            )
            
            self.models[prediction_type] = model
            performance_scores[prediction_type] = score
            
            # Initialize SHAP explainer
            self.shap_explainers[prediction_type] = shap.TreeExplainer(model)
        
        # Initialize LIME explainer
        self.lime_explainer = LimeTabularExplainer(
            X_scaled,
            feature_names=self.feature_names,
            class_names=['No', 'Yes'],
            mode='classification'
        )
        
        # Save models
        self._save_models()
        
        return performance_scores
    
    def predict_candidate_outcomes(self, candidate_data: Dict[str, Any]) -> PredictionResult:
        """
        Predict candidate outcomes using trained models.
        
        Args:
            candidate_data: Dictionary containing candidate information
            
        Returns:
            PredictionResult with predictions and explanations
        """
        try:
            # Extract and prepare features
            features = self._extract_features(candidate_data)
            feature_vector = self._features_to_vector(features)
            
            # Scale features
            feature_vector_scaled = self.scaler.transform([feature_vector])
            
            # Make predictions
            predictions = {}
            feature_importance = {}
            
            for prediction_type, model in self.models.items():
                if model is not None:
                    # Get probability prediction
                    prob = model.predict_proba(feature_vector_scaled)[0][1]  # Probability of positive class
                    predictions[prediction_type] = prob
                    
                    # Get feature importance using SHAP
                    if prediction_type in self.shap_explainers:
                        shap_values = self.shap_explainers[prediction_type].shap_values(feature_vector_scaled)
                        if isinstance(shap_values, list):
                            shap_values = shap_values[1]  # For binary classification
                        
                        # Map SHAP values to feature names
                        for i, feature_name in enumerate(self.feature_names):
                            if feature_name not in feature_importance:
                                feature_importance[feature_name] = 0
                            feature_importance[feature_name] += abs(shap_values[0][i])
                else:
                    predictions[prediction_type] = 0.5  # Default neutral prediction
            
            # Calculate overall recommendation
            overall_rec, confidence = self._calculate_overall_recommendation(predictions)
            
            # Generate explanations
            strengths, weaknesses = self._identify_strengths_weaknesses(features, feature_importance)
            recommendations = self._generate_actionable_recommendations(features, predictions)
            
            return PredictionResult(
                technical_round_probability=predictions.get('technical_round', 0.5),
                culture_fit_probability=predictions.get('culture_fit', 0.5),
                learning_adaptability_probability=predictions.get('learning_adaptability', 0.5),
                overall_recommendation=overall_rec,
                confidence_score=confidence,
                feature_importance=feature_importance,
                top_strengths=strengths,
                top_weaknesses=weaknesses,
                actionable_recommendations=recommendations,
                model_version="1.0",
                prediction_timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            return self._create_default_prediction()
    
    def _train_single_model(self, X_train: np.ndarray, X_test: np.ndarray, 
                           y_train: np.ndarray, y_test: np.ndarray, 
                           prediction_type: str) -> Tuple[Any, float]:
        """
        Train a single ML model with hyperparameter tuning.
        
        Args:
            X_train, X_test, y_train, y_test: Training and test data
            prediction_type: Type of prediction being made
            
        Returns:
            Tuple of (trained_model, performance_score)
        """
        # Define model candidates
        models_to_try = {
            'random_forest': RandomForestClassifier(random_state=42),
            'xgboost': XGBClassifier(random_state=42, eval_metric='logloss'),
            'neural_network': MLPClassifier(random_state=42, max_iter=1000)
        }
        
        # Hyperparameter grids
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'xgboost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
        
        best_model = None
        best_score = 0
        
        # Try each model type
        for model_name, model in models_to_try.items():
            try:
                # Grid search with cross-validation
                grid_search = GridSearchCV(
                    model, param_grids[model_name], 
                    cv=5, scoring='roc_auc', n_jobs=-1
                )
                
                grid_search.fit(X_train, y_train)
                
                # Evaluate on test set
                test_score = roc_auc_score(y_test, grid_search.predict_proba(X_test)[:, 1])
                
                if test_score > best_score:
                    best_score = test_score
                    best_model = grid_search.best_estimator_
                
                self.logger.info(f"{prediction_type} - {model_name}: {test_score:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name} for {prediction_type}: {e}")
                continue
        
        return best_model, best_score
    
    def _extract_features(self, candidate_data: Dict[str, Any]) -> CandidateFeatures:
        """
        Extract structured features from candidate data.
        
        Args:
            candidate_data: Raw candidate data
            
        Returns:
            CandidateFeatures object
        """
        # Extract platform scores
        platform_scores = candidate_data.get('platform_scores', {})
        
        github_score = platform_scores.get('github', {}).get('normalized_score', 0)
        leetcode_score = platform_scores.get('leetcode', {}).get('normalized_score', 0)
        kaggle_score = platform_scores.get('kaggle', {}).get('normalized_score', 0)
        linkedin_score = platform_scores.get('linkedin', {}).get('normalized_score', 0)
        resume_score = platform_scores.get('resume', {}).get('normalized_score', 0)
        certification_score = platform_scores.get('certification', {}).get('normalized_score', 0)
        
        # Extract metadata
        years_experience = candidate_data.get('years_experience', 0)
        education_level = self._encode_education(candidate_data.get('education_level', 'Bachelor'))
        total_projects = candidate_data.get('total_projects', 0)
        programming_languages = candidate_data.get('programming_languages', [])
        programming_languages_count = len(programming_languages) if isinstance(programming_languages, list) else 0
        
        # Calculate derived features
        overall_activity_score = np.mean([github_score, leetcode_score, kaggle_score])
        technical_depth_score = np.mean([github_score, leetcode_score, certification_score])
        social_presence_score = np.mean([linkedin_score, resume_score])
        
        return CandidateFeatures(
            github_score=github_score,
            leetcode_score=leetcode_score,
            kaggle_score=kaggle_score,
            linkedin_score=linkedin_score,
            resume_score=resume_score,
            certification_score=certification_score,
            years_experience=years_experience,
            education_level=education_level,
            total_projects=total_projects,
            programming_languages_count=programming_languages_count,
            overall_activity_score=overall_activity_score,
            technical_depth_score=technical_depth_score,
            social_presence_score=social_presence_score
        )
    
    def _features_to_vector(self, features: CandidateFeatures) -> List[float]:
        """Convert CandidateFeatures to feature vector"""
        return [
            features.github_score,
            features.leetcode_score,
            features.kaggle_score,
            features.linkedin_score,
            features.resume_score,
            features.certification_score,
            features.years_experience,
            features.education_level,
            features.total_projects,
            features.programming_languages_count,
            features.overall_activity_score,
            features.technical_depth_score,
            features.social_presence_score
        ]
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature column names from DataFrame"""
        # Define expected feature columns
        expected_features = [
            'github_score', 'leetcode_score', 'kaggle_score', 'linkedin_score',
            'resume_score', 'certification_score', 'years_experience', 'education_level',
            'total_projects', 'programming_languages_count', 'overall_activity_score',
            'technical_depth_score', 'social_presence_score'
        ]
        
        # Return only columns that exist in the DataFrame
        return [col for col in expected_features if col in df.columns]
    
    def _encode_education(self, education: str) -> int:
        """Encode education level to numeric value"""
        education_mapping = {
            'high school': 0,
            'bachelor': 1,
            'master': 2,
            'phd': 3,
            'doctorate': 3
        }
        return education_mapping.get(education.lower(), 1)  # Default to bachelor
    
    def _calculate_overall_recommendation(self, predictions: Dict[str, float]) -> Tuple[str, float]:
        """
        Calculate overall hiring recommendation based on individual predictions.
        
        Args:
            predictions: Dictionary of prediction probabilities
            
        Returns:
            Tuple of (recommendation_string, confidence_score)
        """
        # Weighted average of predictions
        weights = {
            'technical_round': 0.4,
            'culture_fit': 0.3,
            'learning_adaptability': 0.3
        }
        
        weighted_score = sum(
            predictions.get(pred_type, 0.5) * weight 
            for pred_type, weight in weights.items()
        )
        
        # Calculate confidence based on consistency of predictions
        pred_values = list(predictions.values())
        confidence = 1.0 - np.std(pred_values) if len(pred_values) > 1 else 0.5
        
        # Determine recommendation
        if weighted_score >= 0.8:
            return "Strong Hire", confidence
        elif weighted_score >= 0.65:
            return "Hire", confidence
        elif weighted_score >= 0.45:
            return "Maybe", confidence
        else:
            return "No Hire", confidence
    
    def _identify_strengths_weaknesses(self, features: CandidateFeatures, 
                                     feature_importance: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """
        Identify top strengths and weaknesses based on features and importance.
        
        Args:
            features: Candidate features
            feature_importance: Feature importance scores
            
        Returns:
            Tuple of (strengths_list, weaknesses_list)
        """
        feature_values = asdict(features)
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        strengths = []
        weaknesses = []
        
        for feature_name, importance in sorted_features[:10]:  # Top 10 most important features
            if feature_name in feature_values:
                value = feature_values[feature_name]
                
                # Determine if this is a strength or weakness
                if value >= 70:  # High score
                    strengths.append(f"Strong {feature_name.replace('_', ' ')}: {value:.1f}")
                elif value <= 30:  # Low score
                    weaknesses.append(f"Weak {feature_name.replace('_', ' ')}: {value:.1f}")
        
        return strengths[:3], weaknesses[:3]  # Top 3 each
    
    def _generate_actionable_recommendations(self, features: CandidateFeatures, 
                                           predictions: Dict[str, float]) -> List[str]:
        """
        Generate actionable recommendations for improvement.
        
        Args:
            features: Candidate features
            predictions: Prediction results
            
        Returns:
            List of actionable recommendations
        """
        recommendations = []
        
        # Technical recommendations
        if features.github_score < 50:
            recommendations.append("Improve GitHub presence with more original projects and better code quality")
        
        if features.leetcode_score < 50:
            recommendations.append("Practice more algorithmic problems on LeetCode to improve technical skills")
        
        if features.certification_score < 30:
            recommendations.append("Obtain relevant industry certifications to validate technical knowledge")
        
        # Experience recommendations
        if features.years_experience < 2:
            recommendations.append("Gain more hands-on experience through internships or personal projects")
        
        # Social presence recommendations
        if features.linkedin_score < 40:
            recommendations.append("Enhance LinkedIn profile with more connections and professional content")
        
        # Prediction-specific recommendations
        if predictions.get('technical_round', 0.5) < 0.6:
            recommendations.append("Focus on technical interview preparation and coding practice")
        
        if predictions.get('culture_fit', 0.5) < 0.6:
            recommendations.append("Develop soft skills and demonstrate cultural alignment through projects")
        
        return recommendations[:5]  # Limit to top 5
    
    def _save_models(self) -> None:
        """Save trained models and preprocessing objects"""
        try:
            # Save models
            for model_name, model in self.models.items():
                if model is not None:
                    model_file = self.model_path / f"{model_name}_model.pkl"
                    joblib.dump(model, model_file)
            
            # Save scaler
            scaler_file = self.model_path / "scaler.pkl"
            joblib.dump(self.scaler, scaler_file)
            
            # Save feature names
            features_file = self.model_path / "feature_names.json"
            with open(features_file, 'w') as f:
                json.dump(self.feature_names, f)
            
            self.logger.info("Models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def _load_models(self) -> None:
        """Load pre-trained models and preprocessing objects"""
        try:
            # Load models
            for model_name in self.models.keys():
                model_file = self.model_path / f"{model_name}_model.pkl"
                if model_file.exists():
                    self.models[model_name] = joblib.load(model_file)
            
            # Load scaler
            scaler_file = self.model_path / "scaler.pkl"
            if scaler_file.exists():
                self.scaler = joblib.load(scaler_file)
            
            # Load feature names
            features_file = self.model_path / "feature_names.json"
            if features_file.exists():
                with open(features_file, 'r') as f:
                    self.feature_names = json.load(f)
            
            self.logger.info("Models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
    
    def _create_default_prediction(self) -> PredictionResult:
        """Create default prediction when analysis fails"""
        return PredictionResult(
            technical_round_probability=0.5,
            culture_fit_probability=0.5,
            learning_adaptability_probability=0.5,
            overall_recommendation="Unable to assess",
            confidence_score=0.0,
            feature_importance={},
            top_strengths=[],
            top_weaknesses=[],
            actionable_recommendations=["Unable to generate recommendations due to analysis error"],
            model_version="1.0",
            prediction_timestamp=datetime.now()
        )
    
    def generate_synthetic_training_data(self, num_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic training data for model development.
        Note: In production, this would be replaced with real historical hiring data.
        
        Args:
            num_samples: Number of synthetic samples to generate
            
        Returns:
            DataFrame with synthetic training data
        """
        np.random.seed(42)
        
        data = []
        
        for _ in range(num_samples):
            # Generate correlated features
            github_score = np.random.normal(50, 20)
            leetcode_score = np.random.normal(45, 25)
            kaggle_score = np.random.normal(30, 20)
            linkedin_score = np.random.normal(60, 15)
            resume_score = np.random.normal(55, 20)
            certification_score = np.random.normal(40, 25)
            
            years_experience = max(0, np.random.normal(3, 2))
            education_level = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            total_projects = max(0, int(np.random.normal(5, 3)))
            programming_languages_count = max(1, int(np.random.normal(4, 2)))
            
            # Derived features
            overall_activity_score = np.mean([github_score, leetcode_score, kaggle_score])
            technical_depth_score = np.mean([github_score, leetcode_score, certification_score])
            social_presence_score = np.mean([linkedin_score, resume_score])
            
            # Generate outcomes based on features (with some noise)
            technical_round_success = (
                (github_score * 0.3 + leetcode_score * 0.4 + certification_score * 0.3) / 100 + 
                np.random.normal(0, 0.1)
            ) > 0.5
            
            culture_fit_success = (
                (linkedin_score * 0.4 + resume_score * 0.3 + years_experience * 10 + education_level * 10) / 100 + 
                np.random.normal(0, 0.1)
            ) > 0.5
            
            learning_adaptability_success = (
                (overall_activity_score * 0.5 + total_projects * 5 + programming_languages_count * 5) / 100 + 
                np.random.normal(0, 0.1)
            ) > 0.5
            
            data.append({
                'github_score': max(0, min(100, github_score)),
                'leetcode_score': max(0, min(100, leetcode_score)),
                'kaggle_score': max(0, min(100, kaggle_score)),
                'linkedin_score': max(0, min(100, linkedin_score)),
                'resume_score': max(0, min(100, resume_score)),
                'certification_score': max(0, min(100, certification_score)),
                'years_experience': years_experience,
                'education_level': education_level,
                'total_projects': total_projects,
                'programming_languages_count': programming_languages_count,
                'overall_activity_score': max(0, min(100, overall_activity_score)),
                'technical_depth_score': max(0, min(100, technical_depth_score)),
                'social_presence_score': max(0, min(100, social_presence_score)),
                'technical_round_success': technical_round_success,
                'culture_fit_success': culture_fit_success,
                'learning_adaptability_success': learning_adaptability_success
            })
        
        return pd.DataFrame(data)
    
    def save_prediction_results(self, prediction: PredictionResult, output_path: str) -> None:
        """
        Save prediction results to JSON file.
        
        Args:
            prediction: PredictionResult object
            output_path: Path to save the results
        """
        result = {
            'technical_round_probability': prediction.technical_round_probability,
            'culture_fit_probability': prediction.culture_fit_probability,
            'learning_adaptability_probability': prediction.learning_adaptability_probability,
            'overall_recommendation': prediction.overall_recommendation,
            'confidence_score': prediction.confidence_score,
            'feature_importance': prediction.feature_importance,
            'top_strengths': prediction.top_strengths,
            'top_weaknesses': prediction.top_weaknesses,
            'actionable_recommendations': prediction.actionable_recommendations,
            'model_version': prediction.model_version,
            'prediction_timestamp': prediction.prediction_timestamp.isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)