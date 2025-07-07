import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import shap
from lime.lime_tabular import LimeTabularExplainer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class ExplanationResult:
    """Container for explanation results"""
    feature_importance: Dict[str, float]
    shap_values: Optional[np.ndarray] = None
    lime_explanation: Optional[Any] = None
    feature_names: Optional[List[str]] = None


@dataclass
class ReportVisualization:
    """Container for visualization data"""
    chart_type: str  # 'bar', 'radar', 'heatmap', etc.
    title: str
    data: Any
    layout_options: Dict[str, Any]
    file_path: Optional[str] = None


class ExplainabilityReportingLayer:
    """
    Generate human-interpretable explanations for model predictions and
    create comprehensive visual reports highlighting strengths, weaknesses,
    and actionable recommendations.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the Explainability & Reporting Layer.
        
        Args:
            output_dir: Directory to save reports and visualizations
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir) if output_dir else Path("reports")
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure visualization style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def explain_prediction(self, model: Any, features: np.ndarray, 
                         feature_names: List[str]) -> ExplanationResult:
        """
        Generate model explanations using SHAP and LIME.
        
        Args:
            model: Trained ML model
            features: Feature vector to explain
            feature_names: Names of features
            
        Returns:
            ExplanationResult with various explanation outputs
        """
        try:
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else \
                       shap.KernelExplainer(model.predict_proba, shap.sample(features, 100))
            
            shap_values = explainer.shap_values(features)
            
            # If binary classification, use the positive class
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
            
            # Calculate feature importance from SHAP values
            feature_importance = {}
            for i, name in enumerate(feature_names):
                if len(shap_values.shape) == 2:  # Multiple samples
                    importance = np.mean(np.abs(shap_values[:, i]))
                else:  # Single sample
                    importance = np.abs(shap_values[i])
                feature_importance[name] = float(importance)
            
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True))
            
            # Generate LIME explanation
            lime_explainer = LimeTabularExplainer(
                training_data=np.random.normal(0, 1, (100, len(feature_names))),
                feature_names=feature_names,
                class_names=['Negative', 'Positive'],
                mode='classification'
            )
            
            lime_exp = lime_explainer.explain_instance(
                data_row=features[0] if len(features.shape) > 1 else features,
                predict_fn=model.predict_proba,
                num_features=len(feature_names)
            )
            
            return ExplanationResult(
                feature_importance=feature_importance,
                shap_values=shap_values,
                lime_explanation=lime_exp,
                feature_names=feature_names
            )
            
        except Exception as e:
            self.logger.error(f"Error generating explanations: {e}")
            return ExplanationResult(feature_importance={})
    
    def create_feature_importance_chart(self, explanation: ExplanationResult, 
                                      title: str = "Feature Importance") -> ReportVisualization:
        """
        Create feature importance bar chart visualization.
        
        Args:
            explanation: Explanation result
            title: Chart title
            
        Returns:
            ReportVisualization object
        """
        # Sort features by importance
        features = list(explanation.feature_importance.keys())
        importance = list(explanation.feature_importance.values())
        
        # Create sorted lists
        sorted_indices = np.argsort(importance)
        sorted_features = [features[i] for i in sorted_indices[-10:]]  # Top 10
        sorted_importance = [importance[i] for i in sorted_indices[-10:]]
        
        # Create Plotly figure
        fig = px.bar(
            x=sorted_importance,
            y=sorted_features,
            orientation='h',
            labels={'x': 'Importance', 'y': 'Feature'},
            title=title
        )
        
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return ReportVisualization(
            chart_type='bar',
            title=title,
            data=fig,
            layout_options={},
            file_path=str(self.output_dir / "feature_importance.html")
        )
    
    def create_radar_chart(self, scores: Dict[str, float], 
                         title: str = "Candidate Profile") -> ReportVisualization:
        """
        Create radar chart for candidate profile visualization.
        
        Args:
            scores: Dictionary of category scores
            title: Chart title
            
        Returns:
            ReportVisualization object
        """
        categories = list(scores.keys())
        values = list(scores.values())
        
        # Close the loop for radar chart
        categories.append(categories[0])
        values.append(values[0])
        
        # Create Plotly figure
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Candidate Profile'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title=title,
            showlegend=False
        )
        
        return ReportVisualization(
            chart_type='radar',
            title=title,
            data=fig,
            layout_options={},
            file_path=str(self.output_dir / "radar_chart.html")
        )
    
    def create_prediction_probability_chart(self, probabilities: Dict[str, float], 
                                         title: str = "Prediction Probabilities") -> ReportVisualization:
        """
        Create gauge charts for prediction probabilities.
        
        Args:
            probabilities: Dictionary of prediction probabilities
            title: Chart title
            
        Returns:
            ReportVisualization object
        """
        # Create subplot with multiple gauge charts
        fig = make_subplots(
            rows=1, 
            cols=len(probabilities),
            specs=[[{'type': 'indicator'} for _ in range(len(probabilities))]]
        )
        
        # Add each probability as a gauge
        for i, (name, prob) in enumerate(probabilities.items(), 1):
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,  # Convert to percentage
                    title={'text': name.replace('_', ' ').title()},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': self._get_color_for_probability(prob)},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 75], 'color': "gray"},
                            {'range': [75, 100], 'color': "darkgray"}
                        ]
                    }
                ),
                row=1, col=i
            )
        
        fig.update_layout(
            title=title,
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return ReportVisualization(
            chart_type='gauge',
            title=title,
            data=fig,
            layout_options={},
            file_path=str(self.output_dir / "prediction_probabilities.html")
        )
    
    def create_platform_comparison_chart(self, platform_scores: Dict[str, Dict[str, float]], 
                                       title: str = "Platform Comparison") -> ReportVisualization:
        """
        Create platform comparison chart.
        
        Args:
            platform_scores: Dictionary of platform scores
            title: Chart title
            
        Returns:
            ReportVisualization object
        """
        platforms = []
        scores = []
        weights = []
        
        for platform, data in platform_scores.items():
            platforms.append(platform.replace('_', ' ').title())
            scores.append(data.get('normalized_score', 0))
            weights.append(data.get('weight', 0))
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Platform': platforms,
            'Score': scores,
            'Weight': weights
        })
        
        # Sort by score
        df = df.sort_values('Score', ascending=False)
        
        # Create Plotly figure
        fig = px.bar(
            df,
            x='Platform',
            y='Score',
            color='Score',
            color_continuous_scale='Viridis',
            labels={'Score': 'Normalized Score (0-100)'},
            title=title,
            hover_data=['Weight']
        )
        
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis_range=[0, 100]
        )
        
        return ReportVisualization(
            chart_type='bar',
            title=title,
            data=fig,
            layout_options={},
            file_path=str(self.output_dir / "platform_comparison.html")
        )
    
    def create_strengths_weaknesses_chart(self, strengths: List[str], weaknesses: List[str], 
                                        title: str = "Strengths & Weaknesses") -> ReportVisualization:
        """
        Create strengths and weaknesses visualization.
        
        Args:
            strengths: List of strength descriptions
            weaknesses: List of weakness descriptions
            title: Chart title
            
        Returns:
            ReportVisualization object
        """
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Category': ['Strengths'] * len(strengths) + ['Weaknesses'] * len(weaknesses),
            'Description': strengths + weaknesses,
            'Value': [1] * len(strengths) + [-1] * len(weaknesses)
        })
        
        # Create Plotly figure
        fig = px.bar(
            df,
            x='Value',
            y='Description',
            color='Category',
            color_discrete_map={'Strengths': 'green', 'Weaknesses': 'red'},
            orientation='h',
            title=title
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_visible=False,
            xaxis_showticklabels=False
        )
        
        return ReportVisualization(
            chart_type='bar',
            title=title,
            data=fig,
            layout_options={},
            file_path=str(self.output_dir / "strengths_weaknesses.html")
        )
    
    def generate_comprehensive_report(self, candidate_data: Dict[str, Any], 
                                    prediction_result: Any, 
                                    platform_scores: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate comprehensive HTML report with all visualizations.
        
        Args:
            candidate_data: Candidate information
            prediction_result: Prediction results
            platform_scores: Platform scores
            
        Returns:
            Path to generated HTML report
        """
        try:
            # Create visualizations
            visualizations = []
            
            # 1. Radar chart of platform scores
            radar_data = {}
            for platform, data in platform_scores.items():
                radar_data[platform.replace('_', ' ').title()] = data.get('normalized_score', 0)
            
            visualizations.append(self.create_radar_chart(radar_data, "Platform Performance"))
            
            # 2. Prediction probabilities
            prob_data = {
                'Technical Round': prediction_result.technical_round_probability,
                'Culture Fit': prediction_result.culture_fit_probability,
                'Learning Adaptability': prediction_result.learning_adaptability_probability
            }
            
            visualizations.append(self.create_prediction_probability_chart(prob_data))
            
            # 3. Feature importance
            if prediction_result.feature_importance:
                explanation = ExplanationResult(
                    feature_importance=prediction_result.feature_importance
                )
                visualizations.append(self.create_feature_importance_chart(explanation))
            
            # 4. Strengths and weaknesses
            visualizations.append(self.create_strengths_weaknesses_chart(
                prediction_result.top_strengths,
                prediction_result.top_weaknesses
            ))
            
            # 5. Platform comparison
            platform_comparison_data = {}
            for platform, data in platform_scores.items():
                platform_comparison_data[platform] = {
                    'normalized_score': data.get('normalized_score', 0),
                    'weight': data.get('weight', 0)
                }
            
            visualizations.append(self.create_platform_comparison_chart(platform_comparison_data))
            
            # Save all visualizations
            for viz in visualizations:
                if viz.file_path and viz.data is not None:
                    viz.data.write_html(viz.file_path)
            
            # Generate HTML report
            report_path = self.output_dir / "comprehensive_report.html"
            self._generate_html_report(report_path, candidate_data, prediction_result, visualizations)
            
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            return ""
    
    def _generate_html_report(self, output_path: Path, candidate_data: Dict[str, Any],
                            prediction_result: Any, visualizations: List[ReportVisualization]) -> None:
        """
        Generate HTML report with all visualizations.
        
        Args:
            output_path: Path to save HTML report
            candidate_data: Candidate information
            prediction_result: Prediction results
            visualizations: List of visualizations
        """
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Candidate Assessment Report</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; margin-bottom: 20px; border-radius: 5px; }}
                .section {{ margin-bottom: 30px; }}
                .recommendation {{ background-color: #e9f7ef; padding: 15px; border-radius: 5px; margin-bottom: 10px; }}
                .chart-container {{ height: 400px; margin-bottom: 30px; }}
                .score-card {{ text-align: center; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .score-value {{ font-size: 36px; font-weight: bold; }}
                .grade-A {{ background-color: #d5f5e3; }}
                .grade-B {{ background-color: #d4efdf; }}
                .grade-C {{ background-color: #fcf3cf; }}
                .grade-D {{ background-color: #f5cba7; }}
                .grade-F {{ background-color: #f5b7b1; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Candidate Assessment Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <div class="score-card grade-{prediction_result.overall_recommendation[0]}">
                            <h3>Overall Recommendation</h3>
                            <div class="score-value">{prediction_result.overall_recommendation}</div>
                            <p>Confidence: {prediction_result.confidence_score:.2f}</p>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="score-card">
                            <h3>Prediction Probabilities</h3>
                            <div class="row">
                                <div class="col-md-4">
                                    <p>Technical: {prediction_result.technical_round_probability:.1%}</p>
                                </div>
                                <div class="col-md-4">
                                    <p>Culture: {prediction_result.culture_fit_probability:.1%}</p>
                                </div>
                                <div class="col-md-4">
                                    <p>Learning: {prediction_result.learning_adaptability_probability:.1%}</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Key Insights</h2>
                    <div class="row">
        """
        
        # Add visualizations
        for i, viz in enumerate(visualizations):
            if viz.file_path:
                viz_id = f"viz_{i}"
                viz_path = Path(viz.file_path).name
                
                html_content += f"""
                        <div class="col-md-6">
                            <div class="chart-container" id="{viz_id}"></div>
                        </div>
                """
        
        html_content += f"""
                    </div>
                </div>
                
                <div class="section">
                    <h2>Strengths & Weaknesses</h2>
                    <div class="row">
                        <div class="col-md-6">
                            <h3>Strengths</h3>
                            <ul>
        """
        
        # Add strengths
        for strength in prediction_result.top_strengths:
            html_content += f"<li>{strength}</li>\n"
        
        html_content += f"""
                            </ul>
                        </div>
                        <div class="col-md-6">
                            <h3>Weaknesses</h3>
                            <ul>
        """
        
        # Add weaknesses
        for weakness in prediction_result.top_weaknesses:
            html_content += f"<li>{weakness}</li>\n"
        
        html_content += f"""
                            </ul>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Actionable Recommendations</h2>
        """
        
        # Add recommendations
        for recommendation in prediction_result.actionable_recommendations:
            html_content += f"<div class=\"recommendation\">{recommendation}</div>\n"
        
        html_content += f"""
                </div>
            </div>
            
            <script>
        """
        
        # Add JavaScript to load visualizations
        for i, viz in enumerate(visualizations):
            if viz.file_path:
                viz_id = f"viz_{i}"
                viz_path = Path(viz.file_path).name
                
                html_content += f"""
                fetch('{viz_path}')
                    .then(response => response.text())
                    .then(html => {{                        
                        const parser = new DOMParser();
                        const doc = parser.parseFromString(html, 'text/html');
                        const plotlyDiv = doc.querySelector('.plotly-graph-div');
                        const plotlyData = JSON.parse(plotlyDiv.getAttribute('data-plotly'));
                        Plotly.newPlot('{viz_id}', plotlyData.data, plotlyData.layout);
                    }});
                """
        
        html_content += f"""
            </script>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _get_color_for_probability(self, probability: float) -> str:
        """Get color based on probability value"""
        if probability >= 0.8:
            return "green"
        elif probability >= 0.6:
            return "lightgreen"
        elif probability >= 0.4:
            return "yellow"
        elif probability >= 0.2:
            return "orange"
        else:
            return "red"
    
    def generate_json_report(self, candidate_data: Dict[str, Any], 
                           prediction_result: Any, 
                           platform_scores: Dict[str, Dict[str, Any]],
                           output_path: Optional[str] = None) -> str:
        """
        Generate JSON report with all analysis results.
        
        Args:
            candidate_data: Candidate information
            prediction_result: Prediction results
            platform_scores: Platform scores
            output_path: Path to save JSON report
            
        Returns:
            Path to generated JSON report
        """
        try:
            # Create report structure
            report = {
                "candidate_info": {
                    "name": candidate_data.get("name", "Unknown"),
                    "analysis_timestamp": datetime.now().isoformat()
                },
                "overall_assessment": {
                    "recommendation": prediction_result.overall_recommendation,
                    "confidence_score": prediction_result.confidence_score,
                    "technical_round_probability": prediction_result.technical_round_probability,
                    "culture_fit_probability": prediction_result.culture_fit_probability,
                    "learning_adaptability_probability": prediction_result.learning_adaptability_probability
                },
                "platform_scores": {},
                "strengths": prediction_result.top_strengths,
                "weaknesses": prediction_result.top_weaknesses,
                "recommendations": prediction_result.actionable_recommendations,
                "feature_importance": prediction_result.feature_importance
            }
            
            # Add platform scores
            for platform, data in platform_scores.items():
                report["platform_scores"][platform] = {
                    "normalized_score": data.get("normalized_score", 0),
                    "raw_score": data.get("raw_score", 0),
                    "weight": data.get("weight", 0),
                    "confidence": data.get("confidence", 0)
                }
            
            # Determine output path
            if not output_path:
                output_path = str(self.output_dir / "candidate_assessment_report.json")
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating JSON report: {e}")
            return ""