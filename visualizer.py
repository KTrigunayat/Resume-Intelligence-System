#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualizer for Resume Intelligence System

This module provides visualization capabilities for the Resume Intelligence System,
including skill alignment heatmaps, section score charts, and project validation visualizations.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class Visualizer:
    """Visualizer for Resume Intelligence System results."""
    
    def __init__(self):
        """Initialize the visualizer."""
        # Set up matplotlib style
        plt.style.use('ggplot')
        self.colors = {
            'primary': '#4285F4',  # Google Blue
            'secondary': '#34A853',  # Google Green
            'tertiary': '#FBBC05',  # Google Yellow
            'quaternary': '#EA4335',  # Google Red
            'background': '#F8F9FA'
        }
    
    def visualize_skill_alignment(self, alignment_data, output_path=None):
        """Visualize skill alignment results.
        
        Args:
            alignment_data (dict): Skill alignment results.
            output_path (str, optional): Path to save the visualization. Defaults to None.
        """
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('Resume Skill Alignment Analysis', fontsize=16)
        
        # Define grid layout
        gs = fig.add_gridspec(2, 2)
        
        # 1. Overall alignment score (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_overall_score(ax1, alignment_data)
        
        # 2. Section scores (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_section_scores(ax2, alignment_data)
        
        # 3. Requirement scores (bottom, spans both columns)
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_requirement_scores(ax3, alignment_data)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_project_validation(self, validation_data, output_path=None):
        """Visualize project validation results.
        
        Args:
            validation_data (dict): Project validation results.
            output_path (str, optional): Path to save the visualization. Defaults to None.
        """
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('Project Validation Analysis', fontsize=16)
        
        # Define grid layout
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 2])
        
        # 1. Project scores (top)
        ax1 = fig.add_subplot(gs[0])
        self._plot_project_scores(ax1, validation_data)
        
        # 2. Validation metrics (bottom)
        ax2 = fig.add_subplot(gs[1])
        self._plot_validation_metrics(ax2, validation_data)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _plot_overall_score(self, ax, alignment_data):
        """Plot overall alignment score."""
        score = alignment_data.get('overall_alignment', 0)
        
        # Create gauge chart
        gauge_colors = ['#EA4335', '#FBBC05', '#34A853']
        cmap = plt.cm.RdYlGn
        norm = plt.Normalize(0, 100)
        
        # Draw gauge background
        ax.add_patch(plt.Circle((0.5, 0.5), 0.4, color='#F8F9FA', zorder=0))
        
        # Draw gauge
        for i in range(0, 100, 1):
            angle = np.pi * (0.75 + 1.5 * i / 100)
            x = 0.5 + 0.4 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            ax.add_patch(plt.Circle((x, y), 0.02, color=cmap(norm(i)), zorder=1))
        
        # Draw needle
        angle = np.pi * (0.75 + 1.5 * score / 100)
        ax.plot([0.5, 0.5 + 0.4 * np.cos(angle)], [0.5, 0.5 + 0.4 * np.sin(angle)], 
                color='black', linewidth=2, zorder=3)
        
        # Add score text
        ax.text(0.5, 0.3, f"{score:.1f}%", ha='center', va='center', 
                fontsize=24, fontweight='bold')
        ax.text(0.5, 0.2, "Overall Alignment", ha='center', va='center', fontsize=12)
        
        # Set axis properties
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Overall Alignment Score', fontsize=14)
    
    def _plot_section_scores(self, ax, alignment_data):
        """Plot section scores."""
        section_scores = alignment_data.get('section_scores', {})
        
        if not section_scores:
            ax.text(0.5, 0.5, "No section scores available", ha='center', va='center')
            ax.axis('off')
            return
        
        # Remove total_score if present
        if 'total_score' in section_scores:
            section_scores = {k: v for k, v in section_scores.items() if k != 'total_score'}
        
        # Sort sections by score
        sections = sorted(section_scores.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)
        labels = [s[0] for s in sections]
        values = [s[1] if isinstance(s[1], (int, float)) else 0 for s in sections]
        
        # Define max values for each section
        max_values = {
            'Projects': 30,
            'Work Experience': 30,
            'Skills': 20,
            'Education': 10,
            'Certifications': 5,
            'Summary': 5
        }
        
        # Get max value for each section
        max_vals = [max_values.get(label, 10) for label in labels]
        
        # Calculate percentages
        percentages = [v / m * 100 if m > 0 and isinstance(v, (int, float)) else 0 for v, m in zip(values, max_vals)]
        
        # Create horizontal bar chart
        bars = ax.barh(labels, percentages, color=self.colors['primary'], alpha=0.7)
        
        # Add percentage and raw score labels
        for i, (bar, val, max_val) in enumerate(zip(bars, values, max_vals)):
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, 
                    f"{val:.1f}/{max_val} ({percentages[i]:.1f}%)", 
                    va='center')
        
        # Set axis properties
        ax.set_xlim(0, 105)  # Leave room for labels
        ax.set_xlabel('Score (%)')
        ax.set_title('Section Scores', fontsize=14)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    def _plot_requirement_scores(self, ax, alignment_data):
        """Plot requirement scores."""
        req_scores = alignment_data.get('requirement_scores', {})
        
        if not req_scores:
            ax.text(0.5, 0.5, "No requirement scores available", ha='center', va='center')
            ax.axis('off')
            return
        
        # Sort requirements by score
        requirements = sorted(req_scores.items(), key=lambda x: x[1], reverse=True)
        labels = [r[0][:50] + '...' if len(r[0]) > 50 else r[0] for r in requirements]
        values = [r[1] * 100 for r in requirements]  # Convert to percentage
        
        # Create horizontal bar chart
        bars = ax.barh(labels, values, color=self.colors['secondary'], alpha=0.7)
        
        # Add percentage labels
        for bar in bars:
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                    f"{bar.get_width():.1f}%", va='center')
        
        # Set axis properties
        ax.set_xlim(0, 105)  # Leave room for labels
        ax.set_xlabel('Match Score (%)')
        ax.set_title('Requirement Match Scores', fontsize=14)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    def _plot_project_scores(self, ax, validation_data):
        """Plot project scores."""
        project_scores = validation_data.get('project_scores', {})
        
        if not project_scores:
            ax.text(0.5, 0.5, "No project scores available", ha='center', va='center')
            ax.axis('off')
            return
        
        # Sort projects by score
        projects = sorted(project_scores.items(), key=lambda x: x[1], reverse=True)
        labels = [p[0][:30] + '...' if len(p[0]) > 30 else p[0] for p in projects]
        values = [p[1] * 100 for p in projects]  # Convert to percentage
        
        # Create horizontal bar chart
        bars = ax.barh(labels, values, color=self.colors['tertiary'], alpha=0.7)
        
        # Add percentage labels
        for bar in bars:
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                    f"{bar.get_width():.1f}%", va='center')
        
        # Set axis properties
        ax.set_xlim(0, 105)  # Leave room for labels
        ax.set_xlabel('Validation Score (%)')
        ax.set_title('Project Validation Scores', fontsize=14)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    def _plot_validation_metrics(self, ax, validation_data):
        """Plot validation metrics for each project."""
        validation_metrics = validation_data.get('validation_metrics', {})
        
        if not validation_metrics:
            ax.text(0.5, 0.5, "No validation metrics available", ha='center', va='center')
            ax.axis('off')
            return
        
        # Prepare data for heatmap
        projects = list(validation_metrics.keys())
        metrics = ['skill_alignment', 'technical_depth', 'quantifiable_results']
        
        # Create data matrix
        data = np.zeros((len(projects), len(metrics)))
        for i, project in enumerate(projects):
            for j, metric in enumerate(metrics):
                data[i, j] = validation_metrics[project].get(metric, 0) * 100  # Convert to percentage
        
        # Create heatmap
        sns.heatmap(data, annot=True, fmt='.1f', cmap='YlGnBu', 
                   xticklabels=[m.replace('_', ' ').title() for m in metrics],
                   yticklabels=[p[:30] + '...' if len(p) > 30 else p for p in projects],
                   ax=ax, cbar_kws={'label': 'Score (%)'}, vmin=0, vmax=100)
        
        # Set axis properties
        ax.set_title('Project Validation Metrics', fontsize=14)
        plt.setp(ax.get_xticklabels(), rotation=0)
        plt.setp(ax.get_yticklabels(), rotation=0)
        
    def visualize_quality_scores(self, quality_data, output_path=None):
        """Visualize quality scores for different components.
        
        Args:
            quality_data (dict): Dictionary containing quality scores for different components.
            output_path (str, optional): Path to save the visualization. Defaults to None.
        """
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle('Resume Quality Analysis', fontsize=16)
        
        # Extract scores
        components = [
            'skill_alignment', 'project_validation', 'formatting', 
            'trustworthiness', 'credibility'
        ]
        
        # Get scores, defaulting to 0 if not present
        scores = [float(quality_data.get(comp, 0)) for comp in components]
        
        # Normalize scores to 0-100 range
        max_scores = {
            'skill_alignment': 100,  # Already 0-100
            'project_validation': 100,  # Already 0-100
            'formatting': 20,  # 0-20 scale
            'trustworthiness': 100,  # Already 0-100
            'credibility': 100  # Already 0-100
        }
        
        normalized_scores = []
        for comp, score in zip(components, scores):
            max_score = max_scores.get(comp, 100)
            normalized_scores.append((score / max_score) * 100)
        
        # Create radar chart
        ax = fig.add_subplot(111, polar=True)
        
        # Number of variables
        N = len(components)
        
        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Add scores
        normalized_scores += normalized_scores[:1]  # Close the loop
        
        # Draw the chart
        ax.plot(angles, normalized_scores, linewidth=2, linestyle='solid', color=self.colors['primary'])
        ax.fill(angles, normalized_scores, alpha=0.25, color=self.colors['primary'])
        
        # Add labels
        plt.xticks(angles[:-1], [c.replace('_', ' ').title() for c in components])
        
        # Add score labels
        for i, (angle, score) in enumerate(zip(angles[:-1], normalized_scores[:-1])):
            ax.text(angle, score + 10, f"{scores[i]:.1f}", 
                    ha='center', va='center', fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Set y-axis limits
        ax.set_ylim(0, 100)
        
        # Add grid lines and labels
        ax.set_rlabel_position(0)
        plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=8)
        plt.ylim(0, 100)
        
        # Add a title
        plt.title('Component Quality Scores', y=1.08)
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def visualize_skill_alignment(alignment_data, output_path=None):
    """Convenience function to visualize skill alignment results.
    
    Args:
        alignment_data (dict): Skill alignment results.
        output_path (str, optional): Path to save the visualization. Defaults to None.
    """
    visualizer = Visualizer()
    visualizer.visualize_skill_alignment(alignment_data, output_path)


def visualize_project_validation(validation_data, output_path=None):
    """Convenience function to visualize project validation results.
    
    Args:
        validation_data (dict): Project validation results.
        output_path (str, optional): Path to save the visualization. Defaults to None.
    """
    visualizer = Visualizer()
    visualizer.visualize_project_validation(validation_data, output_path)


def visualize_quality_scores(quality_data, output_path=None):
    """Convenience function to visualize quality scores.
    
    Args:
        quality_data (dict): Quality score data.
        output_path (str, optional): Path to save the visualization. Defaults to None.
    """
    visualizer = Visualizer()
    visualizer.visualize_quality_scores(quality_data, output_path)