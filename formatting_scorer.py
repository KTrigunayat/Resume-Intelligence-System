import fitz  # PyMuPDF
import re
import json
from typing import Dict, List, Tuple, Any
from collections import Counter
import logging

class FormattingScorer:
    """
    Formatting Scorer: Enforces clean, consistent résumé layout—bullets, fonts, 
    headers, section lengths—and penalizes "walls of text" or erratic styling.
    """
    
    def __init__(self, config_path: str = None):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.max_score = 20
        
    def _load_config(self, config_path: str) -> Dict:
        """Load formatting rules configuration"""
        default_config = {
            "font_consistency_weight": 0.25,
            "bullet_consistency_weight": 0.20,
            "section_length_weight": 0.20,
            "margin_consistency_weight": 0.15,
            "text_density_weight": 0.20,
            "max_fonts_allowed": 3,
            "min_margin": 36,  # points
            "max_line_length": 100,  # characters
            "bullet_patterns": ["•", "▪", "▫", "◦", "‣", "-", "*"],
            "section_min_length": 2,
            "section_max_length": 8
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                default_config.update(custom_config)
            except Exception as e:
                self.logger.warning(f"Could not load config from {config_path}: {e}")
                
        return default_config
    
    def analyze_formatting(self, file_path: str, sections: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Analyze resume formatting and return comprehensive scoring
        
        Args:
            file_path: Path to the resume file
            sections: Optional parsed sections for additional analysis
            
        Returns:
            Dictionary containing formatting scores and detailed analysis
        """
        try:
            if file_path.lower().endswith('.pdf'):
                return self._analyze_pdf_formatting(file_path, sections)
            else:
                # For non-PDF files, provide basic text-based analysis
                return self._analyze_text_formatting(file_path, sections)
                
        except Exception as e:
            self.logger.error(f"Error analyzing formatting for {file_path}: {e}")
            return self._get_default_score()
    
    def _analyze_pdf_formatting(self, file_path: str, sections: Dict[str, str] = None) -> Dict[str, Any]:
        """Analyze PDF formatting using PyMuPDF"""
        doc = fitz.open(file_path)
        
        font_analysis = self._analyze_fonts(doc)
        bullet_analysis = self._analyze_bullets(doc)
        margin_analysis = self._analyze_margins(doc)
        text_density_analysis = self._analyze_text_density(doc)
        section_analysis = self._analyze_section_lengths(sections) if sections else {}
        
        doc.close()
        
        # Calculate individual scores
        font_score = self._score_font_consistency(font_analysis)
        bullet_score = self._score_bullet_consistency(bullet_analysis)
        margin_score = self._score_margin_consistency(margin_analysis)
        text_density_score = self._score_text_density(text_density_analysis)
        section_score = self._score_section_lengths(section_analysis)
        
        # Calculate weighted total score
        total_score = (
            font_score * self.config["font_consistency_weight"] +
            bullet_score * self.config["bullet_consistency_weight"] +
            margin_score * self.config["margin_consistency_weight"] +
            text_density_score * self.config["text_density_weight"] +
            section_score * self.config["section_length_weight"]
        ) * self.max_score
        
        return {
            "total_score": min(self.max_score, max(0, total_score)),
            "max_score": self.max_score,
            "detailed_scores": {
                "font_consistency": font_score,
                "bullet_consistency": bullet_score,
                "margin_consistency": margin_score,
                "text_density": text_density_score,
                "section_lengths": section_score
            },
            "analysis_details": {
                "fonts": font_analysis,
                "bullets": bullet_analysis,
                "margins": margin_analysis,
                "text_density": text_density_analysis,
                "sections": section_analysis
            },
            "recommendations": self._generate_recommendations({
                "font_score": font_score,
                "bullet_score": bullet_score,
                "margin_score": margin_score,
                "text_density_score": text_density_score,
                "section_score": section_score
            })
        }
    
    def _analyze_fonts(self, doc: fitz.Document) -> Dict[str, Any]:
        """Analyze font usage throughout the document"""
        fonts = []
        font_sizes = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            fonts.append(span["font"])
                            font_sizes.append(span["size"])
        
        font_counter = Counter(fonts)
        size_counter = Counter([round(size) for size in font_sizes])
        
        return {
            "unique_fonts": len(font_counter),
            "font_distribution": dict(font_counter),
            "unique_sizes": len(size_counter),
            "size_distribution": dict(size_counter),
            "most_common_font": font_counter.most_common(1)[0] if font_counter else None,
            "most_common_size": size_counter.most_common(1)[0] if size_counter else None
        }
    
    def _analyze_bullets(self, doc: fitz.Document) -> Dict[str, Any]:
        """Analyze bullet point usage and consistency"""
        bullet_patterns = []
        bullet_lines = 0
        total_lines = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_lines = page.get_text().split('\n')
            
            for line in text_lines:
                total_lines += 1
                line = line.strip()
                if line:
                    for bullet in self.config["bullet_patterns"]:
                        if line.startswith(bullet):
                            bullet_patterns.append(bullet)
                            bullet_lines += 1
                            break
        
        bullet_counter = Counter(bullet_patterns)
        
        return {
            "bullet_lines": bullet_lines,
            "total_lines": total_lines,
            "bullet_usage_ratio": bullet_lines / max(total_lines, 1),
            "unique_bullet_types": len(bullet_counter),
            "bullet_distribution": dict(bullet_counter),
            "most_common_bullet": bullet_counter.most_common(1)[0] if bullet_counter else None
        }
    
    def _analyze_margins(self, doc: fitz.Document) -> Dict[str, Any]:
        """Analyze margin consistency"""
        left_margins = []
        right_margins = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_rect = page.rect
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "bbox" in block:
                    bbox = block["bbox"]
                    left_margins.append(bbox[0])
                    right_margins.append(page_rect.width - bbox[2])
        
        return {
            "left_margin_consistency": len(set([round(m) for m in left_margins])) if left_margins else 0,
            "right_margin_consistency": len(set([round(m) for m in right_margins])) if right_margins else 0,
            "avg_left_margin": sum(left_margins) / len(left_margins) if left_margins else 0,
            "avg_right_margin": sum(right_margins) / len(right_margins) if right_margins else 0,
            "min_margin": min(min(left_margins, default=0), min(right_margins, default=0))
        }
    
    def _analyze_text_density(self, doc: fitz.Document) -> Dict[str, Any]:
        """Analyze text density and identify walls of text"""
        long_lines = 0
        total_lines = 0
        paragraph_lengths = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Analyze line lengths
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    total_lines += 1
                    if len(line) > self.config["max_line_length"]:
                        long_lines += 1
            
            # Analyze paragraph lengths
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                para = para.strip()
                if para:
                    paragraph_lengths.append(len(para.split()))
        
        return {
            "long_lines": long_lines,
            "total_lines": total_lines,
            "long_line_ratio": long_lines / max(total_lines, 1),
            "avg_paragraph_length": sum(paragraph_lengths) / len(paragraph_lengths) if paragraph_lengths else 0,
            "max_paragraph_length": max(paragraph_lengths) if paragraph_lengths else 0,
            "wall_of_text_count": sum(1 for length in paragraph_lengths if length > 100)
        }
    
    def _analyze_section_lengths(self, sections: Dict[str, str]) -> Dict[str, Any]:
        """Analyze section length consistency"""
        if not sections:
            return {}
        
        section_lengths = {}
        for section_name, content in sections.items():
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            section_lengths[section_name] = len(lines)
        
        lengths = list(section_lengths.values())
        
        return {
            "section_lengths": section_lengths,
            "avg_section_length": sum(lengths) / len(lengths) if lengths else 0,
            "min_section_length": min(lengths) if lengths else 0,
            "max_section_length": max(lengths) if lengths else 0,
            "sections_too_short": sum(1 for length in lengths if length < self.config["section_min_length"]),
            "sections_too_long": sum(1 for length in lengths if length > self.config["section_max_length"])
        }
    
    def _score_font_consistency(self, font_analysis: Dict[str, Any]) -> float:
        """Score font consistency (0-1)"""
        if not font_analysis:
            return 0.5
        
        unique_fonts = font_analysis.get("unique_fonts", 0)
        unique_sizes = font_analysis.get("unique_sizes", 0)
        
        # Penalize too many fonts
        font_penalty = max(0, unique_fonts - self.config["max_fonts_allowed"]) * 0.2
        size_penalty = max(0, unique_sizes - 5) * 0.1  # Allow up to 5 different sizes
        
        score = 1.0 - font_penalty - size_penalty
        return max(0, min(1, score))
    
    def _score_bullet_consistency(self, bullet_analysis: Dict[str, Any]) -> float:
        """Score bullet point consistency (0-1)"""
        if not bullet_analysis:
            return 0.5
        
        unique_bullets = bullet_analysis.get("unique_bullet_types", 0)
        usage_ratio = bullet_analysis.get("bullet_usage_ratio", 0)
        
        # Reward consistent bullet usage
        consistency_score = 1.0 if unique_bullets <= 2 else max(0, 1.0 - (unique_bullets - 2) * 0.3)
        usage_score = min(1.0, usage_ratio * 2)  # Reward appropriate bullet usage
        
        return (consistency_score + usage_score) / 2
    
    def _score_margin_consistency(self, margin_analysis: Dict[str, Any]) -> float:
        """Score margin consistency (0-1)"""
        if not margin_analysis:
            return 0.5
        
        left_consistency = margin_analysis.get("left_margin_consistency", 0)
        right_consistency = margin_analysis.get("right_margin_consistency", 0)
        min_margin = margin_analysis.get("min_margin", 0)
        
        # Reward consistent margins
        consistency_score = 1.0 if left_consistency <= 2 and right_consistency <= 2 else 0.5
        margin_score = 1.0 if min_margin >= self.config["min_margin"] else min_margin / self.config["min_margin"]
        
        return (consistency_score + margin_score) / 2
    
    def _score_text_density(self, density_analysis: Dict[str, Any]) -> float:
        """Score text density and penalize walls of text (0-1)"""
        if not density_analysis:
            return 0.5
        
        long_line_ratio = density_analysis.get("long_line_ratio", 0)
        wall_count = density_analysis.get("wall_of_text_count", 0)
        avg_para_length = density_analysis.get("avg_paragraph_length", 0)
        
        # Penalize long lines and walls of text
        line_score = max(0, 1.0 - long_line_ratio * 2)
        wall_penalty = min(1.0, wall_count * 0.3)
        para_score = 1.0 if avg_para_length <= 50 else max(0, 1.0 - (avg_para_length - 50) / 100)
        
        return max(0, (line_score + para_score) / 2 - wall_penalty)
    
    def _score_section_lengths(self, section_analysis: Dict[str, Any]) -> float:
        """Score section length appropriateness (0-1)"""
        if not section_analysis:
            return 0.5
        
        too_short = section_analysis.get("sections_too_short", 0)
        too_long = section_analysis.get("sections_too_long", 0)
        total_sections = len(section_analysis.get("section_lengths", {}))
        
        if total_sections == 0:
            return 0.5
        
        penalty = (too_short + too_long) / total_sections
        return max(0, 1.0 - penalty)
    
    def _analyze_text_formatting(self, file_path: str, sections: Dict[str, str] = None) -> Dict[str, Any]:
        """Basic text-based formatting analysis for non-PDF files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            return self._get_default_score()
        
        # Basic text analysis
        lines = content.split('\n')
        bullet_count = sum(1 for line in lines if any(line.strip().startswith(bullet) for bullet in self.config["bullet_patterns"]))
        long_lines = sum(1 for line in lines if len(line) > self.config["max_line_length"])
        
        bullet_score = min(1.0, bullet_count / max(len(lines), 1) * 5)
        line_length_score = max(0, 1.0 - long_lines / max(len(lines), 1))
        section_score = self._score_section_lengths(self._analyze_section_lengths(sections))
        
        total_score = (bullet_score + line_length_score + section_score) / 3 * self.max_score
        
        return {
            "total_score": min(self.max_score, max(0, total_score)),
            "max_score": self.max_score,
            "detailed_scores": {
                "bullet_consistency": bullet_score,
                "line_length": line_length_score,
                "section_lengths": section_score
            },
            "analysis_details": {
                "bullet_lines": bullet_count,
                "long_lines": long_lines,
                "total_lines": len(lines)
            },
            "recommendations": self._generate_recommendations({
                "bullet_score": bullet_score,
                "line_length_score": line_length_score,
                "section_score": section_score
            })
        }
    
    def _generate_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """Generate actionable formatting recommendations"""
        recommendations = []
        
        if scores.get("font_score", 1) < 0.7:
            recommendations.append("Reduce font variety - use maximum 2-3 different fonts")
            recommendations.append("Maintain consistent font sizes for similar content types")
        
        if scores.get("bullet_score", 1) < 0.7:
            recommendations.append("Use consistent bullet points throughout the resume")
            recommendations.append("Consider using bullet points for achievements and responsibilities")
        
        if scores.get("margin_score", 1) < 0.7:
            recommendations.append("Ensure consistent margins throughout the document")
            recommendations.append("Maintain adequate white space (at least 0.5 inch margins)")
        
        if scores.get("text_density_score", 1) < 0.7:
            recommendations.append("Break up large blocks of text into smaller paragraphs")
            recommendations.append("Keep lines under 100 characters for better readability")
            recommendations.append("Use white space effectively to improve visual appeal")
        
        if scores.get("section_score", 1) < 0.7:
            recommendations.append("Balance section lengths - avoid very short or very long sections")
            recommendations.append("Ensure each section has sufficient content (2+ lines)")
        
        if not recommendations:
            recommendations.append("Excellent formatting! Your resume has clean, professional layout.")
        
        return recommendations
    
    def _get_default_score(self) -> Dict[str, Any]:
        """Return default score when analysis fails"""
        return {
            "total_score": self.max_score * 0.5,
            "max_score": self.max_score,
            "detailed_scores": {},
            "analysis_details": {},
            "recommendations": ["Could not analyze formatting - ensure file is accessible and properly formatted"]
        }