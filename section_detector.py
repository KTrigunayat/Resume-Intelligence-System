#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Section & Structure Detector for Resume Intelligence System

This module identifies and extracts standard resume sections (education, experience, skills, etc.)
using pattern matching and text analysis.
"""

import json
import re
from pathlib import Path


class SectionDetector:
    """Detector for identifying and extracting resume sections."""
    
    def __init__(self):
        """Initialize the section detector."""
        # Common section headers in resumes
        self.section_patterns = {
            'Summary': [r'(?i)\b(summary|profile|objective|about me|professional summary|career objective)\b'],
            'Education': [r'(?i)\b(education|academic|degree|university|college|educational background|academic qualifications)\b'],
            'Work Experience': [r'(?i)\b(experience|work|employment|job|career|professional|work history|professional experience|employment history)\b'],
            'Skills': [r'(?i)\b(skills|expertise|competencies|proficiencies|technical|technologies|technical skills|core competencies|key skills)\b'],
            'Projects': [r'(?i)\b(projects|portfolio|works|assignments|personal projects|project experience|key projects|technical projects|development projects)\b'],
            'Certifications': [r'(?i)\b(certifications|certificates|credentials|qualifications|professional certifications)\b'],
            'Languages': [r'(?i)\b(languages|language proficiency|spoken languages)\b'],
            'Interests': [r'(?i)\b(interests|hobbies|activities|extracurricular)\b'],
            'References': [r'(?i)\b(references|referees|professional references)\b'],
            'Publications': [r'(?i)\b(publications|papers|articles|research|research papers)\b'],
            'Awards': [r'(?i)\b(awards|honors|achievements|recognitions|accomplishments)\b'],
            'Volunteer': [r'(?i)\b(volunteer|community|service|volunteer experience|community service)\b']
        }
    
    def detect_sections(self, text):
        """Detect and extract sections from resume text.
        
        Args:
            text (str): The preprocessed resume text.
            
        Returns:
            dict: Dictionary mapping section names to their content.
        """
        # Find potential section headers
        potential_headers = self._find_potential_headers(text)
        
        # Print debug information about found headers
        print(f"Found {len(potential_headers)} potential section headers:")
        for section_name, start_idx, pattern in potential_headers:
            print(f"  - '{section_name}' at position {start_idx}, pattern: '{pattern}'")
        
        # If no headers found, try to extract sections based on common patterns
        if not potential_headers:
            print("No section headers found. Attempting to extract sections based on common patterns.")
            return self._extract_common_sections(text)
        
        # Extract sections based on identified headers
        sections = self._extract_sections(text, potential_headers)
        
        # Print debug information about extracted sections
        print(f"Extracted {len(sections)} sections:")
        for section_name, content in sections.items():
            content_preview = content[:50].replace('\n', ' ') + '...' if len(content) > 50 else content
            print(f"  - '{section_name}' with {len(content)} characters: '{content_preview}'")
        
        return sections
    
    def _preprocess_text(self, text):
        """Preprocess the resume text.
        
        Args:
            text (str): The raw resume text.
            
        Returns:
            str: Preprocessed text.
        """
        # Replace multiple newlines with double newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        
        # Remove excessive spaces
        text = re.sub(r' {2,}', ' ', text)
        
        return text
    
    def _find_potential_headers(self, text):
        """Find potential section headers in the text.
        
        Args:
            text (str): The preprocessed resume text.
            
        Returns:
            list: List of tuples (section_name, start_index, pattern_matched).
        """
        potential_headers = []
        
        # Split text into lines
        lines = text.split('\n')
        
        # Track current position in text
        current_pos = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                current_pos += 1  # Account for newline
                continue
            
            # Check if line matches any section pattern
            for section_name, patterns in self.section_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        # Enhanced header detection criteria
                        is_header = False
                        
                        # Standard header formats
                        if len(line) < 50 and (line.endswith(':') or line.isupper() or line.istitle()):
                            is_header = True
                        
                        # Check for standalone words that are likely headers
                        elif len(line.split()) <= 3 and len(line) < 30:
                            # Check if the line is followed by a blank line or bullet points
                            if current_pos + len(line) + 1 < len(text):
                                next_line = text[current_pos + len(line) + 1:].split('\n', 1)[0].strip()
                                if not next_line or next_line.startswith(('•', '-', '*', '\t', '    ')):
                                    is_header = True
                        
                        # Check for centered text (potential header)
                        elif line.strip() == line and len(line) < 30:
                            surrounding_lines = [l.strip() for l in lines[max(0, lines.index(line)-2):min(len(lines), lines.index(line)+3)]]
                            if all(len(l) < len(line) or not l for l in surrounding_lines if l != line):
                                is_header = True
                                
                        # Check for lines with formatting that suggests a header
                        # (e.g., all caps, title case, or followed by a line of dashes/underscores)
                        elif line.isupper() or (line.istitle() and len(line.split()) <= 4):
                            is_header = True
                        elif current_pos + len(line) + 1 < len(text):
                            next_line = text[current_pos + len(line) + 1:].split('\n', 1)[0].strip()
                            if next_line and all(c in '-_=' for c in next_line):
                                is_header = True
                        
                        if is_header:
                            # Use the full section name from the dictionary, not a truncated version
                            potential_headers.append((section_name, current_pos, line))
                            break
            
            current_pos += len(line) + 1  # +1 for newline
        
        # Sort headers by position in text
        potential_headers.sort(key=lambda x: x[1])
        
        return potential_headers
    
    def _extract_sections(self, text, headers):
        """Extract content between identified headers.
        
        Args:
            text (str): The preprocessed resume text.
            headers (list): List of tuples (section_name, start_index, pattern_matched).
            
        Returns:
            dict: Dictionary of extracted sections.
        """
        sections = {}
        
        # Extract content between headers
        for i, (section_name, start_idx, pattern) in enumerate(headers):
            # Find the end of the current section (start of next section or end of text)
            end_idx = headers[i+1][1] if i < len(headers) - 1 else len(text)
            
            # Extract content
            content = text[start_idx:end_idx].strip()
            
            # Remove the header line from the content
            content_lines = content.split('\n')
            if content_lines and content_lines[0].strip() == pattern.strip():
                content = '\n'.join(content_lines[1:]).strip()
            
            # Print debug information
            print(f"Extracted section: '{section_name}' with {len(content)} characters")
            print(f"Content preview: '{content[:50]}...'")
            
            sections[section_name] = content
        
        return sections
    
    def _extract_common_sections(self, text):
        """Extract sections based on common patterns when headers are not found.
        
        Args:
            text (str): The preprocessed resume text.
            
        Returns:
            dict: Dictionary of extracted sections.
        """
        sections = {}
        
        # Split text into lines
        lines = text.split('\n')
        
        # Initialize variables
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line matches any section pattern
            found_section = False
            for section_name, patterns in self.section_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        # If we were in a section, save it
                        if current_section:
                            sections[current_section] = '\n'.join(current_content).strip()
                        
                        # Start new section
                        current_section = section_name
                        current_content = []
                        found_section = True
                        break
                if found_section:
                    break
            
            # If no section found, add line to current section
            if not found_section and current_section:
                current_content.append(line)
        
        # Save the last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def save_sections(self, sections, output_path):
        """Save extracted sections to a JSON file.
        
        Args:
            sections (dict): Dictionary of extracted sections.
            output_path (str): Path to save the JSON file.
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sections, f, indent=2, ensure_ascii=False)
    
    def load_sections(self, input_path):
        """Load sections from a JSON file.
        
        Args:
            input_path (str): Path to the JSON file.
            
        Returns:
            dict: Dictionary of loaded sections.
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def analyze_structure(self, sections):
        """Analyze the structure of extracted sections.
        
        Args:
            sections (dict): Dictionary of extracted sections.
            
        Returns:
            dict: Analysis results.
        """
        analysis = {
            'total_sections': len(sections),
            'section_names': list(sections.keys()),
            'section_lengths': {name: len(content) for name, content in sections.items()},
            'has_summary': 'Summary' in sections,
            'has_education': 'Education' in sections,
            'has_experience': 'Work Experience' in sections,
            'has_skills': 'Skills' in sections
        }
        
        return analysis