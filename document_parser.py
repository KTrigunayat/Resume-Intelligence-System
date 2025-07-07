#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Document Parser for Resume Intelligence System

This module provides functionality to parse PDF, DOCX, and TXT documents
using PyMuPDF, python-docx libraries, and built-in file operations.
"""

import os
from pathlib import Path

import fitz  # PyMuPDF
import docx


class DocumentParser:
    """Parser for PDF, DOCX, and TXT documents."""
    
    def __init__(self):
        """Initialize the document parser."""
        self.supported_extensions = {
            '.pdf': self._parse_pdf,
            '.docx': self._parse_docx,
            '.txt': self._parse_txt
        }
    
    def parse(self, file_path):
        """Parse a document and return its text content.
        
        Args:
            file_path (str): Path to the document file.
            
        Returns:
            str: Extracted text from the document.
            
        Raises:
            ValueError: If the file format is not supported.
            FileNotFoundError: If the file does not exist.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = file_path.suffix.lower()
        if file_ext not in self.supported_extensions:
            raise ValueError(
                f"Unsupported file format: {file_ext}. "
                f"Supported formats: {', '.join(self.supported_extensions.keys())}"
            )
        
        # Call the appropriate parser based on file extension
        return self.supported_extensions[file_ext](file_path)
    
    def _parse_pdf(self, file_path):
        """Parse a PDF document using PyMuPDF.
        
        Args:
            file_path (Path): Path to the PDF file.
            
        Returns:
            str: Extracted text from the PDF.
        """
        text = ""
        try:
            with fitz.open(file_path) as doc:
                for page in doc:
                    # Get regular text
                    text += page.get_text()
                    
                    # Extract links explicitly
                    links = page.get_links()
                    for link in links:
                        if "uri" in link:
                            # Add the link URL directly to the text to ensure it's captured
                            text += f"\n{link['uri']}\n"
        except Exception as e:
            raise RuntimeError(f"Error parsing PDF: {e}")
        
        return text
    
    def _parse_docx(self, file_path):
        """Parse a DOCX document using python-docx.
        
        Args:
            file_path (Path): Path to the DOCX file.
            
        Returns:
            str: Extracted text from the DOCX.
        """
        text = ""
        try:
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            raise RuntimeError(f"Error parsing DOCX: {e}")
        
        return text
    
    def _parse_txt(self, file_path):
        """Parse a plain text file.
        
        Args:
            file_path (Path): Path to the TXT file.
            
        Returns:
            str: Extracted text from the TXT file.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except UnicodeDecodeError:
            # Try with a different encoding if UTF-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    text = file.read()
            except Exception as e:
                raise RuntimeError(f"Error parsing TXT file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error parsing TXT file: {e}")
        
        return text
    
    def parse_document(self, file_path):
        """Alias for parse() method to maintain backward compatibility.
        
        Args:
            file_path (str): Path to the document file.
            
        Returns:
            str: Extracted text from the document.
        """
        return self.parse(file_path)