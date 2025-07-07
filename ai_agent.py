import os
import sys
import json
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging
from datetime import datetime
import threading
import queue
import fitz  # PyMuPDF
import numpy as np
import traceback
import mimetypes
import chardet
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai
from resume_pipeline import ResumePipeline
from resume_intelligence.jd_generator import JobDescriptionGenerator

# --- API Key Configuration ---
# Primary: Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCb5B_wsF-91WRGlyd7N9C6DKqV37y3m-o")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Warning: GEMINI_API_KEY not found. LLM features may not work.")

# Fallback: OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        print("Warning: 'openai' library not installed. OpenAI fallback is disabled.")

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from resume_pipeline import ResumePipeline

# Configure logging with rotation
from logging.handlers import RotatingFileHandler
log_file = 'ai_agent.log'
max_bytes = 10 * 1024 * 1024  # 10MB
backup_count = 5

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)

class FileValidationError(Exception):
    """Custom exception for file validation errors."""
    pass

class AnalysisError(Exception):
    """Custom exception for analysis errors."""
    pass

def validate_file(file_path: Union[str, Path], allowed_extensions: List[str] = None, max_size_mb: int = 10) -> None:
    """Validate file existence, type, and size."""
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        raise FileValidationError(f"File does not exist: {file_path}")
    
    # Check file size
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        raise FileValidationError(f"File too large ({size_mb:.1f}MB). Maximum size is {max_size_mb}MB")
    
    # Check file type
    if allowed_extensions:
        if file_path.suffix.lower() not in allowed_extensions:
            raise FileValidationError(f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}")
    
    # Check if file is readable
    if not os.access(file_path, os.R_OK):
        raise FileValidationError(f"File is not readable: {file_path}")

def detect_encoding(file_path: Union[str, Path]) -> str:
    """Detect file encoding using chardet."""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def read_file_content(file_path: Union[str, Path]) -> str:
    """Read file content with proper encoding handling.
    
    Args:
        file_path (Union[str, Path]): Path to the file to read.
        
    Returns:
        str: File content.
        
    Raises:
        FileValidationError: If file cannot be read.
    """
    # Check if file is PDF
    if isinstance(file_path, Path) and file_path.suffix.lower() == '.pdf':
        try:
            # Read PDF using PyMuPDF
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text.strip()
        except Exception as e:
            logger.error(f"Error reading PDF file {file_path}: {str(e)}")
            raise FileValidationError(f"Failed to read PDF file: {str(e)}")
    # For non-PDF files
    try:
        # Detect encoding
        encoding = detect_encoding(file_path)
        if not encoding:
            encoding = 'utf-8'
        # Try reading with detected encoding
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read().strip()
        except UnicodeDecodeError:
            # Fallback to other encodings
            for enc in ['latin-1', 'cp1252', 'utf-16', 'ascii']:
                try:
                    with open(file_path, 'r', encoding=enc) as f:
                        return f.read().strip()
                except UnicodeDecodeError:
                    continue
            raise FileValidationError(f"Could not decode file {file_path} with any supported encoding")
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise FileValidationError(f"Failed to read file: {str(e)}")

class ResumeAIAgentGUI:
    """Interactive GUI for Resume AI Agent."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Resume Intelligence System")
        self.root.geometry("1400x900")
        
        # Initialize agent
        self.agent = ResumeAIAgent(output_dir="output")
        
        # Message queue for thread-safe updates
        self.message_queue = queue.Queue()
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Store generated JD
        self.generated_jd_text = None
        
        # Setup GUI
        self._setup_gui()
        
        # Start message processing
        self._process_messages()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _setup_gui(self):
        """Setup the GUI components in a single window."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(1, weight=3)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # --- Left Panel ---
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, sticky="nswe", padx=5)
        
        # --- Resume Analysis Section ---
        resume_analysis_frame = ttk.LabelFrame(left_panel, text="Resume Analysis", padding="10")
        resume_analysis_frame.pack(fill=tk.X, expand=True, pady=5)

        ttk.Label(resume_analysis_frame, text="Resume Files:").grid(row=0, column=0, columnspan=2, sticky=tk.W)
        self.resume_list = tk.Listbox(resume_analysis_frame, height=5)
        self.resume_list.grid(row=1, column=0, columnspan=2, sticky="ew")
        ttk.Button(resume_analysis_frame, text="Add Resume", command=self._add_resume).grid(row=2, column=0, sticky=tk.W)
        ttk.Button(resume_analysis_frame, text="Remove", command=self._remove_resume).grid(row=2, column=1, sticky=tk.E)

        self.jd_source = tk.StringVar(value="file")
        ttk.Label(resume_analysis_frame, text="Job Description Source:").grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(10,0))
        jd_source_frame = ttk.Frame(resume_analysis_frame)
        jd_source_frame.grid(row=4, column=0, columnspan=2, sticky="ew")
        ttk.Radiobutton(jd_source_frame, text="File", variable=self.jd_source, value="file").pack(side=tk.LEFT, padx=5)
        self.jd_path_entry = ttk.Entry(jd_source_frame)
        self.jd_path_entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(jd_source_frame, text="Browse", command=self._add_jd).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(resume_analysis_frame, text="Use Generated JD", variable=self.jd_source, value="generated").grid(row=5, column=0, columnspan=2, sticky=tk.W)

        ttk.Label(resume_analysis_frame, text="GitHub Usernames (optional):").grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=(10,0))
        self.github_list = tk.Listbox(resume_analysis_frame, height=3)
        self.github_list.grid(row=7, column=0, columnspan=2, sticky="ew")
        ttk.Button(resume_analysis_frame, text="Add User", command=self._add_github).grid(row=8, column=0, sticky=tk.W)
        ttk.Button(resume_analysis_frame, text="Remove", command=self._remove_github).grid(row=8, column=1, sticky=tk.E)

        self.analyze_button = ttk.Button(resume_analysis_frame, text="Run Analysis", command=self._run_analysis)
        self.analyze_button.grid(row=9, column=0, columnspan=2, pady=10)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(resume_analysis_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=10, column=0, columnspan=2, sticky="ew")

        # --- JD Generator Section ---
        jd_frame = ttk.LabelFrame(left_panel, text="Job Description Generator", padding="10")
        jd_frame.pack(fill=tk.X, expand=True, pady=5)
        
        ttk.Label(jd_frame, text="Role Title:").grid(row=0, column=0, sticky=tk.W)
        self.jd_role_title = ttk.Entry(jd_frame, width=40)
        self.jd_role_title.grid(row=0, column=1, sticky="ew")
        ttk.Label(jd_frame, text="Company (opt):").grid(row=1, column=0, sticky=tk.W)
        self.jd_company = ttk.Entry(jd_frame, width=40)
        self.jd_company.grid(row=1, column=1, sticky="ew")
        ttk.Label(jd_frame, text="Responsibilities:").grid(row=2, column=0, sticky=tk.NW)
        self.jd_responsibilities = tk.Text(jd_frame, height=4)
        self.jd_responsibilities.grid(row=2, column=1, sticky="ew")
        ttk.Label(jd_frame, text="Requirements:").grid(row=3, column=0, sticky=tk.NW)
        self.jd_requirements = tk.Text(jd_frame, height=4)
        self.jd_requirements.grid(row=3, column=1, sticky="ew")
        ttk.Label(jd_frame, text="Perks (opt):").grid(row=4, column=0, sticky=tk.NW)
        self.jd_perks = tk.Text(jd_frame, height=3)
        self.jd_perks.grid(row=4, column=1, sticky="ew")
        self.jd_generate_button = ttk.Button(jd_frame, text="Generate & Show JD", command=self._generate_jd)
        self.jd_generate_button.grid(row=5, column=1, sticky=tk.E, pady=5)
        jd_frame.columnconfigure(1, weight=1)

        # --- Right Panel (Chat/Results) ---
        right_panel = ttk.LabelFrame(main_frame, text="Analysis & Chat", padding="10")
        right_panel.grid(row=0, column=1, sticky="nswe", padx=5)
        right_panel.rowconfigure(0, weight=1)
        right_panel.columnconfigure(0, weight=1)

        self.chat_display = scrolledtext.ScrolledText(right_panel, wrap=tk.WORD)
        self.chat_display.grid(row=0, column=0, sticky="nswe")
        
        input_frame = ttk.Frame(right_panel)
        input_frame.grid(row=1, column=0, sticky="ew", pady=5)
        input_frame.columnconfigure(0, weight=1)
        self.input_field = ttk.Entry(input_frame)
        self.input_field.grid(row=0, column=0, sticky="ew")
        self.input_field.bind("<Return>", self._send_message)
        self.send_button = ttk.Button(input_frame, text="Send", command=self._send_message)
        self.send_button.grid(row=0, column=1, padx=(5,0))

    def _add_resume(self):
        """Add resume file(s) to the list."""
        try:
            files = filedialog.askopenfilenames(
                title="Select Resume Files",
                filetypes=[
                    ("PDF files", "*.pdf"),
                    ("Word files", "*.docx;*.doc"),
                    ("Text files", "*.txt"),
                    ("All files", "*.*")
                ]
            )
            for file in files:
                try:
                    validate_file(file, allowed_extensions=['.pdf', '.docx', '.doc', '.txt'])
                    self.resume_list.insert(tk.END, file)
                except FileValidationError as e:
                    messagebox.showerror("File Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add resume files: {str(e)}")
    
    def _remove_resume(self):
        """Remove selected resume from the list."""
        try:
            selection = self.resume_list.curselection()
            if selection:
                self.resume_list.delete(selection)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove resume: {str(e)}")
    
    def _add_jd(self):
        """Add job description file."""
        try:
            file = filedialog.askopenfilename(
                title="Select Job Description",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if file:
                validate_file(file, allowed_extensions=['.txt'])
                self.jd_path_entry.delete(0, tk.END)
                self.jd_path_entry.insert(0, file)
                self.jd_source.set("file")
        except FileValidationError as e:
            messagebox.showerror("File Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add job description: {str(e)}")
    
    def _add_github(self):
        """Add GitHub username."""
        try:
            username = tk.simpledialog.askstring("GitHub Username", "Enter GitHub username:")
            if username:
                if not username.strip():
                    raise ValueError("Username cannot be empty")
                self.github_list.insert(tk.END, username.strip())
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add GitHub username: {str(e)}")
    
    def _remove_github(self):
        """Remove selected GitHub username."""
        try:
            selection = self.github_list.curselection()
            if selection:
                self.github_list.delete(selection)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove GitHub username: {str(e)}")
    
    def _run_analysis(self):
        """Run the analysis in a separate thread."""
        try:
            if not self.resume_list.size():
                raise ValueError("Please add at least one resume file.")
            
            # Choose JD source
            jd_source = self.jd_source.get()
            if jd_source == "file":
                if not self.jd_path_entry.get():
                    raise ValueError("Please select a job description file.")
                jd_path = self.jd_path_entry.get()
                jd_text = None
            else:
                if not self.generated_jd_text:
                    raise ValueError("No generated job description available.")
                jd_path = None
                jd_text = self.generated_jd_text
            
            # Disable controls during analysis
            self._set_controls_state(tk.DISABLED)
            
            # Get all inputs
            resumes = list(self.resume_list.get(0, tk.END))
            github_usernames = list(self.github_list.get(0, tk.END))
            
            # Clear chat display
            self.chat_display.delete(1.0, tk.END)
            
            # Start analysis in thread pool
            future = self.thread_pool.submit(self._run_analysis_thread, resumes, jd_path, github_usernames, jd_text)
            future.add_done_callback(self._analysis_complete)
            
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            self._set_controls_state(tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start analysis: {str(e)}")
            self._set_controls_state(tk.NORMAL)
    
    def _set_controls_state(self, state: str):
        """Enable/disable GUI controls."""
        self.analyze_button.configure(state=state)
        self.send_button.configure(state=state)
        self.input_field.configure(state=state)
    
    def _run_analysis_thread(self, resume_paths: List[str], job_description_path: str, github_usernames: Optional[List[str]] = None, jd_text: Optional[str] = None):
        """Run analysis in background thread."""
        try:
            self._set_controls_state(tk.DISABLED)
            self.progress_var.set(10)
            
            # If jd_text is provided, write it to a temp file
            if jd_text:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tmp:
                    tmp.write(jd_text)
                    tmp_path = tmp.name
                job_description_path = tmp_path
            
            # Run analysis
            results = self.agent.run(resume_paths, job_description_path, github_usernames)
            
            # Save results to JSON file
            output_file = os.path.join(self.agent.output_dir, "analysis_results.json")
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)
            
            self.message_queue.put(("info", f"Analysis complete! Results saved to {output_file}"))
            for i, (res_path, res) in enumerate(results.items(), 1):
                self.progress_bar['variable'].set(self.progress_bar['variable'].get() + 70/len(results))
                if res['status'] == 'success':
                    report = res['report']
                    assessment = report.get("combined_assessment", {})
                    self.message_queue.put(("result", f"\nResume {i}: {res_path}"))
                    self.message_queue.put(("result", f"Score: {assessment.get('overall_score', 'N/A')}"))
                    self.message_queue.put(("result", f"Grade: {assessment.get('grade', 'N/A')}"))
                    
                    self.message_queue.put(("result", "\nKey Recommendations:"))
                    for rec in assessment.get("recommendations", [])[:3]:
                        self.message_queue.put(("result", f"- {rec}"))
                    
                    self.message_queue.put(("result", "\nNext Steps:"))
                    for step in assessment.get("next_steps", [])[:3]:
                        self.message_queue.put(("result", f"- {step}"))
                    
                    self.message_queue.put(("result", f"\nFull report saved to: {res.get('output_file', 'N/A')}"))
                    
                    # Add HR questions
                    self.message_queue.put(("result", "\nSuggested HR Questions:"))
                    for i, q in enumerate(res.get("hr_questions", []), 1):
                        self.message_queue.put(("result", f"{i}. {q}"))
                else:
                    self.message_queue.put(("error", f"Error analyzing {Path(res_path).name}: {res['error']}"))

        except Exception as e:
            logger.error(f"Analysis error: {str(e)}\n{traceback.format_exc()}")
            self.message_queue.put(("error", f"Error during analysis: {str(e)}"))
        finally:
            self._set_controls_state(tk.NORMAL)
            self.progress_var.set(0)
    
    def _analysis_complete(self, future):
        """Handle analysis completion."""
        try:
            future.result()  # This will raise any exceptions that occurred
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}\n{traceback.format_exc()}")
            messagebox.showerror("Analysis Error", f"Analysis failed: {str(e)}")
        finally:
            self._set_controls_state(tk.NORMAL)
            self.progress_var.set(0)
    
    def _send_message(self, event=None):
        """Send a message to the agent."""
        try:
            message = self.input_field.get().strip()
            if message:
                self._add_message(f"You: {message}", "user")
                self.input_field.delete(0, tk.END)
                
                # Process message in thread pool
                future = self.thread_pool.submit(self._process_message_thread, message)
                future.add_done_callback(self._message_complete)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send message: {str(e)}")
    
    def _process_message_thread(self, message: str):
        """Process user message in background thread."""
        try:
            # Add message to queue for thread-safe update
            self.message_queue.put(("agent", "Processing your question..."))
            
            # Get the last analysis results
            results = self.agent.get_last_results()
            
            # Generate response based on the message and results
            response = self.agent.generate_response(message, results)
            
            # Add response to queue
            self.message_queue.put(("agent", response))
            
        except Exception as e:
            logger.error(f"Message processing error: {str(e)}\n{traceback.format_exc()}")
            self.message_queue.put(("error", f"Error processing message: {str(e)}"))
    
    def _message_complete(self, future):
        """Handle message processing completion."""
        try:
            future.result()  # This will raise any exceptions that occurred
        except Exception as e:
            logger.error(f"Message processing failed: {str(e)}\n{traceback.format_exc()}")
            messagebox.showerror("Error", f"Failed to process message: {str(e)}")
    
    def _process_messages(self):
        try:
            while True:
                msg_type, message = self.message_queue.get_nowait()
                if msg_type == "jd_generated":
                    self.generated_jd_text = message
                    self.jd_source.set("generated")
                    self._show_jd_window(message)
                    messagebox.showinfo("JD Ready", "Generated JD is now set as the source for analysis.")
                elif msg_type == "jd_error":
                    messagebox.showerror("JD Generation Error", message)
                elif msg_type == "agent": self._add_message(f"Assistant: {message}", "agent")
                elif msg_type == "user": self._add_message(message, "user")
                elif msg_type == "info": self._add_message(message, "info")
                elif msg_type == "result": self._add_message(message, "result")
                else: self._add_message(f"Error: {message}", "error")
        except queue.Empty:
            pass
        self.root.after(100, self._process_messages)
    
    def _add_message(self, message, tag=None):
        self.chat_display.configure(state='normal')
        self.chat_display.insert(tk.END, message + "\n\n", tag)
    
    def _on_closing(self):
        """Handle window closing."""
        try:
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=False)
            
            # Close the window
            self.root.destroy()
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            self.root.destroy()

    def _generate_jd(self):
        role = self.jd_role_title.get().strip()
        resp = [line.strip() for line in self.jd_responsibilities.get("1.0", tk.END).splitlines() if line.strip()]
        reqs = [line.strip() for line in self.jd_requirements.get("1.0", tk.END).splitlines() if line.strip()]
        if not role or not resp or not reqs:
            messagebox.showerror("Input Error", "Role, Responsibilities, and Requirements are required.")
            return

        company = self.jd_company.get().strip() or None
        perks = [line.strip() for line in self.jd_perks.get("1.0", tk.END).splitlines() if line.strip()] or None
        
        self.jd_generate_button.config(state=tk.DISABLED)
        
        def run_jd():
            try:
                jd_text = self.agent.generate_job_description(role, resp, reqs, company, perks)
                self.message_queue.put(("jd_generated", jd_text))
            except Exception as e:
                self.message_queue.put(("jd_error", str(e)))
            finally:
                # This needs to be thread-safe if it modifies the GUI
                self.root.after(0, lambda: self.jd_generate_button.config(state=tk.NORMAL))
        
        threading.Thread(target=run_jd, daemon=True).start()

    def _show_jd_window(self, jd_text: str):
        """Show the generated job description in a new window."""
        jd_window = tk.Toplevel(self.root)
        jd_window.title("Generated Job Description")
        jd_window.geometry("800x600")
        
        jd_text_area = scrolledtext.ScrolledText(jd_window, wrap=tk.WORD, height=20, width=80)
        jd_text_area.pack(fill=tk.BOTH, expand=True)
        jd_text_area.insert(tk.END, jd_text)

class ResumeAIAgent:
    """Enhanced AI agent with chat capabilities."""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.pipeline = ResumePipeline(output_dir=str(self.output_dir))
        self.analysis_cache = {}
        self.last_results = None
        self.jd_generator = JobDescriptionGenerator()  # Add JD generator
    
    def get_last_results(self) -> Dict[str, Any]:
        """Get the last analysis results."""
        return self.last_results
    
    def generate_response(self, message: str, results: Dict[str, Any]) -> str:
        """Generate a response to a user message."""
        prompt = f"""Based on the following analysis results, answer the user's question:

User Question: {message}

Analysis Results:
{json.dumps(results, indent=2, cls=NumpyEncoder)}

Provide a clear, concise, and helpful response focusing on the specific question."""

        system_prompt = """You are an expert resume analyst and career advisor. 
Provide specific, actionable advice based on the analysis results.
Be professional but conversational in your responses."""

        response = self._call_llm(prompt, system_prompt)
        return response if response else "I apologize, but I couldn't generate a response at this time."

    def _call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Try Gemini first, fallback to OpenAI if it fails."""
        # Try Gemini
        if GEMINI_API_KEY:
            try:
                model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
                full_prompt = (system_prompt + "\n\n" if system_prompt else "") + prompt
                response = model.generate_content(full_prompt)
                if hasattr(response, 'text'):
                    return response.text.strip()
            except Exception as gemini_e:
                logger.warning(f"Gemini call failed: {gemini_e}. Falling back to OpenAI.")

        # Fallback to OpenAI
        if openai_client:
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=1500,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            except Exception as openai_e:
                logger.error(f"OpenAI call failed: {openai_e}")

        return "[LLM call failed. Please check API keys and configurations.]"

    def _analyze_resume_content(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Deep analysis of resume content using LLM."""
        prompt = f"""Analyze this resume against the job description and provide detailed insights:

Resume:
{resume_text}

Job Description:
{job_description}

Provide analysis in JSON format with the following structure:
{{
    "skill_match": {{
        "matching_skills": [],
        "missing_skills": [],
        "score": 0-100
    }},
    "experience_match": {{
        "relevant_experience": [],
        "gaps": [],
        "score": 0-100
    }},
    "education_match": {{
        "relevant_qualifications": [],
        "missing_qualifications": [],
        "score": 0-100
    }},
    "overall_assessment": {{
        "strengths": [],
        "weaknesses": [],
        "recommendations": [],
        "score": 0-100
    }}
}}"""

        system_prompt = """You are an expert resume analyzer. Provide detailed, objective analysis focusing on:
1. Skill alignment with job requirements
2. Experience relevance and gaps
3. Education and qualifications match
4. Overall fit and potential
Be specific and actionable in your recommendations."""

        analysis = self._call_llm(prompt, system_prompt)
        try:
            return json.loads(analysis) if analysis else {}
        except:
            logger.error("Failed to parse LLM analysis")
            return {}

    def _generate_hr_questions(self, job_description: str, resume_analysis: Dict[str, Any]) -> List[str]:
        """Generate targeted HR questions based on resume analysis."""
        prompt = f"""Generate 5 targeted HR interview questions based on this analysis:

Job Description:
{job_description}

Resume Analysis:
{json.dumps(resume_analysis, indent=2, cls=NumpyEncoder)}

Focus on:
1. Verifying key skills and experiences
2. Addressing potential gaps
3. Understanding career motivations
4. Assessing cultural fit
5. Exploring growth potential

Format each question with a brief explanation of why it's important."""

        questions = self._call_llm(prompt)
        return questions.split('\n') if questions else []

    def _optimize_job_description(self, jd: str) -> str:
        """Optimize job description for clarity and inclusivity."""
        prompt = f"""Optimize this job description for clarity, inclusivity, and effectiveness:

{jd}

Focus on:
1. Clear, specific requirements
2. Inclusive language
3. Structured format
4. Key responsibilities and qualifications
5. Growth opportunities

Provide the optimized version."""

        return self._call_llm(prompt) or jd

    def _generate_comprehensive_report(self, resume_path: str, job_description: str, 
                                    pipeline_results: Dict[str, Any], 
                                    llm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive report combining pipeline and LLM analysis."""
        report = {
            "resume_file": os.path.basename(resume_path),
            "job_description": job_description,
            "pipeline_analysis": pipeline_results,
            "llm_analysis": llm_analysis,
            "combined_assessment": {
                "overall_score": 0,
                "grade": "F",
                "strengths": [],
                "weaknesses": [],
                "recommendations": []
            }
        }
        
        # Calculate combined score
        pipeline_score = pipeline_results.get("overall_score", 0)
        llm_score = llm_analysis.get("overall_score", 0)
        report["combined_assessment"]["overall_score"] = (pipeline_score * 0.7) + (llm_score * 0.3)
        
        # Determine grade
        score = report["combined_assessment"]["overall_score"]
        if score >= 90: grade = "A+"
        elif score >= 80: grade = "A"
        elif score >= 70: grade = "B+"
        elif score >= 60: grade = "B"
        elif score >= 50: grade = "C"
        else: grade = "F"
        report["combined_assessment"]["grade"] = grade
        
        # Combine recommendations safely
        pipeline_recs = pipeline_results.get("recommendations", [])
        llm_recs = llm_analysis.get("recommendations", [])
        
        # Convert all recommendations to strings
        def convert_to_strings(items):
            if isinstance(items, dict):
                return [str(v) for v in items.values()]
            elif isinstance(items, list):
                return [str(item) for item in items]
            else:
                return [str(items)]
        
        pipeline_recs = convert_to_strings(pipeline_recs)
        llm_recs = convert_to_strings(llm_recs)
        
        # Combine and deduplicate recommendations
        all_recs = pipeline_recs + llm_recs
        report["combined_assessment"]["recommendations"] = list(dict.fromkeys(all_recs))
        
        # Add strengths and weaknesses (ensuring they're strings)
        report["combined_assessment"]["strengths"] = convert_to_strings(pipeline_results.get("strengths", []))
        report["combined_assessment"]["weaknesses"] = convert_to_strings(pipeline_results.get("weaknesses", []))
        
        return report

    def run(self, resume_paths: List[str], job_description_path: str, 
            github_usernames: Optional[List[str]] = None) -> Dict[str, Any]:
        """Main execution method with fallback mechanisms."""
        logger.info("Starting Resume AI Agent analysis")
        
        # Read and optimize job description
        job_description = read_file_content(job_description_path)
        optimized_jd = self._optimize_job_description(job_description)
        
        results = {}
        for idx, resume_path in enumerate(resume_paths):
            logger.info(f"Processing resume {idx+1}: {resume_path}")
            
            try:
                # Run pipeline analysis
                github_username = github_usernames[idx] if github_usernames and idx < len(github_usernames) else None
                contact_info = {"github": github_username} if github_username else None
                
                pipeline_results = self.pipeline.run_pipeline(
                    resume_path=resume_path,
                    job_description_path=job_description_path,
                    github_username=github_username,
                    contact_info=contact_info
                )
                
                # Read resume content for LLM analysis
                resume_text = read_file_content(resume_path)
                
                # Perform LLM analysis
                llm_analysis = self._analyze_resume_content(resume_text, optimized_jd)
                
                # Generate comprehensive report
                report = self._generate_comprehensive_report(
                    resume_path, optimized_jd, pipeline_results, llm_analysis
                )
                
                # Generate HR questions
                hr_questions = self._generate_hr_questions(optimized_jd, llm_analysis)
                
                # Save results
                output_file = self.output_dir / f"analysis_{Path(resume_path).stem}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "report": report,
                        "hr_questions": hr_questions
                    }, f, indent=2, cls=NumpyEncoder)
                
                results[resume_path] = {
                    "status": "success",
                    "report": report,
                    "hr_questions": hr_questions,
                    "output_file": str(output_file)
                }
                
            except Exception as e:
                logger.error(f"Error processing resume {resume_path}: {str(e)}")
                results[resume_path] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Store results for chat
        self.last_results = results
        return results

    def generate_job_description(self, role_title, responsibilities, requirements, company=None, perks=None):
        """Generate a job description using the integrated JD generator."""
        return self.jd_generator.generate_jd(role_title, responsibilities, requirements, company, perks)

def main():
    """Main entry point for the GUI application."""
    root = tk.Tk()
    app = ResumeAIAgentGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 