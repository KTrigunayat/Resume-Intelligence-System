import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
from datetime import datetime, timedelta
import uuid
import schedule
import threading
import queue
from dataclasses import dataclass, field

# Import system components
from resume_intelligence.github_skill_predictor import GitHubTechnicalSkillPredictor
from resume_intelligence.multi_platform_scorer import MultiPlatformWeightedScoreEngine
from resume_intelligence.ai_prediction_engine import AIPredictionEngine
from resume_intelligence.explainability_layer import ExplainabilityReportingLayer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class WorkflowTask:
    """Represents a task in the workflow system"""
    task_id: str
    task_type: str  # 'rescore', 'batch_prediction', 'model_training', etc.
    parameters: Dict[str, Any]
    status: str = "pending"  # pending, running, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None


class WorkflowOrchestrator:
    """Orchestrates workflows for periodic re-scoring and batch predictions"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the Workflow Orchestrator"""
        self.output_dir = Path(output_dir) if output_dir else Path("./workflow_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.github_predictor = GitHubTechnicalSkillPredictor()
        self.multi_platform_scorer = MultiPlatformWeightedScoreEngine()
        self.ai_prediction_engine = AIPredictionEngine()
        self.explainability_layer = ExplainabilityReportingLayer(output_dir=str(self.output_dir))
        
        # Task storage
        self.tasks: Dict[str, WorkflowTask] = {}
        
        # Task queue for worker threads
        self.task_queue = queue.Queue()
        
        # Worker threads
        self.worker_threads: List[threading.Thread] = []
        self.stop_event = threading.Event()
        
        # Schedule storage
        self.schedules: Dict[str, Dict[str, Any]] = {}
        self.schedule_thread: Optional[threading.Thread] = None
    
    def start_workers(self, num_workers: int = 2):
        """Start worker threads to process tasks"""
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
        
        logger.info(f"Started {num_workers} worker threads")
    
    def _worker_loop(self, worker_id: int):
        """Worker thread loop to process tasks from the queue"""
        logger.info(f"Worker {worker_id} started")
        
        while not self.stop_event.is_set():
            try:
                # Get task from queue with timeout
                try:
                    task_id = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Get task from storage
                task = self.tasks.get(task_id)
                if not task:
                    logger.warning(f"Worker {worker_id}: Task {task_id} not found in storage")
                    self.task_queue.task_done()
                    continue
                
                # Process task
                logger.info(f"Worker {worker_id}: Processing task {task_id} ({task.task_type})")
                task.status = "running"
                task.started_at = datetime.now()
                
                try:
                    if task.task_type == "rescore":
                        self._process_rescore_task(task)
                    elif task.task_type == "batch_prediction":
                        self._process_batch_prediction_task(task)
                    elif task.task_type == "model_training":
                        self._process_model_training_task(task)
                    else:
                        raise ValueError(f"Unknown task type: {task.task_type}")
                    
                    task.status = "completed"
                    task.completed_at = datetime.now()
                    logger.info(f"Worker {worker_id}: Completed task {task_id}")
                    
                except Exception as e:
                    logger.error(f"Worker {worker_id}: Error processing task {task_id}: {e}")
                    task.status = "failed"
                    task.error = str(e)
                    task.completed_at = datetime.now()
                
                # Mark task as done in queue
                self.task_queue.task_done()
                
                # Save task result
                self._save_task_result(task)
                
            except Exception as e:
                logger.error(f"Worker {worker_id}: Unexpected error: {e}")
        
        logger.info(f"Worker {worker_id} stopped")
    
    def _process_rescore_task(self, task: WorkflowTask):
        """Process a re-scoring task"""
        # Extract parameters
        candidate_id = task.parameters.get("candidate_id")
        profiles = task.parameters.get("profiles", {})
        weight_config = task.parameters.get("weight_config")
        
        if not candidate_id:
            raise ValueError("Missing candidate_id parameter")
        
        # Initialize platform scores
        platform_scores = {}
        
        # Process GitHub profile if available
        github_profile = profiles.get("github")
        if github_profile:
            github_result = self.github_predictor.analyze_github_profile(
                github_profile.get("username"),
                access_token=github_profile.get("access_token")
            )
            platform_scores["github"] = {
                "normalized_score": github_result.technical_skill_score,
                "raw_score": github_result.technical_skill_score,
                "grade": github_result.grade,
                "metrics": {
                    "code_quality": github_result.code_quality_score,
                    "complexity": github_result.complexity_score,
                    "test_coverage": github_result.test_coverage_score,
                    "originality": github_result.originality_percentage
                }
            }
        
        # Process other platforms (placeholder)
        # Similar to the API layer implementation
        
        # Apply weight configuration if provided
        if weight_config:
            self.multi_platform_scorer.update_weight_config(weight_config)
        
        # Calculate final score
        score_result = self.multi_platform_scorer.calculate_final_score(platform_scores)
        
        # Store result
        task.result = {
            "candidate_id": candidate_id,
            "timestamp": datetime.now().isoformat(),
            "overall_score": score_result.final_score,
            "grade": score_result.grade,
            "platform_scores": platform_scores,
            "recommendations": score_result.recommendations
        }
    
    def _process_batch_prediction_task(self, task: WorkflowTask):
        """Process a batch prediction task"""
        # Extract parameters
        candidate_ids = task.parameters.get("candidate_ids", [])
        job_description = task.parameters.get("job_description")
        
        if not candidate_ids:
            raise ValueError("Missing candidate_ids parameter")
        
        # Initialize results
        results = []
        
        # Process each candidate
        for candidate_id in candidate_ids:
            try:
                # Load candidate data (placeholder)
                # In a real implementation, this would load from a database
                candidate_data = self._load_candidate_data(candidate_id)
                
                # Generate predictions
                prediction_result = self.ai_prediction_engine.predict_candidate_outcomes(
                    platform_scores=candidate_data.get("platform_scores", {}),
                    final_score=candidate_data.get("overall_score", 0),
                    job_description=job_description
                )
                
                # Store result
                results.append({
                    "candidate_id": candidate_id,
                    "predictions": {
                        "technical_round": prediction_result.technical_round_probability,
                        "culture_fit": prediction_result.culture_fit_probability,
                        "learning_adaptability": prediction_result.learning_adaptability_probability
                    },
                    "strengths": prediction_result.top_strengths,
                    "weaknesses": prediction_result.top_weaknesses,
                    "recommendations": prediction_result.actionable_recommendations
                })
                
            except Exception as e:
                logger.error(f"Error processing candidate {candidate_id}: {e}")
                results.append({
                    "candidate_id": candidate_id,
                    "error": str(e)
                })
        
        # Store results
        task.result = {
            "job_description": job_description,
            "timestamp": datetime.now().isoformat(),
            "candidates": results
        }
    
    def _process_model_training_task(self, task: WorkflowTask):
        """Process a model training task"""
        # Extract parameters
        training_data_path = task.parameters.get("training_data_path")
        model_type = task.parameters.get("model_type", "default")
        hyperparameters = task.parameters.get("hyperparameters", {})
        
        if not training_data_path:
            raise ValueError("Missing training_data_path parameter")
        
        # Train model
        training_result = self.ai_prediction_engine.train_model(
            training_data_path=training_data_path,
            model_type=model_type,
            hyperparameters=hyperparameters
        )
        
        # Store result
        task.result = {
            "model_type": model_type,
            "timestamp": datetime.now().isoformat(),
            "metrics": training_result.metrics,
            "model_path": training_result.model_path
        }
    
    def _load_candidate_data(self, candidate_id: str) -> Dict[str, Any]:
        """Load candidate data (placeholder)"""
        # In a real implementation, this would load from a database
        # For now, return dummy data
        return {
            "candidate_id": candidate_id,
            "overall_score": 75.0,
            "platform_scores": {
                "github": {"normalized_score": 80.0},
                "leetcode": {"normalized_score": 70.0},
                "resume": {"normalized_score": 65.0}
            }
        }
    
    def _save_task_result(self, task: WorkflowTask):
        """Save task result to file"""
        try:
            # Create result directory
            result_dir = self.output_dir / task.task_type
            result_dir.mkdir(exist_ok=True)
            
            # Save to file
            result_path = result_dir / f"{task.task_id}.json"
            with open(result_path, 'w') as f:
                json.dump({
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "parameters": task.parameters,
                    "status": task.status,
                    "created_at": task.created_at.isoformat(),
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                    "result": task.result,
                    "error": task.error
                }, f, indent=2, default=str)
            
            logger.info(f"Saved task result to {result_path}")
            
        except Exception as e:
            logger.error(f"Error saving task result: {e}")
    
    def submit_task(self, task_type: str, parameters: Dict[str, Any]) -> str:
        """Submit a task to the workflow system"""
        # Create task
        task_id = str(uuid.uuid4())
        task = WorkflowTask(
            task_id=task_id,
            task_type=task_type,
            parameters=parameters
        )
        
        # Store task
        self.tasks[task_id] = task
        
        # Add to queue
        self.task_queue.put(task_id)
        
        logger.info(f"Submitted task {task_id} ({task_type})")
        
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task information"""
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "status": task.status,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "result": task.result,
            "error": task.error
        }
    
    def schedule_task(self, task_type: str, parameters: Dict[str, Any], 
                    schedule_type: str, schedule_params: Dict[str, Any]) -> str:
        """Schedule a recurring task"""
        schedule_id = str(uuid.uuid4())
        
        # Create schedule
        schedule_info = {
            "schedule_id": schedule_id,
            "task_type": task_type,
            "parameters": parameters,
            "schedule_type": schedule_type,
            "schedule_params": schedule_params,
            "created_at": datetime.now(),
            "next_run": self._calculate_next_run(schedule_type, schedule_params),
            "last_run": None,
            "last_task_id": None,
            "active": True
        }
        
        # Store schedule
        self.schedules[schedule_id] = schedule_info
        
        # Save schedule to file
        self._save_schedules()
        
        logger.info(f"Created schedule {schedule_id} for {task_type} ({schedule_type})")
        
        # Start scheduler thread if not already running
        if not self.schedule_thread or not self.schedule_thread.is_alive():
            self._start_scheduler()
        
        return schedule_id
    
    def _calculate_next_run(self, schedule_type: str, schedule_params: Dict[str, Any]) -> datetime:
        """Calculate next run time based on schedule type and parameters"""
        now = datetime.now()
        
        if schedule_type == "interval":
            interval_minutes = schedule_params.get("minutes", 0)
            interval_hours = schedule_params.get("hours", 0)
            interval_days = schedule_params.get("days", 0)
            
            delta = timedelta(
                minutes=interval_minutes,
                hours=interval_hours,
                days=interval_days
            )
            
            return now + delta
            
        elif schedule_type == "daily":
            hour = schedule_params.get("hour", 0)
            minute = schedule_params.get("minute", 0)
            
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            
            return next_run
            
        elif schedule_type == "weekly":
            day_of_week = schedule_params.get("day_of_week", 0)  # 0 = Monday
            hour = schedule_params.get("hour", 0)
            minute = schedule_params.get("minute", 0)
            
            days_ahead = day_of_week - now.weekday()
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7
            
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0) + timedelta(days=days_ahead)
            
            return next_run
            
        elif schedule_type == "monthly":
            day = schedule_params.get("day", 1)
            hour = schedule_params.get("hour", 0)
            minute = schedule_params.get("minute", 0)
            
            next_run = now.replace(day=1, hour=hour, minute=minute, second=0, microsecond=0)
            
            # Move to next month if current month's day has passed
            if now.day > day or (now.day == day and now.hour > hour) or \
               (now.day == day and now.hour == hour and now.minute >= minute):
                next_run = (next_run.replace(day=28) + timedelta(days=4)).replace(day=1)  # Move to next month
            
            # Set the correct day, handling month length
            month_length = (next_run.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            next_run = next_run.replace(day=min(day, month_length.day))
            
            return next_run
        
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    def _start_scheduler(self):
        """Start the scheduler thread"""
        self.schedule_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True
        )
        self.schedule_thread.start()
        
        logger.info("Started scheduler thread")
    
    def _scheduler_loop(self):
        """Scheduler thread loop to check and run scheduled tasks"""
        logger.info("Scheduler started")
        
        while not self.stop_event.is_set():
            try:
                now = datetime.now()
                
                # Check schedules
                for schedule_id, schedule in list(self.schedules.items()):
                    if not schedule["active"]:
                        continue
                    
                    if schedule["next_run"] <= now:
                        # Run task
                        task_id = self.submit_task(
                            task_type=schedule["task_type"],
                            parameters=schedule["parameters"]
                        )
                        
                        # Update schedule
                        schedule["last_run"] = now
                        schedule["last_task_id"] = task_id
                        schedule["next_run"] = self._calculate_next_run(
                            schedule["schedule_type"],
                            schedule["schedule_params"]
                        )
                        
                        logger.info(f"Ran scheduled task {task_id} for schedule {schedule_id}")
                        
                        # Save schedules
                        self._save_schedules()
                
                # Sleep for a bit
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(30)  # Sleep longer on error
        
        logger.info("Scheduler stopped")
    
    def _save_schedules(self):
        """Save schedules to file"""
        try:
            schedules_path = self.output_dir / "schedules.json"
            with open(schedules_path, 'w') as f:
                json.dump({
                    schedule_id: {
                        **schedule,
                        "created_at": schedule["created_at"].isoformat(),
                        "next_run": schedule["next_run"].isoformat(),
                        "last_run": schedule["last_run"].isoformat() if schedule["last_run"] else None
                    }
                    for schedule_id, schedule in self.schedules.items()
                }, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving schedules: {e}")
    
    def _load_schedules(self):
        """Load schedules from file"""
        try:
            schedules_path = self.output_dir / "schedules.json"
            if not schedules_path.exists():
                return
            
            with open(schedules_path, 'r') as f:
                schedules_data = json.load(f)
            
            for schedule_id, schedule_data in schedules_data.items():
                schedule_data["created_at"] = datetime.fromisoformat(schedule_data["created_at"])
                schedule_data["next_run"] = datetime.fromisoformat(schedule_data["next_run"])
                
                if schedule_data["last_run"]:
                    schedule_data["last_run"] = datetime.fromisoformat(schedule_data["last_run"])
                
                self.schedules[schedule_id] = schedule_data
            
            logger.info(f"Loaded {len(self.schedules)} schedules")
            
        except Exception as e:
            logger.error(f"Error loading schedules: {e}")
    
    def get_schedule(self, schedule_id: str) -> Optional[Dict[str, Any]]:
        """Get schedule information"""
        schedule = self.schedules.get(schedule_id)
        if not schedule:
            return None
        
        return {
            **schedule,
            "created_at": schedule["created_at"].isoformat(),
            "next_run": schedule["next_run"].isoformat(),
            "last_run": schedule["last_run"].isoformat() if schedule["last_run"] else None
        }
    
    def update_schedule(self, schedule_id: str, active: Optional[bool] = None,
                      parameters: Optional[Dict[str, Any]] = None,
                      schedule_params: Optional[Dict[str, Any]] = None) -> bool:
        """Update a schedule"""
        schedule = self.schedules.get(schedule_id)
        if not schedule:
            return False
        
        if active is not None:
            schedule["active"] = active
        
        if parameters is not None:
            schedule["parameters"] = parameters
        
        if schedule_params is not None:
            schedule["schedule_params"] = schedule_params
            schedule["next_run"] = self._calculate_next_run(
                schedule["schedule_type"],
                schedule["schedule_params"]
            )
        
        # Save schedules
        self._save_schedules()
        
        logger.info(f"Updated schedule {schedule_id}")
        
        return True
    
    def delete_schedule(self, schedule_id: str) -> bool:
        """Delete a schedule"""
        if schedule_id not in self.schedules:
            return False
        
        del self.schedules[schedule_id]
        
        # Save schedules
        self._save_schedules()
        
        logger.info(f"Deleted schedule {schedule_id}")
        
        return True
    
    def start(self):
        """Start the workflow orchestrator"""
        # Load schedules
        self._load_schedules()
        
        # Start workers
        self.start_workers()
        
        # Start scheduler
        self._start_scheduler()
        
        logger.info("Workflow orchestrator started")
    
    def stop(self):
        """Stop the workflow orchestrator"""
        # Set stop event
        self.stop_event.set()
        
        # Wait for threads to finish
        for worker in self.worker_threads:
            worker.join(timeout=5.0)
        
        if self.schedule_thread:
            self.schedule_thread.join(timeout=5.0)
        
        logger.info("Workflow orchestrator stopped")


# Example usage
if __name__ == "__main__":
    # Create orchestrator
    orchestrator = WorkflowOrchestrator(output_dir="./workflow_output")
    
    # Start orchestrator
    orchestrator.start()
    
    try:
        # Submit a task
        task_id = orchestrator.submit_task(
            task_type="rescore",
            parameters={
                "candidate_id": "test-candidate",
                "profiles": {
                    "github": {
                        "username": "test-user"
                    }
                }
            }
        )
        
        print(f"Submitted task: {task_id}")
        
        # Schedule a task
        schedule_id = orchestrator.schedule_task(
            task_type="batch_prediction",
            parameters={
                "candidate_ids": ["test-candidate-1", "test-candidate-2"],
                "job_description": "Test job description"
            },
            schedule_type="interval",
            schedule_params={
                "hours": 24
            }
        )
        
        print(f"Created schedule: {schedule_id}")
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Stopping...")
        
    finally:
        # Stop orchestrator
        orchestrator.stop()