"""Experiment tracking for MACD framework."""

import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime
import numpy as np

from .logger import get_logger


class ExperimentTracker(ABC):
    """Abstract base class for experiment tracking."""
    
    def __init__(self, experiment_name: str, config: Dict[str, Any]):
        self.experiment_name = experiment_name
        self.config = config
        self.start_time = time.time()
        self.logger = get_logger(self.__class__.__name__)
        self._metrics_history: List[Dict[str, Any]] = []
    
    @abstractmethod
    def log_metric(self, name: str, value: float, step: Optional[int] = None, **kwargs):
        """Log a metric value."""
        pass
    
    @abstractmethod
    def log_config(self, config: Dict[str, Any]):
        """Log configuration."""
        pass
    
    @abstractmethod
    def log_artifact(self, artifact_path: str, artifact_type: str = "file"):
        """Log an artifact (file, model, etc.)."""
        pass
    
    @abstractmethod
    def finish(self):
        """Finish the experiment."""
        pass
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def log_cycle_results(self, cycle: int, results: Dict[str, Any]):
        """Log results for a complete cycle."""
        cycle_metrics = {
            f"cycle_{cycle}_{key}": value for key, value in results.items()
            if isinstance(value, (int, float))
        }
        self.log_metrics(cycle_metrics, step=cycle)
        
        # Log non-numeric results as artifacts or config
        for key, value in results.items():
            if not isinstance(value, (int, float)):
                self.log_artifact(
                    json.dumps(value, indent=2),
                    artifact_type=f"cycle_{cycle}_{key}"
                )
    
    def log_evolution_results(self, generation: int, results: Dict[str, Any]):
        """Log evolution results."""
        evolution_metrics = {
            f"evolution_gen_{generation}_{key}": value for key, value in results.items()
            if isinstance(value, (int, float))
        }
        self.log_metrics(evolution_metrics, step=generation)
    
    def log_evaluation_results(self, results: Dict[str, Any], prefix: str = "eval"):
        """Log evaluation results."""
        eval_metrics = {
            f"{prefix}_{key}": value for key, value in results.items()
            if isinstance(value, (int, float))
        }
        self.log_metrics(eval_metrics)
    
    def log_distillation_results(self, results: Dict[str, Any]):
        """Log distillation results."""
        distill_metrics = {
            f"distill_{key}": value for key, value in results.items()
            if isinstance(value, (int, float))
        }
        self.log_metrics(distill_metrics)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all logged metrics."""
        if not self._metrics_history:
            return {}
        
        # Group metrics by name
        metric_groups = {}
        for entry in self._metrics_history:
            name = entry['name']
            if name not in metric_groups:
                metric_groups[name] = []
            metric_groups[name].append(entry['value'])
        
        # Calculate statistics
        summary = {}
        for name, values in metric_groups.items():
            summary[name] = {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'latest': values[-1] if values else None
            }
        
        return summary


class LocalTracker(ExperimentTracker):
    """Local file-based experiment tracker."""
    
    def __init__(self, experiment_name: str, config: Dict[str, Any], 
                 output_dir: str = "experiments"):
        super().__init__(experiment_name, config)
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking files
        self.metrics_file = self.output_dir / "metrics.jsonl"
        self.config_file = self.output_dir / "config.json"
        self.artifacts_dir = self.output_dir / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)
        
        # Log initial config
        self.log_config(config)
        self.logger.info(f"Local tracker initialized: {self.output_dir}")
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None, **kwargs):
        """Log metric to local file."""
        metric_entry = {
            'timestamp': datetime.now().isoformat(),
            'name': name,
            'value': value,
            'step': step,
            'metadata': kwargs
        }
        
        # Append to metrics file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metric_entry) + '\n')
        
        # Store in memory for summary
        self._metrics_history.append(metric_entry)
        
        self.logger.debug(f"Logged metric: {name}={value} (step={step})")
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration to local file."""
        config_entry = {
            'timestamp': datetime.now().isoformat(),
            'experiment_name': self.experiment_name,
            'config': config
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_entry, f, indent=2)
        
        self.logger.info("Configuration logged")
    
    def log_artifact(self, artifact_path: str, artifact_type: str = "file"):
        """Log artifact to local directory."""
        if artifact_type == "file":
            # Copy file to artifacts directory
            source_path = Path(artifact_path)
            if source_path.exists():
                dest_path = self.artifacts_dir / source_path.name
                import shutil
                shutil.copy2(source_path, dest_path)
                self.logger.info(f"Artifact logged: {dest_path}")
            else:
                self.logger.warning(f"Artifact file not found: {artifact_path}")
        else:
            # Save as text file
            artifact_file = self.artifacts_dir / f"{artifact_type}.txt"
            with open(artifact_file, 'w') as f:
                f.write(str(artifact_path))
            self.logger.info(f"Artifact logged: {artifact_file}")
    
    def finish(self):
        """Finish experiment and save summary."""
        duration = time.time() - self.start_time
        
        # Create summary
        summary = {
            'experiment_name': self.experiment_name,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_seconds': duration,
            'metrics_summary': self.get_metrics_summary(),
            'config': self.config
        }
        
        # Save summary
        summary_file = self.output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Experiment finished: {self.experiment_name}")
        self.logger.info(f"Duration: {duration:.2f} seconds")
        self.logger.info(f"Results saved to: {self.output_dir}")


class WandbTracker(ExperimentTracker):
    """Weights & Biases experiment tracker."""
    
    def __init__(self, experiment_name: str, config: Dict[str, Any],
                 project: str, entity: Optional[str] = None):
        super().__init__(experiment_name, config)
        
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            raise ImportError("wandb package is required for WandbTracker")
        
        # Initialize wandb run
        self.run = self.wandb.init(
            project=project,
            entity=entity,
            name=experiment_name,
            config=config,
            reinit=True
        )
        
        self.logger.info(f"Wandb tracker initialized: {project}/{experiment_name}")
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None, **kwargs):
        """Log metric to wandb."""
        log_dict = {name: value}
        if step is not None:
            log_dict['step'] = step
        
        # Add metadata
        for key, val in kwargs.items():
            log_dict[f"{name}_{key}"] = val
        
        self.wandb.log(log_dict, step=step)
        
        # Store in memory for summary
        self._metrics_history.append({
            'name': name,
            'value': value,
            'step': step,
            'metadata': kwargs
        })
        
        self.logger.debug(f"Logged metric to wandb: {name}={value} (step={step})")
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration to wandb."""
        self.wandb.config.update(config)
        self.logger.info("Configuration logged to wandb")
    
    def log_artifact(self, artifact_path: str, artifact_type: str = "file"):
        """Log artifact to wandb."""
        if artifact_type == "file":
            artifact = self.wandb.Artifact(
                name=f"{self.experiment_name}_{artifact_type}",
                type=artifact_type
            )
            artifact.add_file(artifact_path)
            self.wandb.log_artifact(artifact)
        else:
            # Log as table or other wandb artifact type
            self.wandb.log({artifact_type: artifact_path})
        
        self.logger.info(f"Artifact logged to wandb: {artifact_path}")
    
    def log_table(self, table_name: str, data: List[Dict[str, Any]], columns: List[str]):
        """Log data as wandb table."""
        table = self.wandb.Table(columns=columns, data=data)
        self.wandb.log({table_name: table})
        self.logger.info(f"Table logged to wandb: {table_name}")
    
    def log_model(self, model_path: str, model_name: str = "model"):
        """Log model to wandb."""
        artifact = self.wandb.Artifact(
            name=f"{self.experiment_name}_{model_name}",
            type="model"
        )
        artifact.add_dir(model_path)
        self.wandb.log_artifact(artifact)
        self.logger.info(f"Model logged to wandb: {model_path}")
    
    def finish(self):
        """Finish wandb run."""
        duration = time.time() - self.start_time
        
        # Log final metrics
        self.wandb.log({
            'experiment/duration_seconds': duration,
            'experiment/status': 'completed'
        })
        
        # Finish run
        self.wandb.finish()
        
        self.logger.info(f"Wandb experiment finished: {self.experiment_name}")
        self.logger.info(f"Duration: {duration:.2f} seconds")


def create_tracker(experiment_name: str, config: Dict[str, Any], 
                  tracker_type: str = "local", **kwargs) -> ExperimentTracker:
    """
    Create experiment tracker based on type.
    
    Args:
        experiment_name: Name of the experiment
        config: Experiment configuration
        tracker_type: Type of tracker ("local" or "wandb")
        **kwargs: Additional arguments for tracker initialization
        
    Returns:
        Experiment tracker instance
    """
    if tracker_type == "local":
        return LocalTracker(experiment_name, config, **kwargs)
    elif tracker_type == "wandb":
        return WandbTracker(experiment_name, config, **kwargs)
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")
