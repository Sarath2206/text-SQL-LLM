"""
Enhanced Logging and Metrics Tracking Module

This module provides detailed logging capabilities beyond SQL-R1's baseline:
- Reward component breakdown logging
- GPU memory usage tracking
- Training throughput metrics
- Metrics export to JSON and CSV
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import json
import time
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class TrainingMetrics:
    """Training metrics for a single step."""
    step: int
    epoch: int
    loss: float
    mean_reward: float
    kl_divergence: float
    learning_rate: float
    gpu_memory_allocated: Optional[float] = None
    gpu_memory_reserved: Optional[float] = None
    samples_per_sec: Optional[float] = None
    tokens_per_sec: Optional[float] = None


@dataclass
class RewardBreakdown:
    """Reward component breakdown for a single step."""
    step: int
    total: float
    baseline_total: float
    baseline_format: float
    baseline_execution: float
    baseline_result: float
    baseline_length: float
    schema: float
    structural: float
    syntax: float


class EnhancedLogger:
    """
    Enhanced logging system for training metrics and rewards.
    
    Provides:
    - Console logging at configurable intervals
    - File logging (JSON and CSV)
    - GPU memory tracking
    - Reward component breakdown
    - Training throughput metrics
    """
    
    def __init__(
        self,
        log_dir: str,
        log_interval: int = 10,
        save_interval: int = 100,
        detailed_rewards: bool = True,
        track_gpu_memory: bool = True
    ):
        """
        Initialize enhanced logger.
        
        Args:
            log_dir: Directory for saving log files
            log_interval: Log to console every N steps
            save_interval: Save metrics to file every N steps
            detailed_rewards: Whether to log reward component breakdowns
            track_gpu_memory: Whether to track GPU memory usage
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.detailed_rewards = detailed_rewards
        self.track_gpu_memory = track_gpu_memory
        
        # Storage for metrics history
        self.metrics_history: List[Dict[str, Any]] = []
        self.reward_history: List[Dict[str, Any]] = []
        
        # Timing for throughput calculation
        self.last_log_time = time.time()
        self.last_log_step = 0
    
    def log_training_step(
        self,
        step: int,
        epoch: int,
        loss: float,
        mean_reward: float,
        kl_divergence: float,
        learning_rate: float,
        reward_breakdown: Optional[Dict[str, float]] = None,
        batch_size: Optional[int] = None,
        num_tokens: Optional[int] = None
    ):
        """
        Log training step with detailed metrics.
        
        Args:
            step: Current training step
            epoch: Current epoch
            loss: Training loss
            mean_reward: Mean reward for the batch
            kl_divergence: KL divergence from reference policy
            learning_rate: Current learning rate
            reward_breakdown: Optional reward component breakdown
            batch_size: Batch size for throughput calculation
            num_tokens: Number of tokens for throughput calculation
        """
        # TODO: Implement in Task 9.1
        # Placeholder implementation
        pass
    
    def _log_console(
        self,
        step: int,
        metrics: Dict[str, Any],
        reward_breakdown: Optional[Dict[str, float]] = None
    ):
        """Log metrics to console."""
        # TODO: Implement console logging
        pass
    
    def _log_file(
        self,
        step: int,
        metrics: Dict[str, Any],
        reward_breakdown: Optional[Dict[str, float]] = None
    ):
        """Log metrics to file."""
        # TODO: Implement file logging
        pass
    
    def _get_gpu_memory(self) -> Dict[str, float]:
        """
        Get current GPU memory usage.
        
        Returns:
            Dictionary with 'allocated' and 'reserved' memory in GB
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {'allocated': 0.0, 'reserved': 0.0}
        
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved() / 1024**3
        
        return {
            'allocated': allocated,
            'reserved': reserved
        }
    
    def _calculate_throughput(
        self,
        current_step: int,
        batch_size: Optional[int],
        num_tokens: Optional[int]
    ) -> Dict[str, float]:
        """
        Calculate training throughput.
        
        Returns:
            Dictionary with 'samples_per_sec' and 'tokens_per_sec'
        """
        current_time = time.time()
        elapsed_time = current_time - self.last_log_time
        steps_elapsed = current_step - self.last_log_step
        
        throughput = {}
        if batch_size and elapsed_time > 0:
            samples_per_sec = (batch_size * steps_elapsed) / elapsed_time
            throughput['samples_per_sec'] = samples_per_sec
        
        if num_tokens and elapsed_time > 0:
            tokens_per_sec = (num_tokens * steps_elapsed) / elapsed_time
            throughput['tokens_per_sec'] = tokens_per_sec
        
        # Update timing
        self.last_log_time = current_time
        self.last_log_step = current_step
        
        return throughput
    
    def save_metrics(self):
        """
        Save metrics history to JSON and CSV files.
        
        Creates:
        - metrics.json: Complete metrics history
        - metrics.csv: Metrics in CSV format
        - rewards.csv: Reward breakdown history
        """
        # TODO: Implement in Task 9.1
        # Placeholder implementation
        pass
    
    def log_validation(
        self,
        step: int,
        val_loss: float,
        val_mean_reward: float,
        val_accuracy: Optional[float] = None
    ):
        """
        Log validation metrics.
        
        Args:
            step: Current training step
            val_loss: Validation loss
            val_mean_reward: Mean validation reward
            val_accuracy: Optional validation accuracy
        """
        # TODO: Implement validation logging
        pass
