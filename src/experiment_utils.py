"""Experiment configuration and utilities."""

from dataclasses import dataclass
from typing import Dict, Optional
import yaml
from pathlib import Path

@dataclass
class UDEConfig:
    """UDE pipeline hyperparameters."""
    n_epochs: int = 500
    lr: float = 1e-3
    hidden_dim: int = 32
    batch_size: int = 16
    max_grad_norm: float = 1.0
    random_seed: int = 42

@dataclass
class SINDyConfig:
    """SINDy symbolic regression hyperparameters."""
    degree: int = 2
    threshold: float = 0.05

@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    experiment_name: str
    n_patients: int = 500
    noise_sigma: float = 0.10
    n_timepoints: int = 6
    ode_n_vars: int = 6
    n_replicates: int = 5
    random_seed: int = 42
    ude: UDEConfig = None
    sindy: SINDyConfig = None
    output_dir: str = 'results'

    def __post_init__(self):
        if self.ude is None:
            self.ude = UDEConfig()
        if self.sindy is None:
            self.sindy = SINDyConfig()

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        ude_data = data.pop('ude', {})
        sindy_data = data.pop('sindy', {})

        config = cls(**data)
        config.ude = UDEConfig(**ude_data)
        config.sindy = SINDyConfig(**sindy_data)
        return config

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'experiment_name': self.experiment_name,
            'n_patients': self.n_patients,
            'noise_sigma': self.noise_sigma,
            'n_timepoints': self.n_timepoints,
            'ode_n_vars': self.ode_n_vars,
            'n_replicates': self.n_replicates,
            'random_seed': self.random_seed,
            'ude': {
                'n_epochs': self.ude.n_epochs,
                'lr': self.ude.lr,
                'hidden_dim': self.ude.hidden_dim,
                'batch_size': self.ude.batch_size,
            },
            'sindy': {
                'degree': self.sindy.degree,
                'threshold': self.sindy.threshold,
            },
            'output_dir': self.output_dir,
        }
