"""Experiment configuration and utilities."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Union
import yaml
from pathlib import Path

@dataclass
class UDEConfig:
    """UDE pipeline hyperparameters.

    Attributes:
        n_epochs: Number of training epochs (default: 500)
        lr: Learning rate for Adam optimizer (default: 1e-3)
        hidden_dim: Hidden layer dimension (default: 32)
        batch_size: Batch size for training (default: 16)
        max_grad_norm: Gradient clipping threshold (default: 1.0)
        random_seed: Random seed for reproducibility (default: 42)
    """
    n_epochs: int = 500
    lr: float = 1e-3
    hidden_dim: int = 32
    batch_size: int = 16
    max_grad_norm: float = 1.0
    random_seed: int = 42

@dataclass
class SINDyConfig:
    """SINDy symbolic regression hyperparameters.

    Attributes:
        degree: Maximum polynomial degree (default: 2)
        threshold: Sparsity threshold for coefficient selection (default: 0.05)
    """
    degree: int = 2
    threshold: float = 0.05

@dataclass
class ExperimentConfig:
    """Complete experiment configuration loaded from YAML.

    Attributes:
        experiment_name: Unique identifier for this experiment
        n_patients: Number of synthetic patients to generate (default: 500)
        noise_sigma: Measurement noise level (default: 0.10)
        n_timepoints: Number of observation timepoints (default: 6)
        ode_n_vars: Number of ODE variables (default: 6)
        n_replicates: Number of experiment replicates (default: 5)
        random_seed: Random seed for reproducibility (default: 42)
        ude: UDE pipeline configuration
        sindy: SINDy regression configuration
        output_dir: Directory for experiment results (default: 'results')
    """
    experiment_name: str
    n_patients: int = 500
    noise_sigma: float = 0.10
    n_timepoints: int = 6
    ode_n_vars: int = 6
    n_replicates: int = 5
    random_seed: int = 42
    ude: UDEConfig = field(default_factory=UDEConfig)
    sindy: SINDyConfig = field(default_factory=SINDyConfig)
    output_dir: str = 'results'

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'ExperimentConfig':
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file.

        Returns:
            ExperimentConfig instance.

        Raises:
            FileNotFoundError: If config file does not exist.
            ValueError: If YAML is invalid or missing required fields.
        """
        yaml_path = Path(yaml_path)
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Config file not found: {yaml_path}") from e
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {yaml_path}: {e}") from e

        if not isinstance(data, dict):
            raise ValueError(f"YAML must contain a mapping, got {type(data).__name__}")

        if 'experiment_name' not in data:
            raise ValueError("'experiment_name' is required in config YAML")

        ude_data = data.pop('ude', {})
        sindy_data = data.pop('sindy', {})

        try:
            config = cls(**data)
            config.ude = UDEConfig(**ude_data)
            config.sindy = SINDyConfig(**sindy_data)
        except TypeError as e:
            raise ValueError(f"Invalid configuration parameters: {e}") from e

        return config

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation with all fields automatically serialized.
        """
        return asdict(self)
