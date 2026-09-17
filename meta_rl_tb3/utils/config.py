"""Configuration management utilities.

This module is for loading, saving, and merging
configuration files for experiments.

Config loading was a pain in the last project, 
with incorrect and missing config keys being a frequent issue. 
So I created this to make it easier and less error-prone.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, is_dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import yaml


T = TypeVar('T')


def load_config(
    path: Union[str, Path],
    config_class: Optional[Type[T]] = None,
) -> Union[Dict[str, Any], T]:
    """Load configuration from YAML or JSON file.
    
    Args:
        path: Path to configuration file
        config_class: Optional dataclass to instantiate
        
    Returns:
        Configuration dictionary or dataclass instance
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    # Load based on extension
    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            config_dict = yaml.safe_load(f)
        elif path.suffix == '.json':
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
    
    if config_dict is None:
        config_dict = {}
    
    # Convert to dataclass if specified
    if config_class is not None:
        return dict_to_dataclass(config_dict, config_class)
    
    return config_dict


def save_config(
    config: Union[Dict[str, Any], Any],
    path: Union[str, Path],
    format: str = "yaml",
) -> None:
    """Save configuration to file.
    
    Args:
        config: Configuration dictionary or dataclass
        path: Output path
        format: Output format ('yaml' or 'json')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert dataclass to dict if needed
    if is_dataclass(config) and not isinstance(config, type):
        config_dict = dataclass_to_dict(config)
    elif hasattr(config, '__dict__'):
        config_dict = config.__dict__
    else:
        config_dict = config
    
    with open(path, 'w') as f:
        if format == 'yaml':
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        elif format == 'json':
            json.dump(config_dict, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")


def merge_configs(
    base: Dict[str, Any],
    override: Dict[str, Any],
    deep: bool = True,
) -> Dict[str, Any]:
    """Merge two configuration dictionaries.
    
    Values in override take precedence over base.
    
    Args:
        base: Base configuration
        override: Override configuration
        deep: Whether to merge nested dicts recursively
        
    Returns:
        Merged configuration
    """
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if deep and key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value, deep=True)
        else:
            result[key] = copy.deepcopy(value)
    
    return result


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert a dataclass to a dictionary recursively.
    
    Args:
        obj: Dataclass instance
        
    Returns:
        Dictionary representation
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for field in fields(obj):
            value = getattr(obj, field.name)
            result[field.name] = dataclass_to_dict(value)
        return result
    elif isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: dataclass_to_dict(value) for key, value in obj.items()}
    elif hasattr(obj, 'value'):  # Enum
        return obj.value
    else:
        return obj


def dict_to_dataclass(data: Dict[str, Any], cls: Type[T]) -> T:
    """Convert a dictionary to a dataclass instance.
    
    Args:
        data: Dictionary with configuration values
        cls: Dataclass type to instantiate
        
    Returns:
        Dataclass instance
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    
    # Get field information
    field_types = {f.name: f.type for f in fields(cls)}
    
    kwargs = {}
    for name, value in data.items():
        if name not in field_types:
            continue  # Skip unknown fields
        
        field_type = field_types[name]
        
        # Handle nested dataclasses
        if is_dataclass(field_type) and isinstance(value, dict):
            kwargs[name] = dict_to_dataclass(value, field_type)
        # Handle Optional types
        elif hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
            # Get the non-None type
            args = [a for a in field_type.__args__ if a is not type(None)]
            if args and is_dataclass(args[0]) and isinstance(value, dict):
                kwargs[name] = dict_to_dataclass(value, args[0])
            else:
                kwargs[name] = value
        else:
            kwargs[name] = value
    
    return cls(**kwargs)


def flatten_config(
    config: Dict[str, Any],
    separator: str = ".",
    prefix: str = "",
) -> Dict[str, Any]:
    """Flatten a nested configuration dictionary.
    
    Args:
        config: Nested configuration
        separator: Key separator
        prefix: Key prefix
        
    Returns:
        Flattened dictionary
    """
    result = {}
    
    for key, value in config.items():
        full_key = f"{prefix}{separator}{key}" if prefix else key
        
        if isinstance(value, dict):
            result.update(flatten_config(value, separator, full_key))
        else:
            result[full_key] = value
    
    return result


def unflatten_config(
    config: Dict[str, Any],
    separator: str = ".",
) -> Dict[str, Any]:
    """Unflatten a flattened configuration dictionary.
    
    Args:
        config: Flattened configuration
        separator: Key separator
        
    Returns:
        Nested dictionary
    """
    result = {}
    
    for key, value in config.items():
        parts = key.split(separator)
        current = result
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    return result


class ConfigRegistry:
    """Registry for configuration presets.
    
    Allows registering and retrieving named configurations.
    
    """
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """Initialize registry.
        
        Args:
            config_dir: Optional directory to load configs from
        """
        self._configs: Dict[str, Dict[str, Any]] = {}
        
        if config_dir is not None:
            self.load_from_directory(config_dir)
    
    def register(self, name: str, config: Dict[str, Any]) -> None:
        """Register a configuration.
        
        Args:
            name: Configuration name
            config: Configuration dictionary
        """
        self._configs[name] = copy.deepcopy(config)
    
    def get(
        self,
        name: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get a configuration by name.
        
        Args:
            name: Configuration name
            overrides: Optional overrides to apply
            
        Returns:
            Configuration dictionary
        """
        if name not in self._configs:
            raise KeyError(f"Configuration '{name}' not found")
        
        config = copy.deepcopy(self._configs[name])
        
        if overrides is not None:
            config = merge_configs(config, overrides)
        
        return config
    
    def list(self) -> List[str]:
        """List all registered configurations.
        
        Returns:
            List of configuration names
        """
        return list(self._configs.keys())
    
    def load_from_directory(self, directory: Union[str, Path]) -> None:
        """Load all configurations from a directory.
        
        Args:
            directory: Directory containing config files
        """
        directory = Path(directory)
        
        if not directory.exists():
            return
        
        for path in directory.glob("*.yaml"):
            name = path.stem
            config = load_config(path)
            self.register(name, config)
        
        for path in directory.glob("*.json"):
            name = path.stem
            config = load_config(path)
            self.register(name, config)


def create_experiment_config(
    base_config: Dict[str, Any],
    experiment_name: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Create a complete experiment configuration.
    
    Returns:
        Complete experiment configuration
    """
    config = copy.deepcopy(base_config)
    
    config["experiment"] = {
        "name": experiment_name,
        "seed": seed,
    }
    
    if hyperparameters is not None:
        config = merge_configs(config, hyperparameters)
    
    return config


def validate_config(
    config: Dict[str, Any],
    required_keys: List[str],
    raise_error: bool = True,
) -> bool:
    """Validate that a configuration has required keys.
        
    Returns:
        True if valid, False otherwise
    """
    flat_config = flatten_config(config)
    
    missing = []
    for key in required_keys:
        if key not in flat_config:
            missing.append(key)
    
    if missing:
        if raise_error:
            raise ValueError(f"Missing required configuration keys: {missing}")
        return False
    
    return True
