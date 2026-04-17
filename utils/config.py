"""
Base classes and utilities for configurable model components.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import MISSING, dataclass, fields
import importlib
from typing import Any, Generic, TypeVar

import torch

from utils.fsdp import dist_model_setup
from utils.log import get_logger

logger = get_logger(__name__)

ParamsT = TypeVar("ParamsT", bound="BaseParams")


def create_component(
    module_spec: str, config_params: dict[str, Any], fsdp_spec: dict[str, Any] | None = None, **kwargs: Any
) -> Any:
    """
    Create a component with parameter validation.

    Args:
        module_spec: String like "models.flux_vae.AutoEncoder"
        config_params: Parameters from configuration
        fsdp_spec: Parameters for distributed training

    Returns:
        Instantiated component with validated parameters
    """
    logger.info(f"Creating component {module_spec} with config_params: {config_params} and fsdp_spec: {fsdp_spec}")

    # Resolve class from string
    try:
        module_path, class_name = module_spec.rsplit(".", 1)
        module = importlib.import_module(module_path)
        component_class = getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        raise ImportError(f"Could not import class '{module_spec}': {e}") from e

    # Create validated parameters
    validated_params = component_class.create_params(config_params)

    # Instantiate component, forwarding any additional keyword arguments
    if fsdp_spec is not None:
        fsdp_spec_copy = deepcopy(fsdp_spec)
        if fsdp_spec_copy.get("meta_device_init", False):
            fsdp_spec_copy.pop("meta_device_init")
            with torch.device("meta"):
                component = component_class(validated_params, **kwargs)
        else:
            component = component_class(validated_params, **kwargs)

        component = dist_model_setup(component, **fsdp_spec_copy)
    else:
        component = component_class(validated_params, **kwargs)

    return component


class ConfigurableModule(Generic[ParamsT], ABC):
    """
    Base class for modules with configurable parameters.

    Provides consistent parameter validation and merging across all model components.
    """

    @classmethod
    @abstractmethod
    def get_default_params(cls) -> ParamsT:
        """Return the default parameters dataclass for this module."""
        pass

    @classmethod
    def create_params(cls, config_params: dict[str, Any]) -> ParamsT:
        """
        Create and validate parameters by merging config with defaults.

        Args:
            config_params: Parameters from configuration file

        Returns:
            Validated parameter dataclass instance
        """
        # Get default parameters
        default_params = cls.get_default_params()
        print(f"default_params: {default_params}")

        # Get the dataclass type
        params_class = type(default_params)
        print(f"params_class: {params_class}")

        # Merge config params with defaults
        merged_params: dict[str, Any] = {}
        valid_fields = {f.name for f in fields(params_class)}
        print(f"Valid fields:", valid_fields)

        # Start with defaults
        for field_obj in fields(params_class):
            if field_obj.default is not MISSING:
                merged_params[field_obj.name] = field_obj.default
            elif field_obj.default_factory is not MISSING:
                merged_params[field_obj.name] = field_obj.default_factory()

        # Override with config values
        for key, value in config_params.items():
            if key in valid_fields:
                merged_params[key] = value
            else:
                logger.warning(f"Unknown parameter '{key}' for {cls.__name__}, ignoring")

        # Create and validate the params instance
        try:
            params = params_class(**merged_params)
            logger.debug(f"Created {params_class.__name__} with params: {params}")
            return params
        except Exception as e:
            raise ValueError(f"Invalid parameters for {cls.__name__}: {e}") from e


@dataclass
class BaseParams:
    """Base parameter dataclass with common validation utilities."""

    def __post_init__(self) -> None:
        """Called after initialization for custom validation."""
        self.validate()

    def validate(self) -> None:
        """Override in subclasses for custom parameter validation."""
        pass

    def to_dict(self) -> dict[str, Any]:
        """Convert parameters to dictionary."""
        result = {}
        for field_obj in fields(self):
            value = getattr(self, field_obj.name)
            result[field_obj.name] = value
        return result
