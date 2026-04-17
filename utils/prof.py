"""Profiling utilities for npedit."""

import os

import torch


class Profiler:
    """A wrapper class for PyTorch profiler to simplify profiling operations."""

    def __init__(self) -> None:
        """Initialize the profiler."""
        self._profiler: torch.profiler.profile | None = None
        self._is_active = False

    def start(self) -> None:
        """Start profiling with default configuration."""
        if self._is_active:
            return  # Already active

        torch.cuda.synchronize()
        self._profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        self._profiler.start()
        self._is_active = True

    def stop(self, trace_path: str | None = None) -> None:
        """
        Stop profiling and export the trace.

        Args:
            trace_path: Full path to save the trace file (if None, only stops without exporting)
        """
        if not self._is_active or self._profiler is None:
            return  # Not active or already stopped

        torch.cuda.synchronize()
        self._profiler.stop()

        # Only export if trace_path is provided
        if trace_path:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)

            # Export the trace
            self._profiler.export_chrome_trace(trace_path)

        self._is_active = False
        self._profiler = None

    @property
    def is_active(self) -> bool:
        """Check if profiler is currently active."""
        return self._is_active
