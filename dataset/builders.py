"""Builder-level cache abstraction (NavSim-style).

Each builder owns a slice of the (features, targets) dict and a
``get_unique_name()`` that encodes the config fields its output shape/value
depends on. Caches are stored per-builder under
``<cache_dir>/<sample_token>/<builder_name>.pkl.gz`` so that:

* Configs with different LiDAR range / channel / camera size produce
  different filenames — silent shape mismatches at training time become
  missing-file cache misses instead.
* Multiple builders share the same sample directory — adding a new
  feature/target doesn't invalidate existing cache files.
* Resume is per-builder: only missing files are recomputed.
* Different models (or ablations) can share builders that have identical
  unique names while keeping their model-specific builders separate.

Mirrors the pattern in
``navsim/planning/training/dataset.py::CacheOnlyDataset`` where each
builder's ``get_unique_name()`` controls its ``.gz`` filename inside a
per-token directory.

``driving_command`` is intentionally **not** a builder: it is derived
from the cached trajectory at ``__getitem__`` time, keeping the cache
mode-agnostic between the heading-based and lateral-based ablations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from configs._base import TransfuserConfig
    from dataset.dataset import TruckScenesDataset


# ---------------------------------------------------------------------------
# Builder abstraction
# ---------------------------------------------------------------------------


class Builder(ABC):
    """One slice of the ``(features, targets)`` output dict, cache-aware.

    Subclasses declare ``BUILDER_KIND`` ("feature" or "target") so the
    dataset can route each builder's output to the correct dict.
    """

    BUILDER_KIND: str = ""

    @abstractmethod
    def get_unique_name(self) -> str:
        """Stable, filesystem-safe identifier. Used as the cache filename.

        Must include every config field that affects the output shape or
        value — otherwise a cache built with one config could be loaded
        silently under another.
        """
        ...

    @abstractmethod
    def compute(self, dataset: "TruckScenesDataset", sample: dict) -> Dict[str, torch.Tensor]:
        """Compute this builder's slice of the output dict from raw data."""
        ...


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------


class CameraFeatureBuilder(Builder):
    """4-camera stitched + cropped RGB tensor."""

    BUILDER_KIND = "feature"

    def __init__(self, config: "TransfuserConfig"):
        self._config = config

    def get_unique_name(self) -> str:
        c = self._config
        return f"camera_w{c.camera_width}_h{c.camera_height}"

    def compute(self, dataset, sample) -> Dict[str, torch.Tensor]:
        return {"camera_feature": dataset._get_camera_feature(sample)}


class LidarFeatureBuilder(Builder):
    """LiDAR BEV histogram. Channel count depends on use_ground_plane."""

    BUILDER_KIND = "feature"

    def __init__(self, config: "TransfuserConfig"):
        self._config = config

    def get_unique_name(self) -> str:
        c = self._config
        return (
            f"lidar_x{c.lidar_min_x}_{c.lidar_max_x}"
            f"_y{c.lidar_min_y}_{c.lidar_max_y}"
            f"_res{c.lidar_resolution_width}x{c.lidar_resolution_height}"
            f"_split{c.lidar_split_height}_gp{int(c.use_ground_plane)}"
        )

    def compute(self, dataset, sample) -> Dict[str, torch.Tensor]:
        return {"lidar_feature": dataset._get_lidar_feature(sample)}


class StatusFeatureBuilder(Builder):
    """(vx, vy, ax, ay) ego status. Fixed shape — config-independent."""

    BUILDER_KIND = "feature"

    def get_unique_name(self) -> str:
        return "status_v1"

    def compute(self, dataset, sample) -> Dict[str, torch.Tensor]:
        return {"status_feature": dataset._get_status_feature(sample)}


# ---------------------------------------------------------------------------
# Target builders
# ---------------------------------------------------------------------------


class TrajectoryTargetBuilder(Builder):
    """Future ego trajectory in ego frame. Shape (num_future, 3) = (x, y, heading)."""

    BUILDER_KIND = "target"

    def __init__(self, num_future_samples: int):
        self._num_future_samples = num_future_samples

    def get_unique_name(self) -> str:
        return f"trajectory_n{self._num_future_samples}"

    def compute(self, dataset, sample) -> Dict[str, torch.Tensor]:
        return {"trajectory": dataset._get_trajectory_target(sample)}


class AgentTargetBuilder(Builder):
    """Other vehicles' 2D bounding boxes in ego frame + class labels."""

    BUILDER_KIND = "target"

    def __init__(self, config: "TransfuserConfig"):
        self._config = config

    def get_unique_name(self) -> str:
        return f"agent_n{self._config.num_bounding_boxes}"

    def compute(self, dataset, sample) -> Dict[str, torch.Tensor]:
        states, labels = dataset._get_agent_targets(sample)
        return {"agent_states": states, "agent_labels": labels}


class TrailerTrajectoryTargetBuilder(Builder):
    """ego_trailer future trajectory in tractor ego frame + presence mask.

    Hitch correction (when use_hitch_corrected_trailer=True) re-anchors
    the trailer center at the tractor hitch point — affects the stored
    trajectory values, so the toggle is encoded in the unique name.
    """

    BUILDER_KIND = "target"

    def __init__(self, config: "TransfuserConfig", num_future_samples: int):
        self._config = config
        self._num_future_samples = num_future_samples

    def get_unique_name(self) -> str:
        c = self._config
        return (
            f"trailer_traj_n{self._num_future_samples}"
            f"_hitch{c.trailer_hitch_x}_{c.trailer_hitch_y}"
            f"_corrected{int(c.use_hitch_corrected_trailer)}"
        )

    def compute(self, dataset, sample) -> Dict[str, torch.Tensor]:
        traj, mask = dataset._get_trailer_trajectory_target(sample)
        return {"trailer_trajectory": traj, "trailer_mask": mask}


# ---------------------------------------------------------------------------
# Default builder set
# ---------------------------------------------------------------------------


def make_default_builders(
    config: "TransfuserConfig",
    num_future_samples: int,
) -> List[Builder]:
    """The builder list TruckScenesDataset uses by default.

    Order is for readability only — cache files are addressed by unique
    name, so reordering doesn't invalidate caches.
    """
    return [
        CameraFeatureBuilder(config),
        LidarFeatureBuilder(config),
        StatusFeatureBuilder(),
        TrajectoryTargetBuilder(num_future_samples),
        AgentTargetBuilder(config),
        TrailerTrajectoryTargetBuilder(config, num_future_samples),
    ]
