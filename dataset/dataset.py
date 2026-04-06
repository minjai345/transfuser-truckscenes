"""
TruckScenes dataset for TransFuser training.
Uses truckscenes-devkit to load data in nuScenes-like format.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pyquaternion import Quaternion

from truckscenes.utils.data_classes import LidarPointCloud

from model.config import TransfuserConfig
from model.enums import BoundingBox2DIndex


# TruckScenes camera channel names with crop strategy.
# Front cameras: directional crop to keep outer field-of-view.
# Back cameras: no crop (limited overlap between left/right back).
CAMERA_CHANNELS = [
    ("CAMERA_LEFT_FRONT", "left"),      # keep left, crop right
    ("CAMERA_RIGHT_FRONT", "right"),    # keep right, crop left
    ("CAMERA_LEFT_BACK", None),         # no crop
    ("CAMERA_RIGHT_BACK", None),        # no crop
]

# All LiDAR channels to merge
LIDAR_CHANNELS = [
    "LIDAR_TOP_FRONT",
    "LIDAR_TOP_LEFT",
    "LIDAR_TOP_RIGHT",
    "LIDAR_LEFT",
    "LIDAR_RIGHT",
    "LIDAR_REAR",
]

# TruckScenes object categories to treat as "vehicle"
VEHICLE_CATEGORIES = {
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "emergency_vehicle",
    "motorcycle",
    "bicycle",
}


class TruckScenesDataset(Dataset):
    """PyTorch Dataset for TruckScenes with TransFuser features/targets."""

    def __init__(
        self,
        ts,  # TruckScenes instance
        config: TransfuserConfig,
        num_future_samples: int = 8,
        split_tokens: Optional[List[str]] = None,
    ):
        """
        :param ts: initialized TruckScenes devkit instance
        :param config: TransFuser config
        :param num_future_samples: number of future samples for trajectory target
        :param split_tokens: if provided, only use these scene tokens
        """
        self._ts = ts
        self._config = config
        self._num_future_samples = num_future_samples

        # Collect valid sample tokens: samples that have enough future frames
        self._sample_tokens = self._collect_valid_samples(split_tokens)
        print(f"TruckScenesDataset: {len(self._sample_tokens)} valid samples")

    def _collect_valid_samples(self, split_tokens: Optional[List[str]] = None) -> List[str]:
        """Collect sample tokens that have enough future frames for trajectory."""
        valid_tokens = []

        for scene in self._ts.scene:
            if split_tokens is not None and scene["token"] not in split_tokens:
                continue

            # Walk through samples in this scene
            sample_token = scene["first_sample_token"]
            scene_sample_tokens = []
            while sample_token:
                scene_sample_tokens.append(sample_token)
                sample = self._ts.get("sample", sample_token)
                sample_token = sample.get("next", "")
                if not sample_token:
                    break

            # Only keep samples that have enough future frames
            for i in range(len(scene_sample_tokens) - self._num_future_samples):
                valid_tokens.append(scene_sample_tokens[i])

        return valid_tokens

    def __len__(self) -> int:
        return len(self._sample_tokens)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        sample_token = self._sample_tokens[idx]
        sample = self._ts.get("sample", sample_token)

        # === Features ===
        camera_feature = self._get_camera_feature(sample)
        lidar_feature = self._get_lidar_feature(sample)
        status_feature = self._get_status_feature(sample)

        features = {
            "camera_feature": camera_feature,
            "lidar_feature": lidar_feature,
            "status_feature": status_feature,
        }

        # === Targets ===
        trajectory = self._get_trajectory_target(sample)
        agent_states, agent_labels = self._get_agent_targets(sample)

        targets = {
            "trajectory": trajectory,
            "agent_states": agent_states,
            "agent_labels": agent_labels,
        }

        return features, targets

    def _get_camera_feature(self, sample: dict) -> torch.Tensor:
        """Load 4 cameras, crop front pair directionally, stitch, resize."""
        images = []
        for channel, crop_side in CAMERA_CHANNELS:
            cam_token = sample["data"][channel]
            cam_data = self._ts.get("sample_data", cam_token)
            img_path = Path(self._ts.dataroot) / cam_data["filename"]
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if crop_side is not None:
                img = _crop_to_aspect(img, 1.5, crop_side)
            images.append(img)

        stitched = np.concatenate(images, axis=1)
        resized = cv2.resize(stitched, (self._config.camera_width, self._config.camera_height))
        tensor_image = transforms.ToTensor()(resized)
        return tensor_image

    def _get_lidar_feature(self, sample: dict) -> torch.Tensor:
        """Load all LiDAR point clouds, merge in ego frame, compute 2D histogram."""
        all_points = []

        for channel in LIDAR_CHANNELS:
            if channel not in sample["data"]:
                continue
            sd = self._ts.get("sample_data", sample["data"][channel])
            lidar_path = Path(self._ts.dataroot) / sd["filename"]

            # Load PCD using devkit
            pc = LidarPointCloud.from_file(str(lidar_path))  # (4, N): x, y, z, intensity

            # Transform from sensor frame to ego frame
            cs_record = self._ts.get("calibrated_sensor", sd["calibrated_sensor_token"])
            rotation = Quaternion(cs_record["rotation"]).rotation_matrix
            translation = np.array(cs_record["translation"])
            points_xyz = pc.points[:3].T  # (N, 3)
            points_xyz = points_xyz @ rotation.T + translation
            all_points.append(points_xyz)

        if all_points:
            merged_pc = np.concatenate(all_points, axis=0)  # (N_total, 3)
        else:
            merged_pc = np.zeros((0, 3), dtype=np.float32)

        return self._compute_lidar_histogram(merged_pc)

    def _compute_lidar_histogram(self, pc: np.ndarray) -> torch.Tensor:
        """
        Compute LiDAR 2D histogram from point cloud.
        :param pc: (N, 3+) point cloud array, columns [x, y, z, ...]
        """
        config = self._config

        # Filter by height
        pc = pc[pc[:, 2] < config.max_height_lidar]
        above = pc[pc[:, 2] > config.lidar_split_height]

        def splat_points(point_cloud):
            xbins = np.linspace(
                config.lidar_min_x,
                config.lidar_max_x,
                int((config.lidar_max_x - config.lidar_min_x) * config.pixels_per_meter) + 1,
            )
            ybins = np.linspace(
                config.lidar_min_y,
                config.lidar_max_y,
                int((config.lidar_max_y - config.lidar_min_y) * config.pixels_per_meter) + 1,
            )
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
        
            hist[hist > config.hist_max_per_pixel] = config.hist_max_per_pixel
            return hist / config.hist_max_per_pixel

        above_features = splat_points(above)

        if config.use_ground_plane:
            below = pc[pc[:, 2] <= config.lidar_split_height]
            below_features = splat_points(below)
            features = np.stack([below_features, above_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)

        features = np.transpose(features, (2, 0, 1)).astype(np.float32)
        return torch.tensor(features)

    def _get_status_feature(self, sample: dict) -> torch.Tensor:
        """Construct ego status feature: [vx, vy, ax, ay] from CAN data."""
        lidar_channel = _get_reference_channel(sample)
        sd = self._ts.get("sample_data", sample["data"][lidar_channel])

        chassis = self._ts.getclosest("ego_motion_chassis", sd["timestamp"])
        status = [chassis["vx"], chassis["vy"], chassis["ax"], chassis["ay"]]
        return torch.tensor(status, dtype=torch.float32)

    def _get_trajectory_target(self, sample: dict) -> torch.Tensor:
        """Compute future trajectory in ego-centric coordinates."""
        lidar_channel = _get_reference_channel(sample)

        # Get current ego pose
        sd = self._ts.get("sample_data", sample["data"][lidar_channel])
        current_ego = self._ts.get("ego_pose", sd["ego_pose_token"])
        current_pos = np.array(current_ego["translation"][:2])
        current_rot = Quaternion(current_ego["rotation"])
        current_yaw = _quaternion_to_yaw(current_rot)

        # Collect future ego poses
        trajectory = np.zeros((self._num_future_samples, 3), dtype=np.float32)
        next_token = sample.get("next", "")

        for i in range(self._num_future_samples):
            if not next_token:
                # Pad with last known pose
                if i > 0:
                    trajectory[i] = trajectory[i - 1]
                continue

            next_sample = self._ts.get("sample", next_token)
            next_sd = self._ts.get("sample_data", next_sample["data"][lidar_channel])
            next_ego = self._ts.get("ego_pose", next_sd["ego_pose_token"])

            # Transform to ego-centric coordinates
            future_pos = np.array(next_ego["translation"][:2])
            future_rot = Quaternion(next_ego["rotation"])
            future_yaw = _quaternion_to_yaw(future_rot)

            # Relative position
            delta = future_pos - current_pos
            cos_yaw, sin_yaw = np.cos(-current_yaw), np.sin(-current_yaw)
            local_x = delta[0] * cos_yaw - delta[1] * sin_yaw
            local_y = delta[0] * sin_yaw + delta[1] * cos_yaw

            # Relative heading
            local_heading = future_yaw - current_yaw
            # Normalize to [-pi, pi]
            local_heading = (local_heading + np.pi) % (2 * np.pi) - np.pi

            trajectory[i] = [local_x, local_y, local_heading]
            next_token = next_sample.get("next", "")

        return torch.tensor(trajectory, dtype=torch.float32)

    def _get_agent_targets(self, sample: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract 2D vehicle bounding boxes in ego frame."""
        max_agents = self._config.num_bounding_boxes

        # Get boxes in ego frame
        lidar_channel = _get_reference_channel(sample)
        sd = self._ts.get("sample_data", sample["data"][lidar_channel])
        boxes = self._ts.get_boxes(sd["token"])

        agent_states_list = []
        for box in boxes:
            if box.name.split(".")[0] not in VEHICLE_CATEGORIES:
                continue

            # box.center is in global frame, transform to ego
            ego_pose = self._ts.get("ego_pose", sd["ego_pose_token"])
            ego_pos = np.array(ego_pose["translation"])
            ego_rot = Quaternion(ego_pose["rotation"])

            # Transform box center to ego frame
            center = box.center - ego_pos
            center = ego_rot.inverse.rotate(center)
            x, y = center[0], center[1]

            # Check if in LiDAR range
            if not (self._config.lidar_min_x <= x <= self._config.lidar_max_x and
                    self._config.lidar_min_y <= y <= self._config.lidar_max_y):
                continue

            # Box heading in ego frame
            box_yaw = _quaternion_to_yaw(box.orientation)
            ego_yaw = _quaternion_to_yaw(ego_rot)
            heading = box_yaw - ego_yaw
            heading = (heading + np.pi) % (2 * np.pi) - np.pi

            length, width = box.wlh[1], box.wlh[0]  # wlh = [width, length, height]
            agent_states_list.append([x, y, heading, length, width])

        # Sort by distance, keep closest
        agent_states = np.zeros((max_agents, BoundingBox2DIndex.size()), dtype=np.float32)
        agent_labels = np.zeros(max_agents, dtype=np.float32)

        if agent_states_list:
            arr = np.array(agent_states_list, dtype=np.float32)
            distances = np.linalg.norm(arr[:, :2], axis=-1)
            argsort = np.argsort(distances)[:max_agents]
            arr = arr[argsort]
            agent_states[: len(arr)] = arr
            agent_labels[: len(arr)] = 1.0

        return torch.tensor(agent_states), torch.tensor(agent_labels)


def _crop_to_aspect(
    image: npt.NDArray[np.uint8],
    aspect_ratio: float,
    side: str = "center",
) -> npt.NDArray[np.uint8]:
    """Crop image to target aspect ratio.

    Args:
        side: Which side to keep. "left" crops right, "right" crops left,
              "center" crops both sides equally.
    """
    height, width = image.shape[:2]
    current_aspect = width / float(height)

    if current_aspect > aspect_ratio:
        crop_width = int(round(height * aspect_ratio))
        if side == "left":
            return image[:, :crop_width]
        elif side == "right":
            return image[:, width - crop_width:]
        else:
            left = (width - crop_width) // 2
            return image[:, left: left + crop_width]

    crop_height = int(round(width / aspect_ratio))
    top = (height - crop_height) // 2
    return image[top: top + crop_height, :]


def _get_reference_channel(sample: dict) -> str:
    """Get a reference sensor channel for ego pose lookup. Prefer LIDAR_TOP_FRONT."""
    if "LIDAR_TOP_FRONT" in sample["data"]:
        return "LIDAR_TOP_FRONT"
    for key in sample["data"]:
        if "LIDAR" in key.upper():
            return key
    # Fallback to first camera
    return CAMERA_CHANNELS[0][0]


def _quaternion_to_yaw(q: Quaternion) -> float:
    """Extract yaw angle from quaternion."""
    # Rotation matrix approach
    v = q.rotate(np.array([1.0, 0.0, 0.0]))
    return np.arctan2(v[1], v[0])
