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

# TruckScenes 카테고리는 'vehicle.car', 'vehicle.bus.bendy'처럼 계층형 문자열.
# "vehicle.*" 로 시작하면 모두 vehicle로 취급하되, ego 자체(vehicle.ego_trailer)는 제외.
def _is_vehicle_category(name: str) -> bool:
    return name.startswith("vehicle.") and name != "vehicle.ego_trailer"


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
        trailer_trajectory, trailer_mask = self._get_trailer_trajectory_target(sample)

        targets = {
            "trajectory": trajectory,
            "agent_states": agent_states,
            "agent_labels": agent_labels,
            # ego_trailer 미래 trajectory (트랙터 ego frame).
            # trailer 없는 sample은 zeros + mask=0이라 loss에서 자동 제외됨.
            "trailer_trajectory": trailer_trajectory,
            "trailer_mask": trailer_mask,
        }

        return features, targets

    def _get_camera_feature(self, sample: dict) -> torch.Tensor:
        """Load 4 cameras, crop front pair directionally, stitch, resize.

        NavSim transfuser_features.py와 동일하게 ToTensor만 적용 (mean/std normalize X).
        """
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
        """Construct ego status feature: [vx, vy, ax, ay].

        vx, vy: ego_pose translation의 centered difference로 직접 계산.
                (chassis CAN bus의 vx/vy는 ~32% sample에서 0으로 stuck — 데이터 품질 이슈.
                 측정: tools/checks/check_chassis_zero_rate.py.
                 devkit의 box_velocity와 동일 패턴 적용.)
        ax, ay: chassis IMU 그대로 (가속도는 신뢰 가능).
        """
        lidar_channel = _get_reference_channel(sample)
        sd = self._ts.get("sample_data", sample["data"][lidar_channel])

        # ax, ay는 chassis IMU 그대로
        chassis = self._ts.getclosest("ego_motion_chassis", sd["timestamp"])
        ax, ay = float(chassis["ax"]), float(chassis["ay"])

        # vx, vy — ego_pose 미분
        vx, vy = self._estimate_ego_velocity(sample)
        return torch.tensor([vx, vy, ax, ay], dtype=torch.float32)

    def _estimate_ego_velocity(self, sample: dict, max_time_diff: float = 1.5):
        """ego_pose translation의 centered difference로 ego velocity (ego frame).

        devkit truckscenes.box_velocity (truckscenes-devkit/.../truckscenes.py:443)
        와 동일 로직:
          - prev + next 모두 있으면 centered difference (max_time_diff × 2 허용)
          - 한쪽만 있으면 forward/backward
          - time_diff > max_time_diff면 0 fallback (학습 입력에 nan은 곤란)
        """
        lidar_channel = _get_reference_channel(sample)
        sd_cur = self._ts.get("sample_data", sample["data"][lidar_channel])
        cur_ep = self._ts.get("ego_pose", sd_cur["ego_pose_token"])
        cur_yaw = _quaternion_to_yaw(Quaternion(cur_ep["rotation"]))

        has_prev = sample.get("prev", "") != ""
        has_next = sample.get("next", "") != ""
        if not has_prev and not has_next:
            return 0.0, 0.0

        # first(prev or current) translation/timestamp
        if has_prev:
            prev_s = self._ts.get("sample", sample["prev"])
            prev_sd = self._ts.get("sample_data", prev_s["data"][lidar_channel])
            prev_ep = self._ts.get("ego_pose", prev_sd["ego_pose_token"])
            first_pos = np.array(prev_ep["translation"][:2])
            first_t = prev_sd["timestamp"]
        else:
            first_pos = np.array(cur_ep["translation"][:2])
            first_t = sd_cur["timestamp"]

        # last(next or current)
        if has_next:
            next_s = self._ts.get("sample", sample["next"])
            next_sd = self._ts.get("sample_data", next_s["data"][lidar_channel])
            next_ep = self._ts.get("ego_pose", next_sd["ego_pose_token"])
            last_pos = np.array(next_ep["translation"][:2])
            last_t = next_sd["timestamp"]
        else:
            last_pos = np.array(cur_ep["translation"][:2])
            last_t = sd_cur["timestamp"]

        dt = (last_t - first_t) / 1e6  # μs → s
        if has_prev and has_next:
            max_time_diff *= 2
        if dt <= 0 or dt > max_time_diff:
            return 0.0, 0.0

        # global → ego frame (current ego yaw 기준 회전)
        v_global = (last_pos - first_pos) / dt
        cos_y, sin_y = np.cos(-cur_yaw), np.sin(-cur_yaw)
        vx = v_global[0] * cos_y - v_global[1] * sin_y
        vy = v_global[0] * sin_y + v_global[1] * cos_y
        return float(vx), float(vy)

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

        ego_pose = self._ts.get("ego_pose", sd["ego_pose_token"])
        ego_pos = np.array(ego_pose["translation"])
        ego_rot = Quaternion(ego_pose["rotation"])
        ego_yaw = _quaternion_to_yaw(ego_rot)

        agent_states_list = []
        for box in boxes:
            if not _is_vehicle_category(box.name):
                continue

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

    def _get_trailer_trajectory_target(self, sample: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """미래 num_future_samples keyframe의 ego_trailer 박스를 트랙터 ego frame에서 chain.

        반환:
            trajectory: (num_future_samples, 3) — (x, y, heading) per future frame.
            mask: scalar — 현재 sample에 ego_trailer가 있고 supervision 가능하면 1.0, 아니면 0.0.

        - "trailer 없는 scene"이면 즉시 zeros + mask=0 반환.
        - 미래 frame에서 같은 instance의 trailer를 못 찾으면 마지막 유효 pose로 padding (NavSim 패턴 일치).
        - 좌표/heading 모두 *현재* 트랙터 ego frame 기준 (글로벌 → ego 변환).
        """
        lidar_channel = _get_reference_channel(sample)
        sd = self._ts.get("sample_data", sample["data"][lidar_channel])

        # 1) 현재 sample에서 ego_trailer instance_token 찾기
        trailer_instance_token = None
        for box in self._ts.get_boxes(sd["token"]):
            if box.name == "vehicle.ego_trailer":
                ann = self._ts.get("sample_annotation", box.token)
                trailer_instance_token = ann["instance_token"]
                break

        trajectory = np.zeros((self._num_future_samples, 3), dtype=np.float32)
        if trailer_instance_token is None:
            # trailer 없는 scene → mask=0
            return torch.tensor(trajectory), torch.tensor(0.0, dtype=torch.float32)

        # 2) 현재 트랙터 ego_pose
        current_ego = self._ts.get("ego_pose", sd["ego_pose_token"])
        current_pos = np.array(current_ego["translation"][:2])
        current_yaw = _quaternion_to_yaw(Quaternion(current_ego["rotation"]))

        # 3) 미래 frame chain — 같은 instance_token의 trailer box 따라가며 (x, y, heading) 변환
        next_token = sample.get("next", "")
        for i in range(self._num_future_samples):
            if not next_token:
                if i > 0:
                    trajectory[i] = trajectory[i - 1]
                continue

            next_sample = self._ts.get("sample", next_token)
            next_sd = self._ts.get("sample_data", next_sample["data"][lidar_channel])

            trailer_box = None
            for nb in self._ts.get_boxes(next_sd["token"]):
                if nb.name != "vehicle.ego_trailer":
                    continue
                ann = self._ts.get("sample_annotation", nb.token)
                if ann["instance_token"] == trailer_instance_token:
                    trailer_box = nb
                    break

            if trailer_box is None:
                # chain 끊김 — 마지막 유효 pose로 padding
                if i > 0:
                    trajectory[i] = trajectory[i - 1]
                next_token = next_sample.get("next", "")
                continue

            # 글로벌 → 현재 트랙터 ego frame
            delta = trailer_box.center[:2] - current_pos
            cos_y, sin_y = np.cos(-current_yaw), np.sin(-current_yaw)
            local_x = delta[0] * cos_y - delta[1] * sin_y
            local_y = delta[0] * sin_y + delta[1] * cos_y

            trailer_yaw = _quaternion_to_yaw(trailer_box.orientation)
            local_heading = (trailer_yaw - current_yaw + np.pi) % (2 * np.pi) - np.pi

            trajectory[i] = [local_x, local_y, local_heading]
            next_token = next_sample.get("next", "")

        return torch.tensor(trajectory), torch.tensor(1.0, dtype=torch.float32)


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
