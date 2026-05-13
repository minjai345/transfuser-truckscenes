"""v9_cmd_lateral_2m = v9_cmd_no_status with VAD's original lateral threshold.

Ablation pair for v9_cmd_no_status. Same setup (driving_command input,
ego status omitted, no status_dropout), but the driving_command is derived
the way VAD's nuScenes converter does it:
    if ego_fut_trajs[-1][0] >= 2:   command = Turn Right
    elif ego_fut_trajs[-1][0] <= -2: command = Turn Left
    else:                            command = Go Straight
(https://github.com/hustvl/VAD/blob/main/tools/data_converter/vad_nuscenes_converter.py)

Purpose:
  Reviewer-facing comparison. v9_cmd_no_status (heading 15°) is our
  proposed derivation; v9_cmd_lateral_2m reproduces VAD's lateral 2m
  derivation so we can quote the planning metric delta in the paper's
  ablation table and defend the heading-based choice.

  See dataset._get_driving_command for the boundary-case evidence:
  lateral 2m conflates lane drift (lat 4.3m, Δyaw 6°) with real turn
  (lat 2.9m, Δyaw 29°), while heading 15° separates them.

Diffs vs v9_cmd_no_status:
  - driving_command_mode="lateral"  (was "heading")
  - threshold value follows VAD (2.0 m)
"""
from configs._base import TransfuserConfig

config = TransfuserConfig(
    lidar_min_x=-32,
    lidar_max_x=48,
    lidar_resolution_height=320,
    lidar_vert_anchors=10,
    use_trailer_head=False,
    trailer_weight=0.0,
    use_ground_plane=True,
    use_driving_command=True,
    use_ego_status=False,
    status_dropout_p=0.0,
    # VAD-style lateral threshold (ablation against v9_cmd_no_status's heading 15°).
    driving_command_mode="lateral",
    driving_command_threshold_lateral_m=2.0,
)
