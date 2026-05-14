"""v9_cmd_no_status = v7 base + driving_command input + ego status omitted.

Goal:
  Open-loop-safe baseline for the paper. Two changes from v7_ground_plane:
    (1) feed a driving_command navigation signal so the planner can
        disambiguate left/right/straight intent at intersections,
    (2) drop ego status (vx, vy, ax, ay) from the model input so the
        open-loop L2 / collision metrics aren't inflated by shortcut
        learning (cf. VAD §4.2).

Driving command (input added):
  3-way one-hot [Turn Right, Turn Left, Go Straight] derived from the last
  future ego pose's heading change |Δyaw@4s|, threshold ±15°. Spirit follows
  NavSim/nuPlan's route-derived driving_command (intersection turns only,
  lane changes stay STRAIGHT). TruckScenes has no HD map / route plan, so
  we derive it from the trajectory — but on heading, not lateral, to avoid
  conflating lane drift with real turns. See
  dataset._get_driving_command and tools/checks/check_driving_command_heading_*.py
  for the evidence behind 15°.

  CARLA TransFuser's original `target_point` (RoutePlanner(7.5, 50): the
  next GPS waypoint ≥7.5m away, autopilot.py:147) is expert-route-derived
  and unavailable in TruckScenes; driving_command is the standard
  nuScenes-domain replacement.

Ego status omitted (input removed):
  VAD §4.2:
    "in the main results, VAD omits ego status features to avoid shortcut
     learning in the open-loop planning [50], but the results of VAD using
     ego status features are still preserved in Tab. 1 for reference."
  Their Tab. 1 shows L2 nearly halves (e.g., VAD-Tiny 0.78 → 0.41) when
  ego status is added — most of the gain is shortcut, not perception.
  We follow VAD's main-results protocol: use_ego_status=False, and
  status_dropout_p=0.0 (dropout has no effect when ego_status is unused).

Status encoder input dim:
  4·use_ego_status + 3·use_driving_command = 0 + 3 = 3.

Optimizer (VAD §4.1 verbatim):
  "We use AdamW optimizer and Cosine Annealing scheduler to train VAD with
   weight decay 0.01 and initial learning rate 2 × 10^-4. VAD is trained
   for 60 epochs on 8 NVIDIA GeForce RTX 3090 GPUs with batch size 1 per
   GPU." (effective batch 8)
  TruckScenes train (~16.7k samples) sits between VAD's nuScenes (~28k)
  and the TruckScenes PETR baseline (batch 8, 48 epochs, Appendix A.6).
  We follow VAD's AdamW + weight_decay 0.01 + cosine annealing, with a
  short warmup for stability on B200-scale batches.

Recommended training command (batch 16 on a single B200):
  ./scripts/train.py --config v9_cmd_no_status \\
      --epochs 48 --batch_size 16 --num_workers 16 --lr 2e-4 \\
      --cache_dir data/cache/v1.1-trainval

Diffs vs v7_ground_plane:
  - use_driving_command=True   (was False)
  - use_ego_status=False       (was True)
  - status_dropout_p=0.0       (was 0.5 — no-op now, kept explicit for clarity)
  - optimizer="adamw"          (was "adam")
  - weight_decay=0.01          (was 0.0)
  - lr_warmup_epochs=2         (was 0)
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
    # Navigation input (NavSim/nuPlan-style intent).
    use_driving_command=True,
    # VAD §4.2 protocol — open-loop shortcut prevention.
    use_ego_status=False,
    status_dropout_p=0.0,
    # VAD §4.1 optimizer setup (AdamW + weight_decay 0.01 + cosine + warmup).
    optimizer="adamw",
    weight_decay=0.01,
    lr_warmup_epochs=2,
)
