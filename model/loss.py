from typing import Dict

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from model.config import TransfuserConfig
from model.enums import BoundingBox2DIndex


def transfuser_loss(targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: TransfuserConfig):
    trajectory_loss = F.l1_loss(predictions["trajectory"], targets["trajectory"])
    agent_class_loss, agent_box_loss = _agent_loss(targets, predictions, config)

    loss = (
        config.trajectory_weight * trajectory_loss
        + config.agent_class_weight * agent_class_loss
        + config.agent_box_weight * agent_box_loss
    )

    # BEV semantic loss (optional, disabled by default for TruckScenes)
    if config.bev_semantic_weight > 0 and "bev_semantic_map" in targets:
        bev_semantic_loss = F.cross_entropy(predictions["bev_semantic_map"], targets["bev_semantic_map"].long())
        loss += config.bev_semantic_weight * bev_semantic_loss

    return loss


def _agent_loss(targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: TransfuserConfig):
    gt_states, gt_valid = targets["agent_states"], targets["agent_labels"]
    pred_states, pred_logits = predictions["agent_states"], predictions["agent_labels"]

    if config.latent:
        rad_to_ego = torch.arctan2(
            gt_states[..., BoundingBox2DIndex.Y],
            gt_states[..., BoundingBox2DIndex.X],
        )
        in_latent_rad_thresh = torch.logical_and(
            -config.latent_rad_thresh <= rad_to_ego,
            rad_to_ego <= config.latent_rad_thresh,
        )
        gt_valid = torch.logical_and(in_latent_rad_thresh, gt_valid)

    batch_dim, num_instances = pred_states.shape[:2]
    num_gt_instances = gt_valid.sum()
    num_gt_instances = num_gt_instances if num_gt_instances > 0 else num_gt_instances + 1

    ce_cost = _get_ce_cost(gt_valid, pred_logits)
    l1_cost = _get_l1_cost(gt_states, pred_states, gt_valid)

    cost = config.agent_class_weight * ce_cost + config.agent_box_weight * l1_cost
    cost = cost.cpu()

    indices = [linear_sum_assignment(c) for i, c in enumerate(cost)]
    matching = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    idx = _get_src_permutation_idx(matching)

    pred_states_idx = pred_states[idx]
    gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)

    pred_valid_idx = pred_logits[idx]
    gt_valid_idx = torch.cat([t[i] for t, (_, i) in zip(gt_valid, indices)], dim=0).float()

    l1_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")
    l1_loss = l1_loss.sum(-1) * gt_valid_idx
    l1_loss = l1_loss.view(batch_dim, -1).sum() / num_gt_instances

    ce_loss = F.binary_cross_entropy_with_logits(pred_valid_idx, gt_valid_idx, reduction="none")
    ce_loss = ce_loss.view(batch_dim, -1).mean()

    return ce_loss, l1_loss


@torch.no_grad()
def _get_ce_cost(gt_valid: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    gt_valid_expanded = gt_valid[:, :, None].detach().float()
    pred_logits_expanded = pred_logits[:, None, :].detach()

    max_val = torch.relu(-pred_logits_expanded)
    helper_term = max_val + torch.log(torch.exp(-max_val) + torch.exp(-pred_logits_expanded - max_val))
    ce_cost = (1 - gt_valid_expanded) * pred_logits_expanded + helper_term
    ce_cost = ce_cost.permute(0, 2, 1)
    return ce_cost


@torch.no_grad()
def _get_l1_cost(gt_states: torch.Tensor, pred_states: torch.Tensor, gt_valid: torch.Tensor) -> torch.Tensor:
    gt_states_expanded = gt_states[:, :, None, :2].detach()
    pred_states_expanded = pred_states[:, None, :, :2].detach()
    l1_cost = gt_valid[..., None].float() * (gt_states_expanded - pred_states_expanded).abs().sum(dim=-1)
    l1_cost = l1_cost.permute(0, 2, 1)
    return l1_cost


def _get_src_permutation_idx(indices):
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx
