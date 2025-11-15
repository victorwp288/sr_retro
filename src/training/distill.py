from contextlib import nullcontext

import torch
import torch.nn.functional as F

from src.training import apply_residual_output


class DistillationHelper:
    def __init__(self, state):
        self.enabled = state.distill_enabled
        self.weight_max = state.distill_weight_max
        self.weight_final = state.distill_weight_final
        self.start_step = state.distill_start_step
        self.final_step = state.distill_final_step
        self.total_steps = state.total_steps
        self.sample_frac = state.distill_sample_frac
        self.teacher = state.teacher_model
        self.teacher_scale = state.teacher_scale
        self.teacher_residual = state.teacher_residual
        self.teacher_twopass = state.teacher_twopass

    def weight_for_step(self, step):
        if not self.enabled or self.weight_max <= 0.0 or step < self.start_step:
            return 0.0
        total_range = max(1, self.total_steps - self.start_step)
        progress = min(max((step - self.start_step) / total_range, 0.0), 1.0)
        weight = self.weight_max * progress
        if self.weight_final != self.weight_max and step >= self.final_step:
            tail_range = max(1, self.total_steps - self.final_step)
            tail = min(max((step - self.final_step) / tail_range, 0.0), 1.0)
            weight = self.weight_max + (self.weight_final - self.weight_max) * tail
        return max(0.0, weight)

    def teacher_forward(self, lr_batch):
        if not self.enabled or self.teacher is None:
            return None
        if self.teacher_twopass:
            mid = self.teacher(lr_batch)
            mid = apply_residual_output(mid, lr_batch, self.teacher_scale, self.teacher_residual)
            mid = mid.clamp(0.0, 1.0)
            sr = self.teacher(mid)
            return apply_residual_output(sr, mid, self.teacher_scale, self.teacher_residual)
        sr = self.teacher(lr_batch)
        return apply_residual_output(sr, lr_batch, self.teacher_scale, self.teacher_residual)

    def maybe_apply(self, step, lr_batch, sr_batch, loss_value, loss_terms, amp_enabled, autocast_context):
        if not self.enabled:
            loss_terms["distill"] = 0.0
            return loss_value, 0.0
        weight = self.weight_for_step(step)
        if weight <= 0.0:
            loss_terms["distill"] = 0.0
            return loss_value, weight
        sample = torch.rand((), device=lr_batch.device).item()
        if sample >= self.sample_frac:
            loss_terms["distill"] = 0.0
            return loss_value, weight
        teacher_out = None
        with torch.no_grad():
            ctx = autocast_context if amp_enabled else nullcontext
            with ctx():
                teacher_out = self.teacher_forward(lr_batch)
        if teacher_out is None:
            loss_terms["distill"] = 0.0
            return loss_value, weight
        teacher_out = teacher_out.clamp(0.0, 1.0)
        distill_loss = F.mse_loss(sr_batch, teacher_out)
        loss_terms["distill"] = float(distill_loss.detach())
        return loss_value + weight * distill_loss, weight
