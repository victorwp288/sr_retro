import torch

from .charbonnier import CharbonnierLoss
from .gradient import SobelLoss
from src.training.utils import resolve_step_value


class LossComputer:
    def __init__(self, config, device, total_steps):
        loss_cfg = config.get("loss", {})
        self.device = device
        self.total_steps = total_steps
        self.schedule = self._prepare_schedule(loss_cfg)
        self.charbonnier = CharbonnierLoss(loss_cfg.get("charbonnier_eps", 1e-3)).to(device)
        self.rgb_l1_enabled = any(entry[1]["rgb_l1"] > 0.0 for entry in self.schedule)
        self.grad_loss = SobelLoss().to(device) if any(entry[1]["grad"] > 0.0 for entry in self.schedule) else None
        self.lpips_model = None
        self.lpips_net = None
        self.y_only = bool(loss_cfg.get("y_only", False))

    def _prepare_schedule(self, loss_cfg):
        raw = loss_cfg.get("schedule") or []
        defaults = {
            "charbonnier": float(loss_cfg.get("charbonnier_weight", 1.0)),
            "rgb_l1": float(loss_cfg.get("rgb_l1_weight", loss_cfg.get("l1_weight", 0.0))),
            "grad": float(loss_cfg.get("grad_weight", 0.0)),
            "lpips": float(loss_cfg.get("lpips_weight", 0.0)),
            "lpips_net": loss_cfg.get("lpips_net", "alex"),
        }
        if not raw:
            raw = [{"step": 0, **defaults}]
        schedule = []
        for item in raw:
            step = resolve_step_value(item.get("step", 0), self.total_steps)
            weights = {
                "charbonnier": float(item.get("charbonnier", defaults["charbonnier"])),
                "rgb_l1": float(item.get("rgb_l1", defaults["rgb_l1"])),
                "grad": float(item.get("grad", defaults["grad"])),
                "lpips": float(item.get("lpips", defaults["lpips"])),
                "lpips_net": item.get("lpips_net", defaults["lpips_net"]),
            }
            schedule.append((step, weights))
        schedule.sort(key=lambda pair: pair[0])
        return schedule

    def _current_weights(self, step):
        idx = 0
        for i, (boundary, _) in enumerate(self.schedule):
            if step >= boundary:
                idx = i
            else:
                break
        return self.schedule[idx][1]

    def _ensure_lpips(self, net):
        if self.lpips_model is not None and self.lpips_net == net:
            return
        import lpips

        self.lpips_model = lpips.LPIPS(net=net).to(self.device)
        self.lpips_model.eval()
        for param in self.lpips_model.parameters():
            param.requires_grad = False
        self.lpips_net = net

    def compute(self, sr, hr, step):
        weights = self._current_weights(step)
        total = torch.zeros(1, device=sr.device, dtype=sr.dtype)
        details = {}
        sr_main = to_y(sr) if self.y_only else sr
        hr_main = to_y(hr) if self.y_only else hr
        if weights["charbonnier"] > 0.0:
            charb = self.charbonnier(sr_main, hr_main)
            total = total + weights["charbonnier"] * charb
            details["charbonnier"] = float(charb.detach())
        if weights["rgb_l1"] > 0.0 and self.rgb_l1_enabled:
            rgb_l1 = torch.abs(sr - hr).mean()
            total = total + weights["rgb_l1"] * rgb_l1
            details["rgb_l1"] = float(rgb_l1.detach())
        if weights["grad"] > 0.0 and self.grad_loss is not None:
            grad = self.grad_loss(sr_main, hr_main)
            total = total + weights["grad"] * grad
            details["grad"] = float(grad.detach())
        if weights["lpips"] > 0.0:
            self._ensure_lpips(weights["lpips_net"])
            sr_lp = sr.float().mul(2.0).sub(1.0)
            hr_lp = hr.float().mul(2.0).sub(1.0)
            lpips_value = self.lpips_model(sr_lp, hr_lp).mean()
            total = total + weights["lpips"] * lpips_value
            details["lpips"] = float(lpips_value.detach())
        return total.squeeze(), details


def to_y(tensor):
    if tensor.size(1) == 1:
        return tensor
    if tensor.size(1) != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {tensor.size(1)}")
    r = tensor[:, 0:1]
    g = tensor[:, 1:2]
    b = tensor[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b
