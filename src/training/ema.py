import copy

import torch


class EMAHelper:
    def __init__(self, ema_model, decay, update_every):
        self.model = ema_model
        self.decay = float(decay)
        self.update_every = max(1, int(update_every))
        self.updates = 0

    def update(self, source_model):
        self.updates += 1
        if self.updates % self.update_every:
            return
        with torch.no_grad():
            for ema_param, src_param in zip(self.model.parameters(), source_model.parameters()):
                ema_param.mul_(self.decay).add_(src_param.detach(), alpha=1.0 - self.decay)
            for ema_buffer, src_buffer in zip(self.model.buffers(), source_model.buffers()):
                ema_buffer.copy_(src_buffer.detach())

    def state_dict(self):
        return {"model": self.model.state_dict(), "updates": self.updates}

    def load_state_dict(self, state):
        model_state = state.get("model") or state
        self.model.load_state_dict(model_state, strict=True)
        self.updates = int(state.get("updates", 0))


class SWAHelper:
    def __init__(self, template_model):
        self.model = copy.deepcopy(template_model)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.samples = 0

    def update(self, ema_model):
        self.samples += 1
        alpha = 1.0 / self.samples
        with torch.no_grad():
            for swa_param, ema_param in zip(self.model.parameters(), ema_model.parameters()):
                swa_param.mul_(1.0 - alpha).add_(ema_param, alpha=alpha)
            for swa_buffer, ema_buffer in zip(self.model.buffers(), ema_model.buffers()):
                swa_buffer.copy_(ema_buffer)

    def has_samples(self):
        return self.samples > 0

    def state_dict(self):
        return {"model": self.model.state_dict(), "samples": self.samples}

    def load_state_dict(self, state):
        model_state = state.get("model") or state
        self.model.load_state_dict(model_state, strict=True)
        self.samples = int(state.get("samples", 0))
