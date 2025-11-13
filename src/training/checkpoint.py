import inspect

import torch

_TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY = "weights_only" in inspect.signature(torch.load).parameters


def safe_torch_load(path, *, weights_only=False):
    load_kwargs = {"map_location": "cpu"}
    if weights_only and _TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY:
        load_kwargs["weights_only"] = True
    return torch.load(path, **load_kwargs)


def adapt_pretrained_key(key):
    for pref in ("module.", "model.", "_orig_mod."):
        if key.startswith(pref):
            key = key[len(pref):]

    if key.startswith("sub_mean") or key.startswith("add_mean"):
        return None

    if key.startswith("head.0."):
        return "head." + key[len("head.0."):]

    if key.startswith("body."):
        parts = key.split(".")
        if len(parts) >= 4:
            block_idx = parts[1]
            layer = parts[2]
            remainder = ".".join(parts[3:])
            if layer == "0":
                base = f"body.{block_idx}.conv1"
                return f"{base}.{remainder}" if remainder else base
            if layer == "2":
                base = f"body.{block_idx}.conv2"
                return f"{base}.{remainder}" if remainder else base
            return key

    if key.startswith("body_conv"):
        return key

    if key.startswith("tail.0.0."):
        return "tail.0." + key[len("tail.0.0."):]
    if key.startswith("tail.1."):
        return "tail.2." + key[len("tail.1."):]

    if key.startswith("tail.") or key.startswith("head."):
        return key

    return key


def extract_adapted_weights(state_dict, model_state, allow_tail=True):
    adapted = {}
    matched_tail = set()
    for raw_key, tensor in state_dict.items():
        candidate_keys = []
        if raw_key in model_state:
            candidate_keys.append(raw_key)
        new_key = adapt_pretrained_key(raw_key)
        if new_key is not None and new_key not in candidate_keys:
            candidate_keys.append(new_key)
        for key in candidate_keys:
            is_tail = key.startswith("tail.")
            if is_tail and not allow_tail:
                continue
            if key not in model_state:
                continue
            if model_state[key].shape != tensor.shape:
                continue
            adapted[key] = tensor
            if is_tail:
                matched_tail.add(key)
            break
    return adapted, matched_tail
