def __getattr__(name):
    if name == "CharbonnierLoss":
        from .charbonnier import CharbonnierLoss as _CharbonnierLoss

        return _CharbonnierLoss
    if name == "SobelLoss":
        from .gradient import SobelLoss as _SobelLoss

        return _SobelLoss
    if name in {"LossComputer", "to_y"}:
        from .composite import LossComputer, to_y

        if name == "LossComputer":
            return LossComputer
        return to_y
    raise AttributeError(f"module 'src.losses' has no attribute {name}")


__all__ = ["CharbonnierLoss", "SobelLoss", "LossComputer", "to_y"]
