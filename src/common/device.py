import torch

def resolve_device() -> torch.device:
    """Always picks an accelerator if available, otherwise defaults to cpu"""

    accelerator = torch.accelerator.current_accelerator(check_available=True)
    if accelerator is not None:
        return accelerator
    return torch.device("cpu")
