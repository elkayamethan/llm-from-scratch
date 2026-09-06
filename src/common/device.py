import torch

def resolve_device() -> torch.device:
    """Always picks an accelerator if available, otherwise defaults to cpu"""

    accelerator = torch.accelerator.current_accelerator(check_available=True)
    if accelerator is not None:
        return accelerator
    return torch.device("cpu")


def supports_bf16(device: torch.device) -> bool:
    """Whether bf16 autocast is available on device, and worth enabling"""

    if device.type == "cuda":
        return torch.cuda.is_bf16_supported()

    if device.type == "mps":
        try:
            torch.zeros(1, dtype=torch.bfloat16, device=device)
        except RuntimeError:
            return False
        return True

    return False
