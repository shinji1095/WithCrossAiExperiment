import torch
from pathlib import Path

def load_partial_state_dict(
    model: torch.nn.Module,
    ckpt_path: str | Path,
    device: str | torch.device = "cpu",
    key_candidates = ("model", "state_dict", "state_dict_ema", "model_ema"),
    strip_prefixes = ("module.", "backbone.", "model.", "network.")
):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    # state dict を推測
    if isinstance(ckpt, dict):
        for k in key_candidates:
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break

    def strip(k: str):
        for p in strip_prefixes:
            if k.startswith(p):
                return k[len(p):]
        return k

    model_sd = model.state_dict()
    filtered = {}
    mismatched, unexpected = [], []
    for k, v in ckpt.items():
        k2 = strip(k)
        if k2 in model_sd:
            if v.shape == model_sd[k2].shape:
                # dtype を合わせてからコピー
                filtered[k2] = v.to(dtype=model_sd[k2].dtype)
            else:
                mismatched.append((k, tuple(v.shape), tuple(model_sd[k2].shape)))
        else:
            unexpected.append(k)

    missing = sorted(set(model_sd.keys()) - set(filtered.keys()))
    msg = (
        f"[partial load] loaded={len(filtered)} "
        f"missing={len(missing)} mismatched={len(mismatched)} unexpected={len(unexpected)}"
    )
    print(msg)
    if mismatched:
        print("  mismatched examples:", mismatched[:5])
    if unexpected:
        print("  unexpected examples:", unexpected[:5])

    model.load_state_dict(filtered, strict=False)
    return {"loaded": list(filtered.keys()),
            "missing": missing,
            "mismatched": mismatched,
            "unexpected": unexpected}
