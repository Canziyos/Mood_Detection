import torch
import torch.nn as nn

class FusionAV(nn.Module):
    """
    Late-fusion of audio + image.
    Modes
    -----
    • "avg"  -> fixed α
    • "gate" -> learned per-sample α, (optionally with latents).

    Parameters
    ----------
    num_classes : int
    fusion_mode : {"avg", "gate"}
    alpha : float (only for "avg")
    latent_dim_audio : int | None
    latent_dim_image : int | None
    use_latents : bool - if False -> logits-only gate even when latent dims are provided.
    gate_hidden : int  - hidden size of 2-layer gate MLP (ignored in simplified version)
    """
    def __init__(
        self,
        num_classes: int = 6,
        fusion_mode: str = "avg",
        alpha: float = 0.5,
        latent_dim_audio: int | None = None,
        latent_dim_image: int | None = None,
        use_latents: bool = True,
        gate_hidden: int = 64,  # unused, kept for compatibility
    ):
        super().__init__()

        if fusion_mode not in {"avg", "gate"}:
            raise ValueError("fusion_mode must be 'avg' or 'gate'.")

        self.fusion_mode = fusion_mode
        self.alpha = alpha

        # whether we really fuse latents.
        self.use_latents = (
            use_latents
            and latent_dim_audio is not None
            and latent_dim_image is not None
        )
        self.latent_dim_audio = latent_dim_audio
        self.latent_dim_image = latent_dim_image
        self._warned_missing_latent = False
        # ------------------------------------- #

        if fusion_mode == "gate":
            in_dim = 2 * num_classes
            if self.use_latents:
                in_dim += latent_dim_audio + latent_dim_image
            # --- Only a single linear layer (no hidden) ---
            self.gate_fc = nn.Linear(in_dim, 2)

    # helpers.
    @staticmethod
    def _ensure_2d(t):
        return t.unsqueeze(0) if t is not None and t.dim() == 1 else t

    def _maybe_fill(self, lat, dim, ref, name):
        if lat is not None:
            return self._ensure_2d(lat)
        if not self._warned_missing_latent:
            print(f"[FusionAV] Warning: {name} missing -> using zeros.")
            self._warned_missing_latent = True
        return torch.zeros(ref.size(0), dim, device=ref.device, dtype=ref.dtype)

    # forward.
    def fuse_probs(
        self,
        probs_audio,
        probs_image,
        pre_softmax_audio,
        pre_softmax_image,
        latent_audio=None,
        latent_image=None,
        return_gate=False,
    ):
        # 2-D safety.
        probs_audio, probs_image = map(self._ensure_2d, (probs_audio, probs_image))
        pre_softmax_audio = self._ensure_2d(pre_softmax_audio)
        pre_softmax_image = self._ensure_2d(pre_softmax_image)

        if self.fusion_mode == "avg":
            return self.alpha * probs_audio + (1 - self.alpha) * probs_image

        # gate mode.
        if self.use_latents:
            latent_audio = self._maybe_fill(latent_audio, self.latent_dim_audio, pre_softmax_audio, "latent_audio")
            latent_image = self._maybe_fill(latent_image, self.latent_dim_image, pre_softmax_image, "latent_image")
            gate_in = torch.cat(
                (pre_softmax_audio, pre_softmax_image, latent_audio, latent_image),
                dim=1
            )
        else:
            gate_in = torch.cat((pre_softmax_audio, pre_softmax_image), dim=1)

        alpha_logits = self.gate_fc(gate_in)                  # B × 2
        alpha_weights = torch.softmax(alpha_logits, dim=1)    # B × 2

        fused = (
            alpha_weights[:, 0:1] * probs_audio +
            alpha_weights[:, 1:2] * probs_image
        )

        return (fused, alpha_weights) if return_gate else fused
