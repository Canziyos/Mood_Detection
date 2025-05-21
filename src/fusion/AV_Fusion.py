import torch
import torch.nn as nn

class FusionAV(nn.Module):
    """
    Late fusion module for audio + image branches.
    Supported fusion modes:
        - "avg"    : weighted average of probabilities.
        - "gate"   : learned scalar weight per sample (logits only or logits+latents).
        - "latent" : Linear classifier on concatenated latent vectors.
    """

    def __init__(
        self,
        num_classes: int = 6,
        fusion_mode: str = "avg",
        alpha: float = 0.5,
        use_pre_softmax: bool = True,
        latent_dim_audio: int = None,
        latent_dim_image: int = None,
    ):
        super().__init__()

        assert fusion_mode in {"avg", "gate", "hybrid", "latent"}, "Invalid fusion mode."
        self.fusion_mode = fusion_mode
        self.alpha = alpha

        # --- Enforce: for "gate", always use logits (pre-softmax) ---
        if fusion_mode == "gate" and not use_pre_softmax:
            print("[FusionAV] Warning: For 'gate' head, logits (use_pre_softmax=True) are enforced. Overriding user's setting.")
            use_pre_softmax = True
        self.use_pre_softmax = use_pre_softmax

        self.latent_dim_audio = latent_dim_audio
        self.latent_dim_image = latent_dim_image

        if fusion_mode == "gate":
            if latent_dim_audio is not None and latent_dim_image is not None:
                in_dim = 2 * num_classes + latent_dim_audio + latent_dim_image
                self.use_latent_in_gate = True
            else:
                in_dim = 2 * num_classes
                self.use_latent_in_gate = False
            self.gate_fc = nn.Linear(in_dim, 1)
        elif fusion_mode == "latent":
            # Do not create latent_fc yet if dims not known.
            if latent_dim_audio is not None and latent_dim_image is not None:
                in_dim = latent_dim_audio + latent_dim_image
                self.latent_fc = nn.Linear(in_dim, num_classes)
            else:
                self.latent_fc = None  # To be built on first forward pass.

    def _ensure_2d(self, tensor):
        if tensor is not None and tensor.dim() == 1:
            return tensor.unsqueeze(0)
        return tensor

    def fuse_probs(
        self,
        probs_audio: torch.Tensor,
        probs_image: torch.Tensor,
        pre_softmax_audio: torch.Tensor = None,
        pre_softmax_image: torch.Tensor = None,
        latent_audio: torch.Tensor = None,
        latent_image: torch.Tensor = None,
        return_gate: bool = False,
        return_logits: bool = False,
    ) -> torch.Tensor:
        probs_audio = self._ensure_2d(probs_audio)
        probs_image = self._ensure_2d(probs_image)
        if pre_softmax_audio is not None:
            pre_softmax_audio = self._ensure_2d(pre_softmax_audio)
        if pre_softmax_image is not None:
            pre_softmax_image = self._ensure_2d(pre_softmax_image)
        if latent_audio is not None:
            latent_audio = self._ensure_2d(latent_audio)
        if latent_image is not None:
            latent_image = self._ensure_2d(latent_image)

        for name, probs in [("audio", probs_audio), ("image", probs_image)]:
            if probs is not None and not torch.allclose(probs.sum(dim=1), torch.ones(probs.shape[0]), atol=1e-4):
                print(f"Warning: probs_{name} does not sum to 1 along classes! Got {probs.sum(dim=1)}")

        if self.fusion_mode == "avg":
            out = self.alpha * probs_audio + (1 - self.alpha) * probs_image

        elif self.fusion_mode == "gate":
            if pre_softmax_audio is None or pre_softmax_image is None:
                raise ValueError("FusionAV-gate mode, you must provide pre_softmax_audio and pre_softmax_image (logits) as inputs.")
            if getattr(self, 'use_latent_in_gate', False) and latent_audio is not None and latent_image is not None:
                x = torch.cat([pre_softmax_audio, pre_softmax_image, latent_audio, latent_image], dim=1)
            else:
                x = torch.cat([pre_softmax_audio, pre_softmax_image], dim=1)
            alpha = torch.sigmoid(self.gate_fc(x))
            out = alpha * probs_audio + (1 - alpha) * probs_image
            if return_gate:
                return out, alpha

        elif self.fusion_mode == "latent":
            # Dynamically create latent_fc on first use, if needed.
            if self.latent_fc is None:
                in_dim = latent_audio.shape[1] + latent_image.shape[1]
                self.latent_fc = nn.Linear(in_dim, self.fusion_mode == "latent" and self.latent_fc.out_features or 6)  # fallback for num_classes
                print(f"[FusionAV] Dynamically built latent_fc with in_dim={in_dim}.")
                self.latent_fc.to(latent_audio.device)
            x = torch.cat([latent_audio, latent_image], dim=1)
            logits = self.latent_fc(x)
            out = logits if return_logits else torch.softmax(logits, dim=1)

        else:
            raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")

        return out
