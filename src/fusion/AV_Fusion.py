import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionAV(nn.Module):
    """
    Late fusion module for audio + image branches.
    Supported fusion modes:
        - "avg"    : weighted average of probabilities.
        - "prod"   : geometric mean (log-based softmax).
        - "gate"   : learned scalar weight per sample.
        - "mlp"    : MLP on concatenated probabilities or logits.
        - "latent": Linear classifier on concatenated latent vectors.
    """

    def __init__(self,
                 num_classes: int = 6,
                 fusion_mode: str = "avg",
                 alpha: float = 0.5,
                 use_pre_softmax: bool = False,
                 mlp_hidden_dim: int = 128,
                 latent_dim_audio: int = 512,
                 latent_dim_image: int = 128):
        super().__init__()

        assert fusion_mode in {"avg", "prod", "gate", "mlp", "latent"}, "Invalid fusion mode."
        self.fusion_mode = fusion_mode
        self.alpha = alpha
        self.use_pre_softmax = use_pre_softmax

        if fusion_mode == "gate":
            in_dim = 2 * num_classes
            self.gate_fc = nn.Linear(in_dim, 1)

        elif fusion_mode == "mlp":
            in_dim = 2 * num_classes
            self.mlp_head = nn.Sequential(
                nn.Linear(in_dim, mlp_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(mlp_hidden_dim, num_classes)
            )

        elif fusion_mode == "latent":
            in_dim = latent_dim_audio + latent_dim_image
            self.latent_fc = nn.Linear(in_dim, num_classes)

    def _ensure_2d(self, tensor):
        """
        Makes sure tensor is at least 2D (batch, features).
        """
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
        return_gate: bool = False
    ) -> torch.Tensor:
        """
        Fuses the inputs into final softmax probabilities over emotion classes.
        All input tensors should be batch-first ([N, ...]) or will be unsqueezed as needed.
        Returns softmax probability over classes.
        Optionally returns gate values in 'gate' mode.
        """
        # Ensure batch dimension for all tensors
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

        # Debug: warn if input probs do not sum to 1.
        for name, probs in [("audio", probs_audio), ("image", probs_image)]:
            if probs is not None and not torch.allclose(probs.sum(dim=1), torch.ones(probs.shape[0]), atol=1e-4):
                print(f"Warning: probs_{name} does not sum to 1 along classes! Got {probs.sum(dim=1)}")

        if self.fusion_mode == "avg":
            out = self.alpha * probs_audio + (1 - self.alpha) * probs_image

        elif self.fusion_mode == "prod":
            # Numerically stable log-space geometric mean.
            log_probs = (probs_audio.log() + probs_image.log()) / 2
            out = torch.softmax(log_probs, dim=1)

        elif self.fusion_mode == "gate":
            x = torch.cat([pre_softmax_audio, pre_softmax_image], dim=1) if self.use_pre_softmax \
                else torch.cat([probs_audio, probs_image], dim=1)
            alpha = torch.sigmoid(self.gate_fc(x))
            out = alpha * probs_audio + (1 - alpha) * probs_image
            if return_gate:
                return out, alpha

        elif self.fusion_mode == "mlp":
            x = torch.cat([pre_softmax_audio, pre_softmax_image], dim=1) if self.use_pre_softmax \
                else torch.cat([probs_audio, probs_image], dim=1)
            out = torch.softmax(self.mlp_head(x), dim=1)

        elif self.fusion_mode == "latent":
            x = torch.cat([latent_audio, latent_image], dim=1)
            out = torch.softmax(self.latent_fc(x), dim=1)

        else:
            raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")

        return out
