import torch
import torch.nn as nn

class FusionAV(nn.Module):
    """
    Late fusion module for combining audio and image predictions.
    Supported fusion modes:
        - "avg"    : weighted average of probabilities.
        - "prod"   : geometric mean (log-based softmax).
        - "gate"   : learned scalar weight per sample.
        - "mlp"    : MLP on concatenated probabilities or logits.
        - "latent" : Linear classifier on concatenated latent vectors.
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

        if fusion_mode not in {"avg", "prod", "gate", "mlp", "latent"}:
            raise ValueError(f"Invalid fusion mode: {fusion_mode}")

        self.fusion_mode = fusion_mode
        self.alpha = alpha
        self.use_pre_softmax = use_pre_softmax

        if fusion_mode == "gate":
            self.gate_fc = nn.Linear(2 * num_classes, 1)

        elif fusion_mode == "mlp":
            self.mlp_head = nn.Sequential(
                nn.Linear(2 * num_classes, mlp_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(mlp_hidden_dim, num_classes)
            )

        elif fusion_mode == "latent":
            self.latent_fc = nn.Linear(latent_dim_audio + latent_dim_image, num_classes)

    def fuse_probs(
        self,
        probs_audio: torch.Tensor,
        probs_image: torch.Tensor,
        pre_softmax_audio: torch.Tensor = None,
        pre_softmax_image: torch.Tensor = None,
        latent_audio: torch.Tensor = None,
        latent_image: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Fuses audio and image predictions into final emotion class probabilities.
        """
        if probs_audio.shape != probs_image.shape:
            raise ValueError("Shape mismatch between probs_audio and probs_image.")

        if self.fusion_mode == "avg":
            return self.alpha * probs_audio + (1 - self.alpha) * probs_image

        elif self.fusion_mode == "prod":
            # Prevent numerical instability by using log space
            log_probs = (probs_audio.log() + probs_image.log()) / 2
            return torch.softmax(log_probs, dim=1)

        elif self.fusion_mode == "gate":
            if self.use_pre_softmax:
                if pre_softmax_audio is None or pre_softmax_image is None:
                    raise ValueError("Pre-softmax inputs required for gate fusion.")
                x = torch.cat([pre_softmax_audio, pre_softmax_image], dim=1)
            else:
                x = torch.cat([probs_audio, probs_image], dim=1)
            alpha = torch.sigmoid(self.gate_fc(x))
            return alpha * probs_audio + (1 - alpha) * probs_image

        elif self.fusion_mode == "mlp":
            if self.use_pre_softmax:
                if pre_softmax_audio is None or pre_softmax_image is None:
                    raise ValueError("Pre-softmax inputs required for MLP fusion.")
                x = torch.cat([pre_softmax_audio, pre_softmax_image], dim=1)
            else:
                x = torch.cat([probs_audio, probs_image], dim=1)
            return torch.softmax(self.mlp_head(x), dim=1)

        elif self.fusion_mode == "latent":
            if latent_audio is None or latent_image is None:
                raise ValueError("Latent fusion requires both latent_audio and latent_image.")
            x = torch.cat([latent_audio, latent_image], dim=1)
            return torch.softmax(self.latent_fc(x), dim=1)

        raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")
