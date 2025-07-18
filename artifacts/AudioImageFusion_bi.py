import torch
import torch.nn as nn

class AudioImageFusion(nn.Module):
    """
    Late fusion of audio and image model outputs.

    Modes:
      - "avg": Uses fixed alpha to linearly combine probabilities.
      - "gate": Learns per-sample fusion weights with a single linear layer on concatenated logits.

    Args:
        num_classes (int): Number of emotion classes.
        fusion_mode (str): "avg" or "gate".
        alpha (float): Used for "avg" mode (weight for audio).
    """
    def __init__(self, num_classes=6, fusion_mode="avg", alpha=0.5):
        super().__init__()
        assert fusion_mode in {"avg", "gate"}, "fusion_mode must be 'avg' or 'gate'."
        self.fusion_mode = fusion_mode
        self.alpha = alpha

        if fusion_mode == "gate":
            self.gate_fc = nn.Linear(2 * num_classes, 2)

    @staticmethod
    def _ensure_2d(t):
        return t.unsqueeze(0) if t is not None and t.dim() == 1 else t

    def fuse_probs(
        self,
        probs_audio,
        probs_image,
        pre_softmax_audio=None,
        pre_softmax_image=None,
        return_gate=False,
    ):
        # Ensure batch dimension.
        probs_audio = self._ensure_2d(probs_audio)
        probs_image = self._ensure_2d(probs_image)

        if self.fusion_mode == "avg":
            return self.alpha * probs_audio + (1 - self.alpha) * probs_image

        # Gate mode needs logits.
        if pre_softmax_audio is None or pre_softmax_image is None:
            raise ValueError("Logits must be provided for 'gate' fusion mode.")

        pre_softmax_audio = self._ensure_2d(pre_softmax_audio)
        pre_softmax_image = self._ensure_2d(pre_softmax_image)

        gate_in = torch.cat((pre_softmax_audio, pre_softmax_image), dim=1)
        alpha_logits = self.gate_fc(gate_in)
        alpha_weights = torch.softmax(alpha_logits, dim=1)
        fused = (
            alpha_weights[:, 0:1] * probs_audio +
            alpha_weights[:, 1:2] * probs_image
        )
        return (fused, alpha_weights) if return_gate else fused
