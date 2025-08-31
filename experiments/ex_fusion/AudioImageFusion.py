import torch
import torch.nn as nn

class AudioImageFusion(nn.Module):
    """
    Late fusion for audio and image model outputs (average mode only).

    Args:
        num_classes (int): Number of emotion classes.
        alpha (float): Weight for audio probabilities. Image gets (1 - alpha).
    """
    def __init__(self, num_classes=6, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes

    @staticmethod
    def _ensure_2d(t):
        # Adds batch dimension if needed.
        return t.unsqueeze(0) if t is not None and t.dim() == 1 else t

    def fuse_probs(self, probs_audio, probs_image, **kwargs):
        """
        Fuse class probabilities using weighted average.
        Returns fused probabilities.
        """
        probs_audio = self._ensure_2d(probs_audio)
        probs_image = self._ensure_2d(probs_image)
        return self.alpha * probs_audio + (1 - self.alpha) * probs_image
