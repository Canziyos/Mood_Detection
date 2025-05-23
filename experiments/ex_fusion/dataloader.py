import os, random, numpy as np, torch
from torch.utils.data import Dataset


class FlexibleFusionDataset(Dataset):
    """
    Universal loader for audio/image fusion.

    Parameters
    ----------
    logits_audio_dir : str
    logits_image_dir : str
    latent_audio_dir : str | None (set None to disable audio latents).
    latent_image_dir : str | None
    class_names : list[str]
    pair_mode : bool - True -> strict 1-to-1 pairing (min len).
                     - False ->  random pairing every __getitem__
    oversample_audio : bool -> If True and audio is the smaller side,
                                repeats/upsamples audio so class balance ≈ image count.
    """

    def __init__(
        self,
        logits_audio_dir,
        logits_image_dir,
        class_names,
        latent_audio_dir=None,
        latent_image_dir=None,
        pair_mode=True,
        oversample_audio=False,
        seed=42,
        verbose=True,
    ):
        rng = np.random.default_rng(seed)
        self.class_names = class_names
        self.pair_mode = pair_mode

        self.buffers = {c: {} for c in class_names}  # holds arrays per class.
        self.labels   = []  # flat label list (only used in pair_mode).

        # helper to read .npy if path exists.
        def _safe_load(path):
            return np.load(path) if path and os.path.exists(path) else None

        for idx, cname in enumerate(class_names):
            paths = {
                "ga": os.path.join(logits_audio_dir,  f"{cname}.npy"),
                "gi": os.path.join(logits_image_dir,  f"{cname}.npy"),
                "la": os.path.join(latent_audio_dir,  f"{cname}.npy") if latent_audio_dir else None,
                "li": os.path.join(latent_image_dir,  f"{cname}.npy") if latent_image_dir else None,
            }

            arrays = {k: _safe_load(p) for k, p in paths.items()}
            if arrays["ga"] is None or arrays["gi"] is None:
                if verbose:
                    print(f"Missing logits for class '{cname}'. Skipped.")
                continue

            min_len = min(len(arrays["ga"]), len(arrays["gi"]))
            if min_len == 0:
                if verbose:
                    print(f"Empty data for class '{cname}'. Skipped.")
                continue

            # oversampling audio (optional).
            if oversample_audio and len(arrays["ga"]) < len(arrays["gi"]):
                rep = int(np.ceil(len(arrays["gi"]) / len(arrays["ga"])))
                arrays["ga"] = np.tile(arrays["ga"], (rep, 1))[: len(arrays["gi"])]
                if arrays["la"] is not None:
                    arrays["la"] = np.tile(arrays["la"], (rep, 1))[: len(arrays["gi"])]

            self.buffers[cname] = arrays

            if self.pair_mode:
                pair_len = min(len(arrays["ga"]), len(arrays["gi"]))
                self.labels.extend([idx] * pair_len)

        # build flat indices only for pair_mode.
        if self.pair_mode:
            self.flat_indices = []
            for idx, cname in enumerate(class_names):
                n = len(self.buffers[cname]["ga"])
                m = len(self.buffers[cname]["gi"])
                k = min(n, m)
                for i in range(k):
                    self.flat_indices.append((cname, i))

            if verbose:
                print(f"Pair-mode dataset size: {len(self.flat_indices)}")
        else:
            if verbose:
                sizes = {c: len(self.buffers[c]["ga"]) for c in class_names if self.buffers[c]}
                print(f"Multi-sample dataset (random pairing) – class sizes: {sizes}")

        self.rng = rng

    def _fetch(self, cname, a_idx, i_idx):
        """
        Always returns five Tensors: logits_a , logits_i , latent_a , latent_i , label.
        So DataLoader.default_collate never meets a None.
        """
        buf = self.buffers[cname]

        # logits.
        logits_a = torch.as_tensor(buf["ga"][a_idx], dtype=torch.float32)
        logits_i = torch.as_tensor(buf["gi"][i_idx], dtype=torch.float32)

        # latents.
        if buf.get("la") is not None:   # real audio latent exists.
            latent_a = torch.as_tensor(buf["la"][a_idx], dtype=torch.float32)

        else:                                               # logits-only run
            latent_a = torch.zeros(1, dtype=torch.float32)  # 1-D dummy.

        if buf.get("li") is not None:                       # real image latent exists.
            latent_i = torch.as_tensor(buf["li"][i_idx], dtype=torch.float32)
        else:
            latent_i = torch.zeros(1, dtype=torch.float32)

        # label.
        label = torch.tensor(self.class_names.index(cname), dtype=torch.long)

        return logits_a, logits_i, latent_a, latent_i, label
    
    def __len__(self):
        if self.pair_mode:
            return len(self.flat_indices)
        return sum(len(self.buffers[c]["gi"]) # number of images
                for c in self.class_names if self.buffers[c])

    def __getitem__(self, idx):
        """
        Returns: logits_a , logits_i , latent_a , latent_i , label.
        Latent tensors are dummy zeros when latents are disabled or missing.
        """
        if self.pair_mode:                                    # strict 1-to-1 (audio, image sample).
            cname, local_idx = self.flat_indices[idx]
            return self._fetch(cname, local_idx, local_idx)

        # Random-pair branch.
        # pick class proportional to image count.
        image_counts = [len(self.buffers[c]["gi"]) for c in self.class_names if self.buffers[c]]
        cumulative = np.cumsum(image_counts)
        pick = self.rng.integers(cumulative[-1])
        cname_idx = int(np.searchsorted(cumulative, pick))
        cname = [c for c in self.class_names if self.buffers[c]][cname_idx]

        buf = self.buffers[cname]
        i_idx = self.rng.integers(len(buf["gi"]))
        a_idx = self.rng.integers(len(buf["ga"]))

        return self._fetch(cname, a_idx, i_idx)



class ConflictValDataset(FlexibleFusionDataset):
    """
    Validation loader that flips the image to a different class
    for a fraction of samples (frac_conflict) so the gate sees
    both agreeing and disagreeing pairs.
    """
    def __init__(self, *args, frac_conflict=0.5, **kwargs):
        super().__init__(*args, pair_mode=False, **kwargs)   # random pairing.
        self.frac_conflict = frac_conflict
        self.other_idx = {
            c: [o for o in self.class_names if o != c and self.buffers[o]]
            for c in self.class_names
        }

    def __getitem__(self, idx):
        logits_a, logits_i, lat_a, lat_i, y = super().__getitem__(idx)

        # with probability frac_conflict, replace image with a diff-class sample.
        if np.random.rand() < self.frac_conflict:
            true_cls = self.class_names[y.item()]
            diff_cls = np.random.choice(self.other_idx[true_cls])
            buf = self.buffers[diff_cls]
            i_idx = self.rng.integers(len(buf["gi"]))

            logits_i = torch.tensor(buf["gi"][i_idx], dtype=torch.float32)
            if buf.get("li") is not None:
                lat_i = torch.tensor(buf["li"][i_idx], dtype=torch.float32)

        return logits_a, logits_i, lat_a, lat_i, y

