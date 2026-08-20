#!/usr/bin/env python
"""PickScore v1 -- image quality and aesthetics under human preference.

PickScore is a CLIP-H model finetuned on Pick-a-Pic; the raw score is a
logit-scaled cosine and lands around 19-23. The paper reports 0.204 / 0.212 /
0.221, which is that raw range divided by 100. Both forms are recorded here:
`pickscore_raw` is the honest number and `pickscore_paper` is raw/100 for
comparison against Table 1. Verify the convention on the first scored batch --
if raw values do not land near 20, the assumption is wrong and the report says
so rather than quietly rescaling to fit.

Note the model is normally used to *rank* candidates for one prompt, so absolute
values are only meaningful when compared across methods on the same prompt set,
which is exactly how they are used here.
"""
from __future__ import annotations

from typing import List, Union

import torch
from PIL import Image

from src.metrics.clip_backend import _pool

MODEL_ID = "yuvalkirstain/PickScore_v1"
PROCESSOR_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
PAPER_SCALE = 100.0


class PickScorer:
    def __init__(self, device: str = "cuda"):
        from transformers import AutoModel, AutoProcessor
        self.device = device if torch.cuda.is_available() else "cpu"
        self.proc = AutoProcessor.from_pretrained(PROCESSOR_ID)
        self.model = AutoModel.from_pretrained(MODEL_ID).to(self.device).eval()

    @torch.no_grad()
    def score(self, images: Union[Image.Image, List[Image.Image]],
              text: str) -> List[float]:
        if isinstance(images, Image.Image):
            images = [images]
        ii = self.proc(images=images, return_tensors="pt").to(self.device)
        tt = self.proc(text=[text], return_tensors="pt", padding=True,
                       truncation=True, max_length=77).to(self.device)
        ie = torch.nn.functional.normalize(_pool(self.model.get_image_features(**ii)), dim=-1)
        te = torch.nn.functional.normalize(_pool(self.model.get_text_features(**tt)), dim=-1)
        s = self.model.logit_scale.exp() * (te @ ie.T)[0]
        return [float(v) for v in s]

    def score_one(self, image: Image.Image, text: str) -> float:
        return self.score(image, text)[0]
