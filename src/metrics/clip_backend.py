#!/usr/bin/env python
"""CLIP backend shared by CLIPScore, the style score and FMI.

The paper reports CLIPScore in the range 0.243-0.318 with a stated FLUX.1-dev
ceiling of 0.334. That range is raw cosine similarity, not the common
`2.5 * max(0, cos)` variant -- using the scaled form would inflate every number
by 2.5x and make the comparison to Table 1 meaningless. The paper names no
backbone but says it follows the original CLIPScore paper, which uses ViT-B/32,
so ViT-B/32 leads the report's tables (it also lands closest to the paper's
range). ViT-L/14 is reported too; it is the first backbone score_all.py loads,
so it is the one that drives the style score and FMI.

Text is truncated at CLIP's 77-token limit. Several long prompts exceed it.
That is standard and is almost certainly what the paper did, but it is stated
in the report rather than left implicit.
"""
from __future__ import annotations

import functools
from typing import List, Union

import torch
from PIL import Image

BACKBONES = {
    "L14": "openai/clip-vit-large-patch14",
    "B32": "openai/clip-vit-base-patch32",
}


def _pool(out):
    """Projected joint-space embedding, across transformers versions.

    transformers 4.x returns a Tensor from get_image_features/get_text_features.
    transformers 5.x returns BaseModelOutputWithPooling, where `pooler_output`
    is the projected embedding (its width equals config.projection_dim) and
    `last_hidden_state` is the *unprojected* backbone output. Taking the wrong
    field silently compares a 768-d vision space against a 512-d joint space,
    which either crashes or -- worse -- produces plausible nonsense.
    """
    if isinstance(out, torch.Tensor):
        return out
    pooled = getattr(out, "pooler_output", None)
    if pooled is None:
        raise TypeError(f"unexpected CLIP output {type(out).__name__}")
    return pooled


class CLIPBackend:
    def __init__(self, backbone: str = "L14", device: str = "cuda"):
        from transformers import CLIPModel, CLIPProcessor
        self.model_id = BACKBONES.get(backbone, backbone)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(self.model_id)

    @torch.no_grad()
    def encode_image(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        if isinstance(images, Image.Image):
            images = [images]
        batch = self.proc(images=images, return_tensors="pt").to(self.device)
        f = _pool(self.model.get_image_features(**batch))
        return torch.nn.functional.normalize(f, dim=-1)

    @torch.no_grad()
    @functools.lru_cache(maxsize=4096)
    def _encode_text_cached(self, text: str) -> torch.Tensor:
        batch = self.proc(text=[text], return_tensors="pt", padding=True,
                          truncation=True, max_length=77).to(self.device)
        f = _pool(self.model.get_text_features(**batch))
        return torch.nn.functional.normalize(f, dim=-1)

    def encode_text(self, text: str) -> torch.Tensor:
        # style phrases repeat across thousands of rows; caching them is free
        return self._encode_text_cached(text)

    def clip_similarity(self, image: Image.Image, text: str) -> float:
        return float((self.encode_image(image) @ self.encode_text(text).T).squeeze())

    # CLIPScore, under the paper's convention, is exactly this cosine.
    clipscore = clip_similarity

    def n_text_tokens(self, text: str) -> int:
        return len(self.proc.tokenizer(text)["input_ids"])

    def truncates(self, text: str) -> bool:
        return self.n_text_tokens(text) > 77
