from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from transformers import AutoTokenizer, CLIPTextModel


def _cfg_get(cfg, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, key):
        return getattr(cfg, key)
    try:
        return cfg[key]
    except Exception:
        return default


def _shift_azimuth_deg(azimuth: torch.Tensor) -> torch.Tensor:
    return (azimuth + 180.0) % 360.0 - 180.0


@dataclass
class DirectionConfig:
    name: str
    prompt: Callable[[str], str]
    negative_prompt: Callable[[str], str]
    condition: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class PromptProcessorOutput:
    text_embeddings: torch.Tensor
    uncond_text_embeddings: torch.Tensor
    text_embeddings_vd: torch.Tensor
    uncond_text_embeddings_vd: torch.Tensor
    directions: List[DirectionConfig]
    direction2idx: Dict[str, int]
    use_perp_neg: bool
    perp_neg_f_sb: Tuple[float, float, float]
    perp_neg_f_fsb: Tuple[float, float, float]
    perp_neg_f_fs: Tuple[float, float, float]
    perp_neg_f_sf: Tuple[float, float, float]
    prompt: str
    prompts_vd: List[str]

    def get_text_embeddings(
        self,
        elevation: torch.Tensor,
        azimuth: torch.Tensor,
        camera_distances: torch.Tensor,
        view_dependent_prompting: bool = True,
    ) -> torch.Tensor:
        batch_size = int(elevation.shape[0])
        device = elevation.device
        elevation_cpu = elevation.detach().cpu()
        azimuth_cpu = azimuth.detach().cpu()
        camera_distances_cpu = camera_distances.detach().cpu()

        if view_dependent_prompting:
            direction_idx = torch.zeros(batch_size, dtype=torch.long)
            for direction in self.directions:
                direction_idx[direction.condition(elevation_cpu, azimuth_cpu, camera_distances_cpu)] = self.direction2idx[direction.name]

            text_embeddings = self.text_embeddings_vd[direction_idx].to(device)
            uncond_text_embeddings = self.uncond_text_embeddings_vd[direction_idx].to(device)
        else:
            text_embeddings = self.text_embeddings.expand(batch_size, -1, -1).to(device)
            uncond_text_embeddings = self.uncond_text_embeddings.expand(batch_size, -1, -1).to(device)

        return torch.cat([text_embeddings, uncond_text_embeddings], dim=0)


class ModelscopePromptProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_model_path = os.path.join(repo_root, "text-to-video-ms-1.7b")
        pretrained_model_name_or_path = _cfg_get(
            cfg,
            "pretrained_model_name_or_path",
            local_model_path,
        )
        if os.path.isdir(local_model_path):
            pretrained_model_name_or_path = local_model_path

        self.prompt = _cfg_get(cfg, "prompt", "")
        self.negative_prompt = _cfg_get(cfg, "negative_prompt", "")
        self.prompt_front = _cfg_get(cfg, "prompt_front", None)
        self.prompt_side = _cfg_get(cfg, "prompt_side", None)
        self.prompt_back = _cfg_get(cfg, "prompt_back", None)
        self.prompt_overhead = _cfg_get(cfg, "prompt_overhead", None)

        self.overhead_threshold = float(_cfg_get(cfg, "overhead_threshold", 60.0))
        self.front_threshold = float(_cfg_get(cfg, "front_threshold", 45.0))
        self.back_threshold = float(_cfg_get(cfg, "back_threshold", 45.0))
        self.view_dependent_prompt_front = bool(
            _cfg_get(cfg, "view_dependent_prompt_front", False)
        )
        self.use_cache = bool(_cfg_get(cfg, "use_cache", True))
        self.spawn = bool(_cfg_get(cfg, "spawn", False))
        self.use_perp_neg = bool(_cfg_get(cfg, "use_perp_neg", False))
        self.perp_neg_f_sb = tuple(_cfg_get(cfg, "perp_neg_f_sb", (1, 0.5, -0.606)))
        self.perp_neg_f_fsb = tuple(_cfg_get(cfg, "perp_neg_f_fsb", (1, 0.5, 0.967)))
        self.perp_neg_f_fs = tuple(_cfg_get(cfg, "perp_neg_f_fs", (4, 0.5, -2.426)))
        self.perp_neg_f_sf = tuple(_cfg_get(cfg, "perp_neg_f_sf", (4, 0.5, -2.426)))

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
        ).to(self.device)

        for parameter in self.text_encoder.parameters():
            parameter.requires_grad_(False)
        self.text_encoder.eval()

        self.directions = self._build_directions()
        self.direction2idx = {direction.name: index for index, direction in enumerate(self.directions)}
        self.prompts_vd = [direction.prompt(self.prompt) for direction in self.directions]
        self.negative_prompts_vd = [direction.negative_prompt(self.negative_prompt) for direction in self.directions]

        self._output = self._build_output()

        del self.tokenizer
        del self.text_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build_directions(self) -> List[DirectionConfig]:
        def make_prompt(prefix: str, prompt: str) -> str:
            if self.view_dependent_prompt_front:
                return f"{prefix} view of {prompt}"
            return f"{prompt}, {prefix} view"

        return [
            DirectionConfig(
                name="side",
                prompt=lambda prompt: self.prompt_side or make_prompt("side", prompt),
                negative_prompt=lambda prompt: prompt,
                condition=lambda ele, azi, dis: (
                    _shift_azimuth_deg(azi).abs() >= self.front_threshold
                )
                & (
                    _shift_azimuth_deg(azi).abs() < 180 - self.back_threshold
                )
                & (ele <= self.overhead_threshold),
            ),
            DirectionConfig(
                name="front",
                prompt=lambda prompt: self.prompt_front or make_prompt("front", prompt),
                negative_prompt=lambda prompt: prompt,
                condition=lambda ele, azi, dis: (
                    _shift_azimuth_deg(azi) > -self.front_threshold
                )
                & (
                    _shift_azimuth_deg(azi) < self.front_threshold
                )
                & (ele <= self.overhead_threshold),
            ),
            DirectionConfig(
                name="back",
                prompt=lambda prompt: self.prompt_back or make_prompt("back", prompt),
                negative_prompt=lambda prompt: prompt,
                condition=lambda ele, azi, dis: (
                    _shift_azimuth_deg(azi) > 180 - self.back_threshold
                )
                | (
                    _shift_azimuth_deg(azi) < -180 + self.back_threshold
                ),
            ),
            DirectionConfig(
                name="overhead",
                prompt=lambda prompt: self.prompt_overhead or make_prompt("overhead", prompt),
                negative_prompt=lambda prompt: prompt,
                condition=lambda ele, azi, dis: ele > self.overhead_threshold,
            ),
        ]

    def _encode_prompts(
        self,
        prompts: Sequence[str],
        negative_prompts: Sequence[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(
            list(prompts),
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_tokens = self.tokenizer(
            list(negative_prompts),
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(tokens.input_ids.to(self.device))[0].detach().cpu()
            uncond_text_embeddings = self.text_encoder(uncond_tokens.input_ids.to(self.device))[0].detach().cpu()

        return text_embeddings, uncond_text_embeddings

    def _build_output(self) -> PromptProcessorOutput:
        text_embeddings, uncond_text_embeddings = self._encode_prompts(
            [self.prompt],
            [self.negative_prompt],
        )
        text_embeddings_vd, uncond_text_embeddings_vd = self._encode_prompts(
            self.prompts_vd,
            self.negative_prompts_vd,
        )

        return PromptProcessorOutput(
            text_embeddings=text_embeddings,
            uncond_text_embeddings=uncond_text_embeddings,
            text_embeddings_vd=text_embeddings_vd,
            uncond_text_embeddings_vd=uncond_text_embeddings_vd,
            directions=self.directions,
            direction2idx=self.direction2idx,
            use_perp_neg=self.use_perp_neg,
            perp_neg_f_sb=self.perp_neg_f_sb,
            perp_neg_f_fsb=self.perp_neg_f_fsb,
            perp_neg_f_fs=self.perp_neg_f_fs,
            perp_neg_f_sf=self.perp_neg_f_sf,
            prompt=self.prompt,
            prompts_vd=self.prompts_vd,
        )

    def __call__(self) -> PromptProcessorOutput:
        return self._output
