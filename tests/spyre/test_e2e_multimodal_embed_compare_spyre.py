"""Generic multimodal embedding accuracy and retrieval through the public ST API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

import hf_adapters.st_backend  # noqa: F401
from hf_adapters.auto_spyre_model import dtype_for_model_path
from tests._vision_helpers import _load_sample_image
from tests.model_registry import MULTIMODAL_EMBED_PATHS, MULTIMODAL_EMBEDDING_MODELS

pytestmark = pytest.mark.model_harness("embedding")

COSINE_THRESHOLD = 0.99
CAPTIONS: tuple[str, ...] = (
    "A photo of a cat.",
    "Two dogs at the beach.",
    "A picture of London at night.",
)
MATCHING_CAPTION_INDEX = 0
_MODEL_INFO_BY_PATH: dict[str, dict[str, Any]] = {
    info["path"]: info for info in MULTIMODAL_EMBEDDING_MODELS.values()
}


@dataclass(frozen=True)
class MultimodalInputs:
    """Inputs and indices for one image-to-caption retrieval case."""

    values: list[Any]
    labels: list[str]
    caption_indices: list[int]
    image_index: int


def _build_inputs(model_path: str) -> MultimodalInputs:
    """Build a semantic retrieval case from registry-declared capabilities."""
    image = _load_sample_image()
    inputs: list[Any] = list(CAPTIONS)
    labels: list[str] = [f"caption: {caption}" for caption in CAPTIONS]
    caption_indices: list[int] = list(range(len(CAPTIONS)))
    image_index = len(inputs)
    inputs.append({"image": image})
    labels.append("image: pipeline-cat-chonk.jpeg")

    model_info: dict[str, Any] = _MODEL_INFO_BY_PATH.get(model_path, {})
    if model_info.get("supports_interleaved_media", False):
        placeholder: Any = model_info.get("interleaved_media_placeholder")
        if not isinstance(placeholder, str) or not placeholder:
            raise ValueError(
                f"{model_path} supports interleaved media but has no placeholder"
            )
        inputs.append({"text": f"A photo of a cat. {placeholder}", "image": image})
        labels.append("interleaved: cat caption + image")

    return MultimodalInputs(
        values=inputs,
        labels=labels,
        caption_indices=caption_indices,
        image_index=image_index,
    )


@pytest.mark.parametrize(
    "model_path", MULTIMODAL_EMBED_PATHS, ids=MULTIMODAL_EMBED_PATHS
)
def test_multimodal_embedding_compare_spyre(model_path: str) -> None:
    """Compare registered multimodal inputs and retrieval results with CPU."""
    if model_path not in _MODEL_INFO_BY_PATH:
        pytest.skip(f"{model_path} does not declare image embedding support")

    inputs: MultimodalInputs = _build_inputs(model_path)
    cpu_dtype: torch.dtype = dtype_for_model_path(model_path, target_device="cpu")
    cpu_model: SentenceTransformer = SentenceTransformer(
        model_path, device="cpu", model_kwargs={"dtype": cpu_dtype}
    )
    expected: torch.Tensor = cpu_model.encode(
        inputs.values, batch_size=1, convert_to_tensor=True
    ).cpu()
    del cpu_model

    spyre_dtype: torch.dtype = dtype_for_model_path(model_path, target_device="spyre")
    spyre_model: SentenceTransformer = SentenceTransformer(
        model_path, backend="spyre", model_kwargs={"dtype": spyre_dtype}
    )
    actual: torch.Tensor = spyre_model.encode(
        inputs.values, batch_size=1, convert_to_tensor=True
    ).cpu()

    assert actual.shape == expected.shape
    assert actual.ndim == 2
    assert actual.shape[0] == len(inputs.values)
    assert torch.isfinite(actual).all()
    cosine: torch.Tensor = F.cosine_similarity(expected.float(), actual.float(), dim=-1)
    print("\n## Multimodal Embeddings: CPU vs Spyre\n")
    print("| Input | Cosine |")
    print("|-------|--------|")
    for label, similarity in zip(inputs.labels, cosine.tolist(), strict=True):
        print(f"| {label!r} | {similarity:.6f} |")
    assert cosine.min().item() >= COSINE_THRESHOLD

    caption_indices: torch.Tensor = torch.tensor(inputs.caption_indices)
    cpu_scores: torch.Tensor = F.cosine_similarity(
        expected[inputs.image_index].float().unsqueeze(0),
        expected[caption_indices].float(),
        dim=-1,
    )
    spyre_scores: torch.Tensor = F.cosine_similarity(
        actual[inputs.image_index].float().unsqueeze(0),
        actual[caption_indices].float(),
        dim=-1,
    )

    cpu_ranking: list[int] = cpu_scores.argsort(descending=True).tolist()
    spyre_ranking: list[int] = spyre_scores.argsort(descending=True).tolist()

    print("\n## Image-to-caption Retrieval\n")
    print("| Caption | CPU score | Spyre score |")
    print("|---------|-----------|-------------|")
    for index, caption in enumerate(CAPTIONS):
        print(
            f"| {caption!r} | {cpu_scores[index].item():.4f} "
            f"| {spyre_scores[index].item():.4f} |"
        )
    print(f"\nCPU ranking:   {[CAPTIONS[index] for index in cpu_ranking]}")
    print(f"Spyre ranking: {[CAPTIONS[index] for index in spyre_ranking]}")

    assert cpu_ranking[0] == MATCHING_CAPTION_INDEX, (
        f"Expected {CAPTIONS[MATCHING_CAPTION_INDEX]!r} to rank first for the cat "
        f"image on CPU, got {CAPTIONS[cpu_ranking[0]]!r}"
    )
    assert spyre_ranking[0] == MATCHING_CAPTION_INDEX, (
        f"Expected {CAPTIONS[MATCHING_CAPTION_INDEX]!r} to rank first for the cat "
        f"image on Spyre, got {CAPTIONS[spyre_ranking[0]]!r}"
    )
    assert cpu_ranking == spyre_ranking, (
        "Image-to-caption ranking differs between CPU and Spyre.\n"
        f"  CPU:   {cpu_ranking}\n"
        f"  Spyre: {spyre_ranking}"
    )
