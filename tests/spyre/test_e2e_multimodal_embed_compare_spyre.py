"""Generic multimodal embedding accuracy and retrieval through the public ST API."""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from sentence_transformers import SentenceTransformer

import hf_adapters.st_backend  # noqa: F401
from hf_adapters.auto_spyre_model import dtype_for_model_path
from tests.model_registry import EMBEDDING_MODELS, MULTIMODAL_EMBED_PATHS

pytestmark = pytest.mark.model_harness("embedding")

COSINE_THRESHOLD = 0.99
_MODEL_INFO_BY_PATH: dict[str, dict[str, Any]] = {
    info["path"]: info for info in EMBEDDING_MODELS.values()
}


def _build_inputs(model_path: str) -> list[Any]:
    """Build media-dictionary inputs from registry-declared capabilities."""
    image = Image.new("RGB", (224, 224), color=(30, 130, 210))
    inputs: list[Any] = [
        "A blue image",
        "A solid blue picture.",
        "A dog running in the snow.",
        {"image": image},
    ]
    model_info: dict[str, Any] = _MODEL_INFO_BY_PATH.get(model_path, {})
    if model_info.get("supports_interleaved_media", False):
        placeholder: Any = model_info.get("interleaved_media_placeholder")
        if not isinstance(placeholder, str) or not placeholder:
            raise ValueError(
                f"{model_path} supports interleaved media but has no placeholder"
            )
        inputs.append({"text": f"A blue picture. {placeholder}", "image": image})
    return inputs


@pytest.mark.parametrize(
    "model_path", MULTIMODAL_EMBED_PATHS, ids=MULTIMODAL_EMBED_PATHS
)
def test_multimodal_embedding_compare_spyre(model_path: str) -> None:
    """Compare registered multimodal inputs and retrieval results with CPU."""
    if model_path not in MULTIMODAL_EMBED_PATHS:
        pytest.skip(f"{model_path} does not declare image embedding support")

    inputs: list[Any] = _build_inputs(model_path)
    cpu_dtype: torch.dtype = dtype_for_model_path(model_path, target_device="cpu")
    cpu_model: SentenceTransformer = SentenceTransformer(
        model_path, device="cpu", model_kwargs={"dtype": cpu_dtype}
    )
    expected: torch.Tensor = cpu_model.encode(
        inputs, batch_size=1, convert_to_tensor=True
    ).cpu()
    del cpu_model

    spyre_dtype: torch.dtype = dtype_for_model_path(model_path, target_device="spyre")
    spyre_model: SentenceTransformer = SentenceTransformer(
        model_path, backend="spyre", model_kwargs={"dtype": spyre_dtype}
    )
    actual: torch.Tensor = spyre_model.encode(
        inputs, batch_size=1, convert_to_tensor=True
    ).cpu()

    assert actual.shape == expected.shape
    assert actual.ndim == 2
    assert actual.shape[0] == len(inputs)
    assert torch.isfinite(actual).all()
    cosine: torch.Tensor = F.cosine_similarity(expected.float(), actual.float(), dim=-1)
    print(f"Per-input CPU/Spyre cosine: {cosine.tolist()}")
    assert cosine.min().item() >= COSINE_THRESHOLD

    cpu_scores: torch.Tensor = F.cosine_similarity(
        expected[:1].float(), expected[1:4].float(), dim=-1
    )
    spyre_scores: torch.Tensor = F.cosine_similarity(
        actual[:1].float(), actual[1:4].float(), dim=-1
    )
    assert torch.equal(
        cpu_scores.argsort(descending=True), spyre_scores.argsort(descending=True)
    )
