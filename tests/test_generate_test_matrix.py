# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the shared GitHub Actions model-matrix generator."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from tests import model_registry

_GENERATOR_PATH = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "scripts"
    / "generate_test_matrix.py"
)


def _load_generator() -> ModuleType:
    """Load the workflow script as a module without making .github a package."""
    spec = importlib.util.spec_from_file_location(
        "generate_test_matrix", _GENERATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_perf_defaults_to_representative_causal_paths() -> None:
    """An empty models input uses the same causal defaults as PR/push CI."""
    generator = _load_generator()

    matrices: dict[str, list[str]] = generator.generate_matrices()

    assert matrices["perf"] == model_registry.CAUSAL_PATHS


def test_perf_honors_explicit_model_allowlist() -> None:
    """Daily and manual model lists still restrict the perf matrix."""
    model_path = model_registry.ALL_CAUSAL_PATHS[0]
    generator = _load_generator()

    matrices: dict[str, list[str]] = generator.generate_matrices(
        only_models=[model_path]
    )

    assert matrices["perf"] == [model_path]
