from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Type


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    module: str
    class_name: str
    impl_status: str
    hyperparams: dict[str, Any]
    checkpoint_subdir: str


def get_model_spec(model_id: str, models: Mapping[str, Any]) -> ModelSpec:
    if model_id not in models:
        raise KeyError(f"unknown model id: {model_id!r}; choose from {sorted(models)}")
    m = models[model_id]
    return ModelSpec(
        model_id=model_id,
        module=str(m["module"]),
        class_name=str(m["class"]),
        impl_status=str(m.get("impl_status", "unknown")),
        hyperparams=dict(m.get("hyperparams") or {}),
        checkpoint_subdir=str(m.get("checkpoint_subdir", model_id)),
    )


def load_model_class(spec: ModelSpec) -> Type[Any]:
    mod = importlib.import_module(spec.module)
    return getattr(mod, spec.class_name)


def instantiate_model(spec: ModelSpec, **kwargs: Any) -> Any:
    cls = load_model_class(spec)
    return cls(spec=spec, **kwargs)
