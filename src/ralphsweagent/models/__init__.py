"""Model overrides for ralph-swe-agent custom tool-call behavior."""

from __future__ import annotations

from ralphsweagent._bootstrap import ensure_vendor_minisweagent_on_path

ensure_vendor_minisweagent_on_path()

from minisweagent import models as miniswe_models

_MODEL_OVERRIDES = {
    "litellm": "ralphsweagent.models.litellm_model.LitellmModel",
    "litellm_response": "ralphsweagent.models.litellm_response_model.LitellmResponseModel",
    "openrouter": "ralphsweagent.models.openrouter_model.OpenRouterModel",
    "openrouter_response": "ralphsweagent.models.openrouter_response_model.OpenRouterResponseModel",
    "portkey": "ralphsweagent.models.portkey_model.PortkeyModel",
    "portkey_response": "ralphsweagent.models.portkey_response_model.PortkeyResponseAPIModel",
    "requesty": "ralphsweagent.models.requesty_model.RequestyModel",
}


def register_model_overrides() -> None:
    """Patch mini-swe-agent model-class shortcuts to point at ralph overrides."""
    miniswe_models._MODEL_CLASS_MAPPING.update(_MODEL_OVERRIDES)
    # mini-swe-agent defaults to importing minisweagent.models.litellm_model.LitellmModel
    # when model_class is omitted; patch that symbol too.
    from minisweagent.models import litellm_model as miniswe_litellm_model

    from ralphsweagent.models.litellm_model import LitellmModel

    miniswe_litellm_model.LitellmModel = LitellmModel
