"""Mine-type dispatch for structurally different expected-state engines.

The configured mine type is resolved once at the application boundary. Consumers
receive a matching physics engine and residual model without mine-type conditionals.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol


class ExpectedStateEngine(Protocol):
    def monitoring_mode(self, *args: Any, **kwargs: Any) -> Any: ...
    def planning_mode(self, *args: Any, **kwargs: Any) -> Any: ...


class ResidualModel(Protocol):
    def predict(self, *args: Any, **kwargs: Any) -> Any: ...


ExpectedStateFactory = Callable[[], ExpectedStateEngine]
ResidualModelFactory = Callable[[], ResidualModel]
EXPECTED_STATE_REGISTRY: dict[str, ExpectedStateFactory] = {}
MODEL_REGISTRY: dict[str, ResidualModelFactory] = {}


def register_expected_state(mine_type: str) -> Callable[[ExpectedStateFactory], ExpectedStateFactory]:
    def decorator(factory: ExpectedStateFactory) -> ExpectedStateFactory:
        if mine_type in EXPECTED_STATE_REGISTRY:
            raise ValueError(f"expected-state engine already registered for {mine_type}")
        EXPECTED_STATE_REGISTRY[mine_type] = factory
        return factory
    return decorator


def register_model(mine_type: str) -> Callable[[ResidualModelFactory], ResidualModelFactory]:
    def decorator(factory: ResidualModelFactory) -> ResidualModelFactory:
        if mine_type in MODEL_REGISTRY:
            raise ValueError(f"residual model already registered for {mine_type}")
        MODEL_REGISTRY[mine_type] = factory
        return factory
    return decorator


def get_pipeline(mine_type: str) -> tuple[ExpectedStateEngine, ResidualModel]:
    """Return the registered physics and residual pair for one configured mine type."""
    try:
        return EXPECTED_STATE_REGISTRY[mine_type](), MODEL_REGISTRY[mine_type]()
    except KeyError as error:
        raise ValueError(f"unsupported or incomplete mine type: {mine_type}") from error


class BordAndPillarExpectedStateStub:
    """Dispatch-safe placeholder; CPHSR and pillar mechanics are a later phase."""
    def monitoring_mode(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("bord_and_pillar expected-state model is not implemented yet")

    def planning_mode(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("bord_and_pillar expected-state model is not implemented yet")


# Imports are intentionally at module end: they avoid a registry/import cycle and
# make both configured mine types resolve before their advanced implementations land.
from pignn.expected_state.bord_and_pillar import BordAndPillarExpectedStateEngine
from pignn.expected_state.longwall import LongwallExpectedStateEngine
from pignn.model.gp_residual import GPResidualModel
from pignn.model.longwall_pignn import LongwallPIGNNModel

register_expected_state("longwall")(LongwallExpectedStateEngine)
register_model("longwall")(LongwallPIGNNModel)
register_expected_state("bord_and_pillar")(BordAndPillarExpectedStateEngine)
register_model("bord_and_pillar")(GPResidualModel)
