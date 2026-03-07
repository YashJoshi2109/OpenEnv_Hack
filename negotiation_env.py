"""Client for SalaryNegotiationArena.

Per official slides (page 12):
  from openenv import EnvClient, StepResult
  Subclass EnvClient[ActionT, ObsT, StateT]
  Implement: _step_payload, _parse_result, _parse_state
"""

from openenv import EnvClient, StepResult
from server.models import NegotiationAction, NegotiationObservation, NegotiationState


class NegotiationEnv(
    EnvClient[NegotiationAction, NegotiationObservation, NegotiationState]
):
    """Client for the SalaryNegotiationArena environment."""

    def _step_payload(self, action: NegotiationAction):
        """Serialize action to JSON for WebSocket."""
        return action.model_dump()

    def _parse_result(self, payload) -> StepResult:
        """Deserialize server response into StepResult."""
        obs = NegotiationObservation(**payload)
        return StepResult(
            observation=obs,
            reward=obs.reward,
            done=obs.done,
        )

    def _parse_state(self, payload) -> NegotiationState:
        """Deserialize state response."""
        return NegotiationState(**payload)
