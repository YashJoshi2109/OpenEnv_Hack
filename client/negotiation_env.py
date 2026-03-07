"""Client for SalaryNegotiationArena. EnvClient pattern from slides page 12."""
from openenv import EnvClient, StepResult
from server.models import NegotiationAction, NegotiationObservation, NegotiationState


class NegotiationEnv(EnvClient[NegotiationAction, NegotiationObservation, NegotiationState]):
    def _step_payload(self, action: NegotiationAction):
        return action.model_dump()

    def _parse_result(self, payload) -> StepResult:
        obs = NegotiationObservation(**payload)
        return StepResult(observation=obs, reward=obs.reward, done=obs.done)

    def _parse_state(self, payload) -> NegotiationState:
        return NegotiationState(**payload)
