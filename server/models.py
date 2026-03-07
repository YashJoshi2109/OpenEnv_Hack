"""Pydantic models for SalaryNegotiationArena.
Official OpenEnv base classes: Action, Observation, State.
"""
from openenv import Action, Observation, State
from typing import Optional


class NegotiationAction(Action):
    action_type: str = "propose"
    base_salary: Optional[int] = None
    equity: Optional[float] = None
    start_date: Optional[int] = None
    message: str = ""


class NegotiationObservation(Observation):
    turn: int = 0
    max_turns: int = 10
    phase: str = "negotiating"
    challenger_message: str = ""
    current_offer_salary: int = 0
    current_offer_equity: float = 0.0
    current_offer_start: int = 0
    agent_role: str = "candidate"
    expert_name: str = ""
    expert_style: str = ""
    done: bool = False
    reward: float = 0.0


class NegotiationState(State):
    episode_id: str = ""
    turn: int = 0
    max_turns: int = 10
    phase: str = "negotiating"
    profile_name: str = "Balanced"
    expert_name: str = ""
    expert_style: str = ""
    episode_count: int = 0
    current_offer_salary: int = 0
    current_offer_equity: float = 0.0
    current_offer_start: int = 0
    deal_count: int = 0
    total_episodes: int = 0
