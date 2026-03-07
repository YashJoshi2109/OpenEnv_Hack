"""Pydantic models for Salary Negotiation Arena.

Uses official OpenEnv base classes: Action, Observation, State.
Reference: Official OpenEnv Hackathon slides, page 10 (slide 7).
All Pydantic models live here — Rule #8.
"""

from openenv import Action, Observation, State
from typing import Optional


class NegotiationAction(Action):
    """Action the agent sends each turn."""
    action_type: str = "propose"  # propose, accept, reject, counter, walk_away
    base_salary: Optional[int] = None
    equity: Optional[float] = None
    start_date: Optional[int] = None
    message: str = ""


class NegotiationObservation(Observation):
    """Observation returned to the agent each turn."""
    turn: int = 0
    max_turns: int = 10
    phase: str = "negotiating"
    challenger_message: str = ""
    current_offer_salary: int = 0
    current_offer_equity: float = 0.0
    current_offer_start: int = 0
    agent_role: str = "candidate"
    done: bool = False
    reward: float = 0.0


class NegotiationState(State):
    """Full environment state tracked across the episode."""
    episode_id: str = ""
    turn: int = 0
    max_turns: int = 10
    phase: str = "negotiating"
    profile_name: str = "Balanced"
    episode_count: int = 0
    current_offer_salary: int = 0
    current_offer_equity: float = 0.0
    current_offer_start: int = 0
    deal_count: int = 0
    total_episodes: int = 0
