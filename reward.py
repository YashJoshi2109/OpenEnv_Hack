"""Standalone reward computation for SalaryNegotiationArena.

Rule #3: Reward logic MUST live here, never inside the environment file.

Three reward functions for GRPO training:
  1. reward_format — checks action JSON structure
  2. reward_negotiation — outcome-based reward
  3. reward_deal_quality — Snorkel-weighted utility bonus
"""

import json
import re
from typing import Optional


# ---------- Baseline midpoints (50/50 fair split) ----------
BASELINE = {
    "base_salary": 140_000,
    "equity": 2.5,
    "start_date": 90,
}

# ---------- Snorkel Preference Profiles ----------
PREFERENCE_PROFILES = [
    {"name": "Balanced",     "salary_wt": 0.4, "equity_wt": 0.3, "start_wt": 0.3},
    {"name": "Cash-heavy",   "salary_wt": 0.7, "equity_wt": 0.1, "start_wt": 0.2},
    {"name": "Equity-heavy", "salary_wt": 0.2, "equity_wt": 0.6, "start_wt": 0.2},
    {"name": "Fast-start",   "salary_wt": 0.2, "equity_wt": 0.2, "start_wt": 0.6},
]


def reward_format(completion: str, **kwargs) -> float:
    """
    Reward for correct action JSON format.

    Returns:
        1.0 if valid JSON with required keys, 0.0 otherwise.
    """
    try:
        # Try to extract JSON from the completion
        json_match = re.search(r'\{[^{}]*\}', completion)
        if not json_match:
            return 0.0

        parsed = json.loads(json_match.group())

        # Must have action_type
        if "action_type" not in parsed:
            return 0.0

        valid_actions = {"propose", "accept", "reject", "counter", "walk_away"}
        if parsed["action_type"] not in valid_actions:
            return 0.0

        # If proposing/countering, should have at least one resource field
        if parsed["action_type"] in ("propose", "counter"):
            has_resource = any(
                k in parsed for k in ("base_salary", "equity", "start_date")
            )
            return 1.0 if has_resource else 0.5

        return 1.0

    except (json.JSONDecodeError, AttributeError):
        return 0.0


def reward_negotiation(completion: str, **kwargs) -> float:
    """
    Outcome-based reward for the negotiation.

    Reward table:
        Deal above 50/50 baseline, early close  → +1.0
        Deal at baseline                         → +0.5
        Snorkel bonus (weighted utility ≥ 0.5)   → +0.3
        Per-step penalty                         → -0.1
        No deal / timeout / walk_away            → -1.0

    Args:
        completion: The agent's action string
        **kwargs: Must include 'env_state' dict with phase, turn, current_offer info.
    """
    env_state = kwargs.get("env_state", {})
    phase = env_state.get("phase", "negotiating")
    turn = env_state.get("turn", 0)
    max_turns = env_state.get("max_turns", 10)
    current_offer = env_state.get("current_offer", None)

    # Terminal penalties
    if phase in ("no_deal", "walked_away"):
        return -1.0

    # Still negotiating — per-step penalty
    if phase == "negotiating":
        return -0.1

    # Deal reached — compute quality
    if phase == "deal_reached" and current_offer:
        return _compute_deal_reward(current_offer, turn, max_turns)

    return 0.0


def _compute_deal_reward(offer: dict, turn: int, max_turns: int) -> float:
    """Compute reward for a completed deal."""
    reward = 0.0

    # Normalize each resource to [0, 1] relative to baseline
    salary_score = min(offer.get("base_salary", 0) / BASELINE["base_salary"], 2.0)
    equity_score = min(offer.get("equity", 0) / BASELINE["equity"], 2.0)
    # Lower start_date is better for candidate
    start_raw = offer.get("start_date", BASELINE["start_date"])
    start_score = max(0, 1.0 - (start_raw / BASELINE["start_date"] - 1.0))

    avg_score = (salary_score + equity_score + start_score) / 3.0

    # Above baseline?
    if avg_score > 1.0:
        reward = 1.0
        # Early close bonus
        if turn < max_turns * 0.5:
            reward += 0.2
    else:
        reward = 0.5

    # Step penalty still applies
    reward -= turn * 0.1

    return max(reward, -1.0)


def reward_deal_quality(completion: str, **kwargs) -> float:
    """
    Snorkel-weighted utility bonus.

    Returns +0.3 if the deal's weighted utility ≥ 0.5 under
    the current preference profile, else 0.0.
    """
    env_state = kwargs.get("env_state", {})
    phase = env_state.get("phase", "")
    current_offer = env_state.get("current_offer", None)
    profile_idx = env_state.get("profile_idx", 0)

    if phase != "deal_reached" or not current_offer:
        return 0.0

    profile = PREFERENCE_PROFILES[profile_idx % len(PREFERENCE_PROFILES)]

    # Normalize to [0, 1]
    salary_norm = min(current_offer.get("base_salary", 0) / 200_000, 1.0)
    equity_norm = min(current_offer.get("equity", 0) / 5.0, 1.0)
    start_norm = max(0, 1.0 - current_offer.get("start_date", 180) / 180)

    weighted_utility = (
        profile["salary_wt"] * salary_norm
        + profile["equity_wt"] * equity_norm
        + profile["start_wt"] * start_norm
    )

    return 0.3 if weighted_utility >= 0.5 else 0.0


def compute_reward(completion: str, **kwargs) -> float:
    """
    Combined reward function for all three components.
    Used as the single entry point during training.
    """
    r_format = reward_format(completion, **kwargs)
    r_negotiation = reward_negotiation(completion, **kwargs)
    r_quality = reward_deal_quality(completion, **kwargs)

    # Format is a gate: if 0, everything else is penalized
    if r_format == 0.0:
        return -0.5

    return r_format * 0.2 + r_negotiation * 0.5 + r_quality * 0.3
