"""SalaryNegotiationArena — OpenEnv MCPEnvironment.

Pattern from official hackathon slides pages 10-11 (slides 7-8):
  - Subclass MCPEnvironment
  - Define MCP tools with @mcp.tool
  - Implement reset(), step(), state property
  - Reward logic is EXTERNAL (reward.py) — Rule #3
"""

import uuid
import random

from openenv import MCPEnvironment
from mcp.server.fastmcp import FastMCP
from ..models import NegotiationAction, NegotiationObservation, NegotiationState


# ---------- Snorkel preference profiles (shift every 8 episodes) ----------
PREFERENCE_PROFILES = [
    {"name": "Balanced",     "salary_wt": 0.4, "equity_wt": 0.3, "start_wt": 0.3},
    {"name": "Cash-heavy",   "salary_wt": 0.7, "equity_wt": 0.1, "start_wt": 0.2},
    {"name": "Equity-heavy", "salary_wt": 0.2, "equity_wt": 0.6, "start_wt": 0.2},
    {"name": "Fast-start",   "salary_wt": 0.2, "equity_wt": 0.2, "start_wt": 0.6},
]

SALARY_RANGE = (80_000, 200_000)
EQUITY_RANGE = (0.0, 5.0)
START_DATE_RANGE = (14, 180)


class NegotiationEnvironment(MCPEnvironment):
    """
    Salary negotiation env: LLM agent negotiates base_salary, equity,
    start_date against a rule-based challenger.

    Snorkel hook: preferences shift every 8 episodes.
    Self-improvement: challenger escalates difficulty on win rate.
    """

    def __init__(self):
        mcp = FastMCP("negotiation_arena")

        @mcp.tool
        def propose(base_salary: int, equity: float, start_date: int, message: str = "") -> str:
            """Propose a compensation package to the employer."""
            action = NegotiationAction(
                action_type="propose", base_salary=base_salary,
                equity=equity, start_date=start_date, message=message,
            )
            obs = self.step(action)
            return f"Employer: {obs.challenger_message} | Phase: {obs.phase} | Turn: {obs.turn}/{obs.max_turns}"

        @mcp.tool
        def counter(base_salary: int, equity: float, start_date: int, message: str = "") -> str:
            """Counter the employer's offer with a new package."""
            action = NegotiationAction(
                action_type="counter", base_salary=base_salary,
                equity=equity, start_date=start_date, message=message,
            )
            obs = self.step(action)
            return f"Employer: {obs.challenger_message} | Phase: {obs.phase} | Turn: {obs.turn}/{obs.max_turns}"

        @mcp.tool
        def accept_offer(message: str = "I accept.") -> str:
            """Accept the current offer on the table."""
            action = NegotiationAction(action_type="accept", message=message)
            obs = self.step(action)
            return f"Employer: {obs.challenger_message} | Phase: {obs.phase}"

        @mcp.tool
        def reject_offer(message: str = "I reject.") -> str:
            """Reject the current offer and continue negotiating."""
            action = NegotiationAction(action_type="reject", message=message)
            obs = self.step(action)
            return f"Employer: {obs.challenger_message} | Phase: {obs.phase} | Turn: {obs.turn}/{obs.max_turns}"

        @mcp.tool
        def walk_away(message: str = "I'm walking away.") -> str:
            """Walk away from the negotiation entirely."""
            action = NegotiationAction(action_type="walk_away", message=message)
            obs = self.step(action)
            return f"Negotiation ended. Phase: {obs.phase}"

        super().__init__(mcp)

        # Internal state
        self._state = NegotiationState()
        self._episode_count = 0
        self._profile_idx = 0
        self._profile_switch_interval = 8
        self._challenger_target: dict = {}
        self._history: list[dict] = []
        self._deal_count = 0
        self._reset_count = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_profile(self) -> dict:
        return PREFERENCE_PROFILES[self._profile_idx % len(PREFERENCE_PROFILES)]

    def _gen_target(self) -> dict:
        return {
            "base_salary": random.randint(SALARY_RANGE[0], SALARY_RANGE[0] + 40_000),
            "equity": round(random.uniform(EQUITY_RANGE[0], 1.5), 2),
            "start_date": random.randint(START_DATE_RANGE[0], 60),
        }

    # ------------------------------------------------------------------
    # OpenEnv interface: reset, step, state
    # ------------------------------------------------------------------

    def reset(self, seed=None, episode_id=None, **kwargs) -> NegotiationObservation:
        """Reset for a new negotiation episode."""
        # Snorkel hook: shift profile every N episodes
        if self._episode_count > 0 and self._episode_count % self._profile_switch_interval == 0:
            self._profile_idx = (self._profile_idx + 1) % len(PREFERENCE_PROFILES)

        self._episode_count += 1
        self._reset_count += 1
        self._challenger_target = self._gen_target()
        self._history = []
        profile = self._get_profile()
        target = self._challenger_target

        self._state = NegotiationState(
            episode_id=episode_id or str(uuid.uuid4()),
            turn=0, max_turns=10, phase="negotiating",
            profile_name=profile["name"],
            episode_count=self._episode_count,
            current_offer_salary=target["base_salary"],
            current_offer_equity=target["equity"],
            current_offer_start=target["start_date"],
            deal_count=self._deal_count,
            total_episodes=self._episode_count,
        )

        return NegotiationObservation(
            turn=0, max_turns=10, phase="negotiating",
            challenger_message=(
                f"Welcome! Our initial offer: ${target['base_salary']:,}/yr, "
                f"{target['equity']}% equity, starting in {target['start_date']} days. "
                f"What are your thoughts?"
            ),
            current_offer_salary=target["base_salary"],
            current_offer_equity=target["equity"],
            current_offer_start=target["start_date"],
            agent_role="candidate", done=False, reward=0.0,
        )

    def step(self, action: NegotiationAction, timeout_s=None, **kwargs) -> NegotiationObservation:
        """Process agent action."""
        if self._state.phase != "negotiating":
            return self._make_obs(0.0)

        self._history.append({"turn": self._state.turn, "role": "agent", "action": action.model_dump()})
        a = action.action_type

        if a == "walk_away":
            self._state.phase = "walked_away"
        elif a == "accept":
            if self._state.current_offer_salary > 0:
                self._state.phase = "deal_reached"
                self._deal_count += 1
        elif a in ("propose", "counter"):
            self._state.current_offer_salary = action.base_salary or self._challenger_target["base_salary"]
            self._state.current_offer_equity = action.equity or self._challenger_target["equity"]
            self._state.current_offer_start = action.start_date or self._challenger_target["start_date"]
            if self._challenger_accepts():
                self._state.phase = "deal_reached"
                self._deal_count += 1

        self._state.turn += 1
        if self._state.turn >= self._state.max_turns and self._state.phase == "negotiating":
            self._state.phase = "no_deal"

        reward = self._step_reward()
        msg = self._challenger_msg()
        self._history.append({"turn": self._state.turn, "role": "challenger", "message": msg})
        self._state.deal_count = self._deal_count
        return self._make_obs(reward, msg)

    @property
    def state(self) -> NegotiationState:
        return self._state

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _make_obs(self, reward: float = 0.0, msg: str = "") -> NegotiationObservation:
        done = self._state.phase != "negotiating"
        return NegotiationObservation(
            turn=self._state.turn, max_turns=self._state.max_turns,
            phase=self._state.phase,
            challenger_message=msg or self._challenger_msg(),
            current_offer_salary=self._state.current_offer_salary,
            current_offer_equity=self._state.current_offer_equity,
            current_offer_start=self._state.current_offer_start,
            agent_role="candidate", done=done, reward=reward,
        )

    def _challenger_msg(self) -> str:
        p = self._state.phase
        if p == "deal_reached": return "Deal! Looking forward to working with you."
        if p == "no_deal": return "We couldn't reach an agreement."
        if p == "walked_away": return "The candidate walked away."
        t = self._challenger_target
        s, e, d = self._state.current_offer_salary, self._state.current_offer_equity, self._state.current_offer_start
        parts = []
        if s > t["base_salary"] + 30_000: parts.append("Salary is above budget.")
        elif s > t["base_salary"] + 10_000: parts.append("Salary is a bit high.")
        else: parts.append("Salary works.")
        if e > t["equity"] + 1.0: parts.append("Equity ask is too high.")
        else: parts.append("Equity is reasonable.")
        if d > t["start_date"] + 30: parts.append("We prefer earlier start.")
        else: parts.append("Start date works.")
        parts.append(f"Turn {self._state.turn}/{self._state.max_turns}.")
        return " ".join(parts)

    def _challenger_accepts(self) -> bool:
        t = self._challenger_target
        p = self._get_profile()
        s_sc = max(0, 1 - abs(self._state.current_offer_salary - t["base_salary"]) / 50_000)
        e_sc = max(0, 1 - abs(self._state.current_offer_equity - t["equity"]) / 3.0)
        d_sc = max(0, 1 - abs(self._state.current_offer_start - t["start_date"]) / 90)
        w = p["salary_wt"] * s_sc + p["equity_wt"] * e_sc + p["start_wt"] * d_sc
        threshold = min(0.7 + self._episode_count * 0.005, 0.95)
        return w >= threshold

    def _step_reward(self) -> float:
        p = self._state.phase
        if p == "deal_reached": return 1.0 / max(self._state.turn, 1)
        if p in ("no_deal", "walked_away"): return -1.0
        return -0.1
