"""SalaryNegotiationArena — OpenEnv MCPEnvironment.

Official pattern: MCPEnvironment + FastMCP @mcp.tool decorators.
5 expert personas with hidden priorities (Snorkel hook).
Preference drift every 8 episodes.
Information asymmetry: agent sees style hints but not deal-breakers.
Reward: 3-component formula from reward.py (Format + Negotiation + DealQuality).
"""
import uuid
import json
import random
from openenv.core import MCPEnvironment
from mcp.server.fastmcp import FastMCP
from .models import NegotiationAction, NegotiationObservation, NegotiationState
from reward import reward_format, reward_negotiation, reward_deal_quality

EXPERT_PERSONAS = [
    {"name": "Sarah Chen — VP Engineering", "style": "analytical",
     "salary_wt": 0.4, "equity_wt": 0.3, "start_wt": 0.3,
     "deal_breakers": {"max_salary": 180000, "max_equity": 4.0},
     "hidden_priority": "fast start", "opening_bias": 0},
    {"name": "Marcus Rivera — CFO", "style": "aggressive",
     "salary_wt": 0.7, "equity_wt": 0.1, "start_wt": 0.2,
     "deal_breakers": {"max_salary": 150000, "max_equity": 2.0},
     "hidden_priority": "low cash", "opening_bias": -10000},
    {"name": "Dr. Aisha Patel — CTO", "style": "collaborative",
     "salary_wt": 0.2, "equity_wt": 0.6, "start_wt": 0.2,
     "deal_breakers": {"max_salary": 200000, "max_equity": 5.0},
     "hidden_priority": "equity alignment", "opening_bias": 5000},
    {"name": "James O'Brien — HR Director", "style": "bureaucratic",
     "salary_wt": 0.2, "equity_wt": 0.2, "start_wt": 0.6,
     "deal_breakers": {"max_salary": 160000, "max_equity": 3.0},
     "hidden_priority": "fill ASAP", "opening_bias": -5000},
    {"name": "Elena Volkov — Founder/CEO", "style": "visionary",
     "salary_wt": 0.3, "equity_wt": 0.4, "start_wt": 0.3,
     "deal_breakers": {"max_salary": 170000, "max_equity": 4.5},
     "hidden_priority": "mission alignment", "opening_bias": 0},
]
SALARY_RANGE = (80000, 200000)


class NegotiationEnvironment(MCPEnvironment):
    """Salary negotiation env with expert personas and self-improvement hooks."""

    def __init__(self):
        mcp = FastMCP("negotiation_arena")

        @mcp.tool()
        def propose(base_salary: int, equity: float, start_date: int, message: str = "") -> str:
            """Propose a compensation package."""
            obs = self.step(NegotiationAction(action_type="propose",
                base_salary=base_salary, equity=equity, start_date=start_date, message=message))
            return f"Employer: {obs.challenger_message} | Phase: {obs.phase} | Turn: {obs.turn}/{obs.max_turns}"

        @mcp.tool()
        def counter(base_salary: int, equity: float, start_date: int, message: str = "") -> str:
            """Counter the employer's offer."""
            obs = self.step(NegotiationAction(action_type="counter",
                base_salary=base_salary, equity=equity, start_date=start_date, message=message))
            return f"Employer: {obs.challenger_message} | Phase: {obs.phase} | Turn: {obs.turn}/{obs.max_turns}"

        @mcp.tool()
        def accept_offer(message: str = "I accept.") -> str:
            """Accept the current offer."""
            obs = self.step(NegotiationAction(action_type="accept", message=message))
            return f"Employer: {obs.challenger_message} | Phase: {obs.phase}"

        @mcp.tool()
        def reject_offer(message: str = "I reject.") -> str:
            """Reject and continue negotiating."""
            obs = self.step(NegotiationAction(action_type="reject", message=message))
            return f"Employer: {obs.challenger_message} | Phase: {obs.phase} | Turn: {obs.turn}/{obs.max_turns}"

        @mcp.tool()
        def walk_away(message: str = "I'm walking away.") -> str:
            """Walk away from the negotiation."""
            obs = self.step(NegotiationAction(action_type="walk_away", message=message))
            return f"Negotiation ended. Phase: {obs.phase}"

        super().__init__(mcp)
        self._state = NegotiationState()
        self._ep = 0
        self._pidx = 0
        self._shift = 8
        self._target: dict = {}
        self._history: list = []
        self._deals = 0
        self._rapport = 0.0
        self._frustration = 0.0

    def _persona(self): return EXPERT_PERSONAS[self._pidx % len(EXPERT_PERSONAS)]
    def _gen_target(self):
        return {"base_salary": random.randint(80000, 120000),
                "equity": round(random.uniform(0.5, 1.5), 2),
                "start_date": random.randint(14, 60)}

    def reset(self, seed=None, episode_id=None, **kw) -> NegotiationObservation:
        self._ep += 1
        if self._ep > 1 and self._ep % self._shift == 0:
            self._pidx = (self._pidx + 1) % len(EXPERT_PERSONAS)
        self._target = self._gen_target()
        self._history = []
        self._rapport = 0.0
        self._frustration = 0.0
        p = self._persona()
        t = self._target
        t["base_salary"] += p["opening_bias"]
        self._state = NegotiationState(
            episode_id=episode_id or str(uuid.uuid4()),
            turn=0, max_turns=10, phase="negotiating",
            profile_name=p["name"], expert_name=p["name"], expert_style=p["style"],
            episode_count=self._ep,
            current_offer_salary=t["base_salary"], current_offer_equity=t["equity"],
            current_offer_start=t["start_date"], deal_count=self._deals, total_episodes=self._ep)
        style_openings = {
            "analytical": f"Market data suggests ${t['base_salary']:,}/yr. What are your expectations?",
            "aggressive": f"Budget: ${t['base_salary']:,}/yr. That's firm.",
            "collaborative": f"We're excited about you! Thinking ${t['base_salary']:,}/yr.",
            "bureaucratic": f"Per comp bands: ${t['base_salary']:,}/yr. Some flexibility within policy.",
            "visionary": f"Before numbers — what excites you about our mission? We offer ${t['base_salary']:,}/yr."}
        return NegotiationObservation(
            turn=0, max_turns=10, phase="negotiating",
            challenger_message=style_openings.get(p["style"], f"Offer: ${t['base_salary']:,}/yr."),
            current_offer_salary=t["base_salary"], current_offer_equity=t["equity"],
            current_offer_start=t["start_date"], agent_role="candidate",
            expert_name=p["name"], expert_style=p["style"], done=False, reward=0.0)

    def _step_impl(self, action, timeout_s=None, **kw):
        if not isinstance(action, NegotiationAction):
            action = NegotiationAction(**action) if isinstance(action, dict) else action
        if self._state.phase != "negotiating": return self._obs(0.0)
        self._history.append({"turn": self._state.turn, "role": "agent", "action": action.model_dump()})
        self._update_emotions(action)
        a = action.action_type
        if a == "walk_away":
            self._state.phase = "walked_away"
        elif a == "accept":
            if self._state.current_offer_salary > 0:
                self._state.phase = "deal_reached"; self._deals += 1
        elif a in ("propose", "counter"):
            self._state.current_offer_salary = action.base_salary or self._target["base_salary"]
            self._state.current_offer_equity = action.equity or self._target["equity"]
            self._state.current_offer_start = action.start_date or self._target["start_date"]
            breaker = self._check_breakers()
            if breaker:
                self._state.phase = "no_deal"
            elif self._challenger_accepts():
                self._state.phase = "deal_reached"; self._deals += 1
        self._state.turn += 1
        if self._state.turn >= self._state.max_turns and self._state.phase == "negotiating":
            self._state.phase = "no_deal"
        reward = self._step_reward()
        msg = self._challenger_msg()
        self._history.append({"turn": self._state.turn, "role": "challenger", "message": msg})
        self._state.deal_count = self._deals
        return self._obs(reward, msg)

    @property
    def state(self) -> NegotiationState: return self._state

    def _obs(self, reward=0.0, msg="") -> NegotiationObservation:
        done = self._state.phase != "negotiating"
        p = self._persona()
        return NegotiationObservation(
            turn=self._state.turn, max_turns=self._state.max_turns, phase=self._state.phase,
            challenger_message=msg or self._challenger_msg(),
            current_offer_salary=self._state.current_offer_salary,
            current_offer_equity=self._state.current_offer_equity,
            current_offer_start=self._state.current_offer_start,
            agent_role="candidate", expert_name=p["name"], expert_style=p["style"],
            done=done, reward=reward)

    def _update_emotions(self, action):
        msg = (action.message or "").lower()
        if any(w in msg for w in ["understand", "fair", "appreciate", "value", "excited", "mission"]):
            self._rapport += 0.1
        if any(w in msg for w in ["demand", "must", "non-negotiable", "ridiculous"]):
            self._frustration += 0.15
        if (action.base_salary or 0) > self._target["base_salary"] + 50000:
            self._frustration += 0.1

    def _check_breakers(self) -> bool:
        db = self._persona()["deal_breakers"]
        s = self._state.current_offer_salary
        e = self._state.current_offer_equity
        return s > db["max_salary"] or e > db["max_equity"]

    def _challenger_accepts(self) -> bool:
        t, p = self._target, self._persona()
        s_sc = max(0, 1 - abs(self._state.current_offer_salary - t["base_salary"]) / 50000)
        e_sc = max(0, 1 - abs(self._state.current_offer_equity - t["equity"]) / 3.0)
        d_sc = max(0, 1 - abs(self._state.current_offer_start - t["start_date"]) / 90)
        w = p["salary_wt"]*s_sc + p["equity_wt"]*e_sc + p["start_wt"]*d_sc
        threshold = min(0.7 + self._ep*0.005, 0.95) - self._rapport*0.1 + self._frustration*0.05
        return w >= threshold

    def _challenger_msg(self) -> str:
        ph = self._state.phase
        if ph == "deal_reached": return "Deal! Looking forward to working with you."
        if ph == "no_deal": return "We couldn't reach an agreement."
        if ph == "walked_away": return "The candidate walked away."
        t, p = self._target, self._persona()
        s, e, d = self._state.current_offer_salary, self._state.current_offer_equity, self._state.current_offer_start
        parts = []
        if s > t["base_salary"]+30000: parts.append("Salary is above budget.")
        elif s > t["base_salary"]+10000: parts.append("Salary is a bit high.")
        else: parts.append("Salary works.")
        if e > t["equity"]+1.0: parts.append("Equity ask is too high.")
        else: parts.append("Equity is reasonable.")
        if d > t["start_date"]+30: parts.append("We prefer earlier start.")
        else: parts.append("Start date works.")
        style_hints = {"analytical": "Can you justify with data?", "aggressive": "That's our limit.",
            "collaborative": "Let's find middle ground.", "bureaucratic": "Policy constrains us.",
            "visionary": "Tell me about your passion for this."}
        parts.append(style_hints.get(p["style"], ""))
        parts.append(f"Turn {self._state.turn}/{self._state.max_turns}.")
        return " ".join(parts)

    def _step_reward(self) -> float:
        """3-component reward matching reward.py: 0.2×Format + 0.5×Negotiation + 0.3×DealQuality."""
        s = self._state
        action_json = json.dumps({
            "action_type": "propose",
            "base_salary": s.current_offer_salary,
            "equity": s.current_offer_equity,
            "start_date": s.current_offer_start,
        })
        env_state = {
            "phase": s.phase,
            "turn": s.turn,
            "max_turns": s.max_turns,
            "current_offer": {
                "base_salary": s.current_offer_salary,
                "equity": s.current_offer_equity,
                "start_date": s.current_offer_start,
            },
            "profile_idx": self._pidx,
        }
        rf = reward_format(action_json, env_state=env_state)
        if rf == 0.0:
            return -0.5
        rn = reward_negotiation(action_json, env_state=env_state)
        rq = reward_deal_quality(action_json, env_state=env_state)
        return rf * 0.2 + rn * 0.5 + rq * 0.3
