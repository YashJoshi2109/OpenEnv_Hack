"""Standalone reward functions for GRPO training. NEVER inside environment."""
import json, re

BASELINE = {"base_salary": 140000, "equity": 2.5, "start_date": 90}
PROFILES = [
    {"name": "Balanced", "salary_wt": 0.4, "equity_wt": 0.3, "start_wt": 0.3},
    {"name": "Cash-heavy", "salary_wt": 0.7, "equity_wt": 0.1, "start_wt": 0.2},
    {"name": "Equity-heavy", "salary_wt": 0.2, "equity_wt": 0.6, "start_wt": 0.2},
    {"name": "Fast-start", "salary_wt": 0.2, "equity_wt": 0.2, "start_wt": 0.6},
]


def reward_format(completion: str, **kw) -> float:
    """1.0 for valid action JSON, 0.0 otherwise."""
    try:
        m = re.search(r'\{[^{}]*\}', completion)
        if not m: return 0.0
        d = json.loads(m.group())
        if d.get("action_type") not in ("propose","accept","reject","counter","walk_away"): return 0.0
        if d["action_type"] in ("propose","counter"):
            return 1.0 if any(k in d for k in ("base_salary","equity","start_date")) else 0.5
        return 1.0
    except: return 0.0


def reward_negotiation(completion: str, **kw) -> float:
    """Outcome-based: +1 deal above baseline, +0.5 at baseline, -1 no deal."""
    env_state = kw.get("env_state", {})
    phase = env_state.get("phase", "negotiating")
    if phase in ("no_deal", "walked_away"): return -1.0
    if phase == "negotiating": return -0.1
    if phase == "deal_reached":
        offer = env_state.get("current_offer", {})
        s = min(offer.get("base_salary", 0) / BASELINE["base_salary"], 2.0)
        e = min(offer.get("equity", 0) / BASELINE["equity"], 2.0)
        d = max(0, 1.0 - (offer.get("start_date", 90) / BASELINE["start_date"] - 1.0))
        avg = (s + e + d) / 3
        turn = env_state.get("turn", 5)
        reward = 1.0 if avg > 1.0 else 0.5
        if turn < 5: reward += 0.2
        reward -= turn * 0.1
        return max(reward, -1.0)
    return 0.0


def reward_deal_quality(completion: str, **kw) -> float:
    """Snorkel-weighted utility bonus. +0.3 if weighted utility >= 0.5."""
    env_state = kw.get("env_state", {})
    if env_state.get("phase") != "deal_reached": return 0.0
    offer = env_state.get("current_offer", {})
    pidx = env_state.get("profile_idx", 0)
    p = PROFILES[pidx % len(PROFILES)]
    s_n = min(offer.get("base_salary", 0) / 200000, 1.0)
    e_n = min(offer.get("equity", 0) / 5.0, 1.0)
    d_n = max(0, 1.0 - offer.get("start_date", 180) / 180)
    wu = p["salary_wt"]*s_n + p["equity_wt"]*e_n + p["start_wt"]*d_n
    return 0.3 if wu >= 0.5 else 0.0


def compute_reward(completion: str, **kw) -> float:
    """Combined reward for GRPO."""
    rf = reward_format(completion, **kw)
    if rf == 0.0: return -0.5
    rn = reward_negotiation(completion, **kw)
    rq = reward_deal_quality(completion, **kw)
    return rf * 0.2 + rn * 0.5 + rq * 0.3
