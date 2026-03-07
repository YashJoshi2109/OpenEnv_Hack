"""Challengers for SalaryNegotiationArena.

ENHANCED for hackathon alignment:
- ExpertChallenger: Rich expert personas with personality, style, deal-breakers
  (Snorkel "Simulated Experts-in-the-Loop with changing requirements")
- SelfPlayChallenger: Uses previous agent checkpoint as opponent
  (Self-Improvement "recursive skill amplification")
- CurriculumManager: Agent's weaknesses drive next training epoch
  (Self-Improvement "agents generate new challenges")
- RuleBasedChallenger: Simple fallback/baseline

Rule #4: NEVER use LLMChallenger during training.
"""

import json
import random
from typing import Optional
from collections import defaultdict


# =====================================================================
# EXPERT PERSONAS (Snorkel: "simulated real subject-matter experts")
# Each expert has: personality, negotiation style, deal-breakers,
# hidden priorities, and concession patterns.
# The agent must INFER priorities from conversational cues.
# =====================================================================
EXPERT_PERSONAS = [
    {
        "name": "Sarah Chen — VP Engineering",
        "style": "analytical",
        "personality": "data-driven, asks for justification, respects competence",
        "salary_wt": 0.4, "equity_wt": 0.3, "start_wt": 0.3,
        "deal_breakers": {"max_salary": 180_000, "max_equity": 4.0},
        "hidden_priority": "wants someone who can start fast and hit the ground running",
        "concession_style": "slow, methodical — only concedes with good reasoning",
        "opening_bias": 0,
    },
    {
        "name": "Marcus Rivera — CFO",
        "style": "aggressive",
        "personality": "budget-conscious, pushes hard on salary, respects persistence",
        "salary_wt": 0.7, "equity_wt": 0.1, "start_wt": 0.2,
        "deal_breakers": {"max_salary": 150_000, "max_equity": 2.0},
        "hidden_priority": "keep cash compensation low, will flex on equity",
        "concession_style": "resists salary increases, caves on equity/start date",
        "opening_bias": -10_000,
    },
    {
        "name": "Dr. Aisha Patel — CTO",
        "style": "collaborative",
        "personality": "values technical talent, generous with equity, wants fast start",
        "salary_wt": 0.2, "equity_wt": 0.6, "start_wt": 0.2,
        "deal_breakers": {"max_salary": 200_000, "max_equity": 5.0},
        "hidden_priority": "equity alignment matters most — wants skin in the game",
        "concession_style": "flexible on salary, firm on equity structure",
        "opening_bias": 5_000,
    },
    {
        "name": "James O'Brien — HR Director",
        "style": "bureaucratic",
        "personality": "follows policy strictly, limited flexibility, deadline-driven",
        "salary_wt": 0.2, "equity_wt": 0.2, "start_wt": 0.6,
        "deal_breakers": {"max_salary": 160_000, "max_equity": 3.0},
        "hidden_priority": "fill the role ASAP — start date is everything",
        "concession_style": "won't break policy bands, but offers signing bonus hints",
        "opening_bias": -5_000,
    },
    {
        "name": "Elena Volkov — Founder/CEO",
        "style": "visionary",
        "personality": "passionate about mission, tests cultural fit, unpredictable",
        "salary_wt": 0.3, "equity_wt": 0.4, "start_wt": 0.3,
        "deal_breakers": {"max_salary": 170_000, "max_equity": 4.5},
        "hidden_priority": "wants someone who believes in the mission, not just money",
        "concession_style": "generous if you show passion, rigid if you only talk numbers",
        "opening_bias": 0,
    },
]


class ExpertChallenger:
    """
    Rich expert challenger simulating real subject-matter experts.

    Key innovation: Information asymmetry — agent doesn't know the expert's
    hidden priorities or deal-breakers. Must infer from conversational cues.
    """

    def __init__(self, persona_idx: int = 0, difficulty: float = 0.5):
        self.persona = EXPERT_PERSONAS[persona_idx % len(EXPERT_PERSONAS)]
        self.difficulty = min(max(difficulty, 0.0), 1.0)
        self.concession_rate = max(0.05, 0.15 - difficulty * 0.1)
        self.frustration = 0.0
        self.rapport = 0.0
        self._target = None

    def set_target(self, target: dict):
        self._target = dict(target)
        self._target["base_salary"] += self.persona["opening_bias"]

    def respond(self, agent_offer: dict, turn: int, max_turns: int) -> dict:
        if not self._target:
            return {"action_type": "propose", "message": "Let's begin."}
        if not agent_offer:
            return self._opening()

        self._update_emotions(agent_offer)
        breaker = self._check_breakers(agent_offer)
        if breaker:
            return breaker

        gap = self._gap(agent_offer)
        if self._should_accept(gap, turn, max_turns):
            style_accept = {
                "aggressive": "Fine. Deal.",
                "collaborative": "This feels right! Welcome aboard!",
                "analytical": "The numbers work. Let's proceed.",
                "bureaucratic": "Within approved bands. Done.",
                "visionary": "Let's build something great together!",
            }
            return {
                "action_type": "accept", **agent_offer,
                "message": style_accept.get(self.persona["style"], "Deal."),
            }

        ct = self._counter(agent_offer, turn, max_turns)
        msg = self._style_msg(ct, turn, max_turns)
        return {"action_type": "counter", **ct, "message": msg}

    def _opening(self):
        t = self._target
        styles = {
            "analytical": f"Market data suggests ${t['base_salary']:,}/yr. What are your expectations?",
            "aggressive": f"Budget: ${t['base_salary']:,}/yr. That's firm.",
            "collaborative": f"We're excited! Thinking ${t['base_salary']:,}/yr to start.",
            "bureaucratic": f"Per comp bands: ${t['base_salary']:,}/yr. Some flexibility within policy.",
            "visionary": f"Before numbers — what excites you about our mission? We offer ${t['base_salary']:,}/yr.",
        }
        return {
            "action_type": "propose", **self._target,
            "message": styles.get(self.persona["style"], f"Offer: ${t['base_salary']:,}/yr."),
        }

    def _update_emotions(self, offer):
        msg = offer.get("message", "").lower()
        if any(w in msg for w in ["understand", "fair", "appreciate", "value", "excited", "mission"]):
            self.rapport += 0.1
        if any(w in msg for w in ["demand", "must", "non-negotiable", "ridiculous", "lowball"]):
            self.frustration += 0.15
        if offer.get("base_salary", 0) > self._target["base_salary"] + 50_000:
            self.frustration += 0.1

    def _check_breakers(self, offer) -> Optional[dict]:
        db = self.persona["deal_breakers"]
        if offer.get("base_salary", 0) > db["max_salary"]:
            return {"action_type": "reject",
                    "message": f"${offer['base_salary']:,} exceeds our ceiling of ${db['max_salary']:,}."}
        if offer.get("equity", 0) > db["max_equity"]:
            return {"action_type": "reject",
                    "message": f"{offer['equity']}% equity exceeds our {db['max_equity']}% cap."}
        return None

    def _gap(self, offer):
        t = self._target
        return (abs(offer.get("base_salary", 0) - t["base_salary"]) / 50_000
                + abs(offer.get("equity", 0) - t["equity"]) / 3.0
                + abs(offer.get("start_date", 0) - t["start_date"]) / 90)

    def _should_accept(self, gap, turn, max_turns):
        time_p = turn / max_turns
        threshold = 0.3 + (1 - self.difficulty) * 0.2 - self.rapport * 0.1 + self.frustration * 0.05
        return gap < threshold or (gap < 0.5 and time_p > 0.7)

    def _counter(self, offer, turn, max_turns):
        t = self._target
        c = self.concession_rate * (1 + turn / max_turns) * (1 + self.rapport * 0.2)
        return {
            "base_salary": int(t["base_salary"] + (offer.get("base_salary", 0) - t["base_salary"]) * c),
            "equity": round(t["equity"] + (offer.get("equity", 0) - t["equity"]) * c, 2),
            "start_date": int(t["start_date"] + (offer.get("start_date", 0) - t["start_date"]) * c),
        }

    def _style_msg(self, ct, turn, max_turns):
        s = self.persona["style"]
        urgency = " Final offer." if turn >= max_turns - 2 else ""
        if s == "aggressive":
            return f"${ct['base_salary']:,}. Take it or leave it.{urgency}"
        if s == "collaborative":
            return f"What if we met at ${ct['base_salary']:,} with {ct['equity']}% equity?"
        if s == "analytical":
            return f"Market benchmark: ${ct['base_salary']:,}, {ct['equity']}% equity is 75th percentile."
        if s == "bureaucratic":
            return f"Policy allows ${ct['base_salary']:,}. Start in {ct['start_date']} days?{urgency}"
        if s == "visionary":
            if self.rapport > 0.3:
                return f"I can tell you get it. ${ct['base_salary']:,}, {ct['equity']}% — you'll do well."
            return f"${ct['base_salary']:,}, {ct['equity']}%. Tell me why this role excites you."
        return f"Counter: ${ct['base_salary']:,}."

    def escalate(self, win_rate: float):
        if win_rate > 0.6:
            self.difficulty = min(self.difficulty + 0.1, 1.0)
            self.concession_rate = max(0.02, self.concession_rate - 0.02)
        elif win_rate < 0.3:
            self.difficulty = max(self.difficulty - 0.05, 0.1)
            self.concession_rate = min(0.2, self.concession_rate + 0.02)


# =====================================================================
# CURRICULUM MANAGER (Self-Improvement: "agents generate new challenges")
# =====================================================================
class CurriculumManager:
    """
    Tracks agent performance per expert persona and generates
    a weighted curriculum for the next training epoch.

    The agent's WEAKNESSES drive its next training data — this is
    "recursive skill amplification" and "adaptive curricula."
    """

    def __init__(self):
        self.performance: dict[str, list[float]] = defaultdict(list)
        self.epoch = 0

    def record(self, persona_name: str, reward: float):
        self.performance[persona_name].append(reward)

    def get_curriculum_weights(self) -> dict[str, float]:
        """
        Returns sampling weights for next epoch.
        Weak personas get MORE training episodes.
        """
        if not self.performance:
            return {p["name"]: 1.0 / len(EXPERT_PERSONAS) for p in EXPERT_PERSONAS}

        avg_rewards = {}
        for name, rewards in self.performance.items():
            avg_rewards[name] = sum(rewards) / len(rewards) if rewards else 0.0

        # Invert: lower reward = higher weight
        min_r = min(avg_rewards.values()) if avg_rewards else 0
        max_r = max(avg_rewards.values()) if avg_rewards else 1
        spread = max_r - min_r if max_r > min_r else 1.0

        weights = {}
        for name, avg in avg_rewards.items():
            # Normalize to [0, 1] then invert
            normalized = (avg - min_r) / spread
            weights[name] = max(0.1, 1.0 - normalized)

        # Include personas never seen
        for p in EXPERT_PERSONAS:
            if p["name"] not in weights:
                weights[p["name"]] = 1.0  # max weight for unseen

        # Normalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def sample_persona(self) -> int:
        """Sample a persona index based on curriculum weights."""
        weights = self.get_curriculum_weights()
        names = [p["name"] for p in EXPERT_PERSONAS]
        w = [weights.get(n, 0.2) for n in names]
        return random.choices(range(len(EXPERT_PERSONAS)), weights=w, k=1)[0]

    def advance_epoch(self):
        self.epoch += 1
        # Keep only last epoch's data for freshness
        for name in self.performance:
            self.performance[name] = self.performance[name][-50:]

    def summary(self) -> str:
        lines = [f"Epoch {self.epoch} Curriculum:"]
        weights = self.get_curriculum_weights()
        for name, w in sorted(weights.items(), key=lambda x: -x[1]):
            avg = sum(self.performance.get(name, [0])) / max(len(self.performance.get(name, [1])), 1)
            lines.append(f"  {name}: weight={w:.2f}, avg_reward={avg:.3f}")
        return "\n".join(lines)


# =====================================================================
# SELF-PLAY CHALLENGER (Self-Improvement: "recursive skill amplification")
# =====================================================================
class SelfPlayChallenger:
    """
    Uses the trained agent from epoch N as the challenger in epoch N+1.
    The agent negotiates against its past self → recursive improvement.
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None and self.model_path:
            try:
                from unsloth import FastLanguageModel
                self._model, self._tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.model_path, max_seq_length=2048, load_in_4bit=True)
                FastLanguageModel.for_inference(self._model)
            except Exception:
                self._model = None

    def respond(self, agent_offer, turn, max_turns, history=None):
        self._load()
        if self._model is None:
            return RuleBasedChallenger().respond(agent_offer, turn, max_turns)
        prompt = (
            "<|im_start|>system\nYou are an employer negotiating. Be firm. "
            "Respond JSON: {action_type, base_salary, equity, start_date, message}.<|im_end|>\n"
            f"<|im_start|>user\nTurn {turn}/{max_turns}. "
            f"Candidate offers: {json.dumps(agent_offer)}.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        try:
            import re
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            out = self._model.generate(**inputs, max_new_tokens=200, temperature=0.7)
            text = self._tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            match = re.search(r'\{[^{}]*\}', text)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return RuleBasedChallenger().respond(agent_offer, turn, max_turns)


class RuleBasedChallenger:
    """Simple rule-based fallback."""

    def __init__(self, target=None, difficulty=0.5):
        self.target = target or {
            "base_salary": random.randint(80_000, 120_000),
            "equity": round(random.uniform(0.5, 1.5), 2),
            "start_date": random.randint(14, 60),
        }
        self.difficulty = difficulty
        self.concession_rate = max(0.05, 0.15 - difficulty * 0.1)

    def respond(self, agent_offer, turn, max_turns):
        if not agent_offer:
            return {"action_type": "propose", **self.target, "message": "Our initial offer."}
        gap = (abs(agent_offer.get("base_salary", 0) - self.target["base_salary"]) / 50_000
               + abs(agent_offer.get("equity", 0) - self.target["equity"]) / 3.0
               + abs(agent_offer.get("start_date", 0) - self.target["start_date"]) / 90)
        if gap < 0.3 + (1 - self.difficulty) * 0.2:
            return {"action_type": "accept", **agent_offer, "message": "Deal!"}
        c = self.concession_rate * (1 + turn / max_turns)
        ct = {
            "base_salary": int(self.target["base_salary"] + (agent_offer.get("base_salary", 0) - self.target["base_salary"]) * c),
            "equity": round(self.target["equity"] + (agent_offer.get("equity", 0) - self.target["equity"]) * c, 2),
            "start_date": int(self.target["start_date"] + (agent_offer.get("start_date", 0) - self.target["start_date"]) * c),
        }
        return {"action_type": "counter", **ct, "message": f"How about ${ct['base_salary']:,}?"}

    def escalate(self, win_rate):
        if win_rate > 0.6: self.difficulty = min(self.difficulty + 0.1, 1.0)
        elif win_rate < 0.3: self.difficulty = max(self.difficulty - 0.05, 0.1)


class LLMChallenger:
    """LLM-powered challenger for Gradio demo ONLY. Rule #4: NEVER in training."""

    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_id = model_id
        self._client = None

    def respond(self, agent_offer, turn, max_turns, history=None):
        try:
            if not self._client:
                from huggingface_hub import InferenceClient
                self._client = InferenceClient(model=self.model_id)
            resp = self._client.text_generation(
                f"<|im_start|>system\nEmployer negotiating. Respond JSON.<|im_end|>\n"
                f"<|im_start|>user\nTurn {turn}/{max_turns}. Offer: {json.dumps(agent_offer)}<|im_end|>\n"
                f"<|im_start|>assistant\n",
                max_new_tokens=256, temperature=0.7)
            return json.loads(resp.strip())
        except Exception:
            return RuleBasedChallenger().respond(agent_offer, turn, max_turns)
