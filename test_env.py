"""8 unit tests. Run: python -m pytest test_env.py -v"""
import pytest
from server.negotiation_environment import NegotiationEnvironment, EXPERT_PERSONAS
from server.models import NegotiationAction, NegotiationObservation, NegotiationState
from reward import reward_format, reward_negotiation, reward_deal_quality
from challenger import RuleBasedChallenger, ExpertChallenger, CurriculumManager

class TestEnv:
    def test_reset(self):
        e = NegotiationEnvironment(); o = e.reset()
        assert isinstance(o, NegotiationObservation) and o.turn==0 and o.phase=="negotiating"
    def test_step(self):
        e = NegotiationEnvironment(); e.reset()
        o = e.step(NegotiationAction(action_type="propose",base_salary=150000,equity=2.5,start_date=60))
        assert o.turn == 1
    def test_walk_away(self):
        e = NegotiationEnvironment(); e.reset()
        o = e.step(NegotiationAction(action_type="walk_away"))
        assert o.done and o.phase=="walked_away"
    def test_timeout(self):
        e = NegotiationEnvironment(); e.reset(); e._state.max_turns=3
        for _ in range(3): o = e.step(NegotiationAction(action_type="reject"))
        assert o.phase=="no_deal"
    def test_snorkel_shift(self):
        e = NegotiationEnvironment(); e._shift=8; i=e._pidx
        for _ in range(8): e.reset()
        assert e._pidx != i
    def test_expert_style_visible(self):
        e = NegotiationEnvironment(); o = e.reset()
        assert o.expert_style in ("analytical","aggressive","collaborative","bureaucratic","visionary")

class TestReward:
    def test_valid(self): assert reward_format('{"action_type":"propose","base_salary":150000}') == 1.0
    def test_invalid(self): assert reward_format("garbage") == 0.0

class TestChallenger:
    def test_expert(self):
        c = ExpertChallenger(0); c.set_target({"base_salary":100000,"equity":1.0,"start_date":30})
        r = c.respond({"base_salary":140000,"equity":2.0,"start_date":60}, 1, 10)
        assert "action_type" in r
    def test_curriculum(self):
        cm = CurriculumManager()
        cm.record("Sarah Chen — VP Engineering", 0.8)
        cm.record("Marcus Rivera — CFO", 0.1)
        w = cm.get_weights()
        assert w["Marcus Rivera — CFO"] > w["Sarah Chen — VP Engineering"]
