"""Challengers: ExpertChallenger, CurriculumManager, SelfPlayChallenger, RuleBasedChallenger, LLMChallenger.
Rule #4: NEVER use LLMChallenger during training."""
import json, random
from typing import Optional
from collections import defaultdict

EXPERT_PERSONAS = [
    {"name":"Sarah Chen — VP Engineering","style":"analytical","salary_wt":0.4,"equity_wt":0.3,"start_wt":0.3,
     "deal_breakers":{"max_salary":180000,"max_equity":4.0},"opening_bias":0},
    {"name":"Marcus Rivera — CFO","style":"aggressive","salary_wt":0.7,"equity_wt":0.1,"start_wt":0.2,
     "deal_breakers":{"max_salary":150000,"max_equity":2.0},"opening_bias":-10000},
    {"name":"Dr. Aisha Patel — CTO","style":"collaborative","salary_wt":0.2,"equity_wt":0.6,"start_wt":0.2,
     "deal_breakers":{"max_salary":200000,"max_equity":5.0},"opening_bias":5000},
    {"name":"James O'Brien — HR Director","style":"bureaucratic","salary_wt":0.2,"equity_wt":0.2,"start_wt":0.6,
     "deal_breakers":{"max_salary":160000,"max_equity":3.0},"opening_bias":-5000},
    {"name":"Elena Volkov — Founder/CEO","style":"visionary","salary_wt":0.3,"equity_wt":0.4,"start_wt":0.3,
     "deal_breakers":{"max_salary":170000,"max_equity":4.5},"opening_bias":0},
]

class ExpertChallenger:
    def __init__(self, persona_idx=0, difficulty=0.5):
        self.persona = EXPERT_PERSONAS[persona_idx % len(EXPERT_PERSONAS)]
        self.difficulty = min(max(difficulty, 0.0), 1.0)
        self.concession_rate = max(0.05, 0.15 - difficulty * 0.1)
        self.frustration = 0.0
        self.rapport = 0.0
        self._target = None

    def set_target(self, target): self._target = dict(target); self._target["base_salary"] += self.persona["opening_bias"]

    def respond(self, offer, turn, max_turns):
        if not self._target or not offer: return self._opening()
        self._emotions(offer)
        db = self.persona["deal_breakers"]
        if offer.get("base_salary",0) > db["max_salary"]:
            return {"action_type":"reject","message":f"${offer['base_salary']:,} exceeds our ${db['max_salary']:,} ceiling."}
        if offer.get("equity",0) > db["max_equity"]:
            return {"action_type":"reject","message":f"{offer['equity']}% exceeds our {db['max_equity']}% cap."}
        gap = self._gap(offer)
        if self._accept(gap, turn, max_turns):
            msgs = {"aggressive":"Fine. Deal.","collaborative":"Welcome aboard!","analytical":"Numbers work.",
                    "bureaucratic":"Within bands. Done.","visionary":"Let's build something great!"}
            return {"action_type":"accept",**offer,"message":msgs.get(self.persona["style"],"Deal.")}
        ct = self._counter(offer, turn, max_turns)
        return {"action_type":"counter",**ct,"message":self._msg(ct, turn, max_turns)}

    def _opening(self):
        if not self._target: return {"action_type":"propose","message":"Let's begin."}
        t = self._target
        return {"action_type":"propose","base_salary":t["base_salary"],"equity":t["equity"],
                "start_date":t["start_date"],"message":f"Our offer: ${t['base_salary']:,}/yr."}

    def _emotions(self, offer):
        msg = offer.get("message","").lower()
        if any(w in msg for w in ["understand","fair","appreciate","excited","mission"]): self.rapport += 0.1
        if any(w in msg for w in ["demand","must","ridiculous"]): self.frustration += 0.15

    def _gap(self, offer):
        t = self._target
        return (abs(offer.get("base_salary",0)-t["base_salary"])/50000 +
                abs(offer.get("equity",0)-t["equity"])/3.0 +
                abs(offer.get("start_date",0)-t["start_date"])/90)

    def _accept(self, gap, turn, mt):
        tp = turn / mt
        thr = 0.3 + (1-self.difficulty)*0.2 - self.rapport*0.1 + self.frustration*0.05
        return gap < thr or (gap < 0.5 and tp > 0.7)

    def _counter(self, offer, turn, mt):
        t = self._target; c = self.concession_rate * (1 + turn/mt) * (1 + self.rapport*0.2)
        return {"base_salary":int(t["base_salary"]+(offer.get("base_salary",0)-t["base_salary"])*c),
                "equity":round(t["equity"]+(offer.get("equity",0)-t["equity"])*c,2),
                "start_date":int(t["start_date"]+(offer.get("start_date",0)-t["start_date"])*c)}

    def _msg(self, ct, turn, mt):
        s = self.persona["style"]; urg = " Final offer." if turn >= mt-2 else ""
        if s=="aggressive": return f"${ct['base_salary']:,}. Take or leave.{urg}"
        if s=="collaborative": return f"Meet at ${ct['base_salary']:,}, {ct['equity']}%?"
        if s=="analytical": return f"Benchmark: ${ct['base_salary']:,}, {ct['equity']}% equity."
        if s=="bureaucratic": return f"Policy: ${ct['base_salary']:,}. Start {ct['start_date']}d?{urg}"
        if s=="visionary": return f"${ct['base_salary']:,}, {ct['equity']}%. Why does this excite you?"
        return f"Counter: ${ct['base_salary']:,}."

    def escalate(self, wr):
        if wr > 0.6: self.difficulty = min(self.difficulty+0.1,1.0); self.concession_rate = max(0.02,self.concession_rate-0.02)
        elif wr < 0.3: self.difficulty = max(self.difficulty-0.05,0.1)

class CurriculumManager:
    """Tracks reward per persona. Weak personas get MORE training. Agent failures drive curriculum."""
    def __init__(self): self.perf = defaultdict(list); self.epoch = 0
    def record(self, name, reward): self.perf[name].append(reward)
    def get_weights(self):
        if not self.perf: return {p["name"]:1/len(EXPERT_PERSONAS) for p in EXPERT_PERSONAS}
        avgs = {n: sum(r)/len(r) for n,r in self.perf.items() if r}
        mn, mx = min(avgs.values(),default=0), max(avgs.values(),default=1)
        sp = mx - mn if mx > mn else 1.0
        w = {n: max(0.1, 1.0-(a-mn)/sp) for n,a in avgs.items()}
        for p in EXPERT_PERSONAS:
            if p["name"] not in w: w[p["name"]] = 1.0
        t = sum(w.values()); return {k:v/t for k,v in w.items()}
    def sample_persona(self):
        w = self.get_weights(); names = [p["name"] for p in EXPERT_PERSONAS]
        wts = [w.get(n,0.2) for n in names]
        return random.choices(range(len(EXPERT_PERSONAS)), weights=wts, k=1)[0]
    def advance(self): self.epoch += 1; [self.perf.__setitem__(n, self.perf[n][-50:]) for n in self.perf]

class SelfPlayChallenger:
    """Epoch N model becomes epoch N+1 opponent."""
    def __init__(self, model_path=None): self.model_path = model_path; self._m = None; self._t = None
    def respond(self, offer, turn, mt, history=None):
        if self._m is None and self.model_path:
            try:
                from unsloth import FastLanguageModel
                self._m, self._t = FastLanguageModel.from_pretrained(model_name=self.model_path, max_seq_length=2048, load_in_4bit=True)
                FastLanguageModel.for_inference(self._m)
            except: pass
        if self._m is None: return RuleBasedChallenger().respond(offer, turn, mt)
        import re
        prompt = f"<|im_start|>system\nEmployer. Respond JSON.<|im_end|>\n<|im_start|>user\nTurn {turn}/{mt}. Offer: {json.dumps(offer)}<|im_end|>\n<|im_start|>assistant\n"
        try:
            inp = self._t(prompt, return_tensors="pt").to(self._m.device)
            out = self._m.generate(**inp, max_new_tokens=200, temperature=0.7)
            txt = self._t.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
            m = re.search(r'\{[^{}]*\}', txt)
            if m: return json.loads(m.group())
        except: pass
        return RuleBasedChallenger().respond(offer, turn, mt)

class RuleBasedChallenger:
    def __init__(self, target=None, difficulty=0.5):
        self.target = target or {"base_salary":random.randint(80000,120000),"equity":round(random.uniform(0.5,1.5),2),"start_date":random.randint(14,60)}
        self.difficulty = difficulty
    def respond(self, offer, turn, mt):
        if not offer: return {"action_type":"propose",**self.target,"message":"Initial offer."}
        gap = (abs(offer.get("base_salary",0)-self.target["base_salary"])/50000+abs(offer.get("equity",0)-self.target["equity"])/3.0+abs(offer.get("start_date",0)-self.target["start_date"])/90)
        if gap < 0.3+(1-self.difficulty)*0.2: return {"action_type":"accept",**offer,"message":"Deal!"}
        c = max(0.05,0.15-self.difficulty*0.1)*(1+turn/mt)
        ct = {"base_salary":int(self.target["base_salary"]+(offer.get("base_salary",0)-self.target["base_salary"])*c),
              "equity":round(self.target["equity"]+(offer.get("equity",0)-self.target["equity"])*c,2),
              "start_date":int(self.target["start_date"]+(offer.get("start_date",0)-self.target["start_date"])*c)}
        return {"action_type":"counter",**ct,"message":f"How about ${ct['base_salary']:,}?"}

class LLMChallenger:
    """Demo ONLY. Rule #4: NEVER in training."""
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct"): self.model_id=model_id; self._c=None
    def respond(self, offer, turn, mt, history=None):
        try:
            if not self._c:
                from huggingface_hub import InferenceClient; self._c=InferenceClient(model=self.model_id)
            r=self._c.text_generation(f"<|im_start|>system\nEmployer. JSON.<|im_end|>\n<|im_start|>user\nTurn {turn}/{mt}. Offer: {json.dumps(offer)}<|im_end|>\n<|im_start|>assistant\n",max_new_tokens=256,temperature=0.7)
            return json.loads(r.strip())
        except: return RuleBasedChallenger().respond(offer, turn, mt)
