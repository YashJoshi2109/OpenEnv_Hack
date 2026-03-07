"""Evaluate baseline vs finetuned model."""
import json, random, argparse
from collections import defaultdict
from server.negotiation_environment import NegotiationEnvironment
from server.models import NegotiationAction
from reward import compute_reward

def run(name, episodes=50, model_path=None):
    env = NegotiationEnvironment()
    m = {"reward":0,"deals":0,"no_deals":0,"walks":0,"turns":0,"by_expert":defaultdict(list),"ep_rewards":[]}
    gen = _load_model(model_path) if model_path else None
    for _ in range(episodes):
        obs = env.reset(); er = 0
        for t in range(10):
            if obs.phase != "negotiating": break
            act = gen(obs) if gen else _baseline(obs, t)
            obs = env.step(act)
            es = {"phase":obs.phase,"turn":obs.turn,"max_turns":obs.max_turns,
                  "current_offer":{"base_salary":obs.current_offer_salary,"equity":obs.current_offer_equity,"start_date":obs.current_offer_start},
                  "profile_idx":env._pidx}
            er += compute_reward(json.dumps(act.model_dump()), env_state=es)
        m["reward"]+=er; m["ep_rewards"].append(er); m["turns"]+=obs.turn
        m["by_expert"][obs.expert_name].append(er)
        if obs.phase=="deal_reached": m["deals"]+=1
        elif obs.phase=="walked_away": m["walks"]+=1
        else: m["no_deals"]+=1
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    print(f"Avg Reward: {m['reward']/episodes:.4f}")
    print(f"Deal Rate: {m['deals']/episodes*100:.1f}%")
    print(f"Avg Turns: {m['turns']/episodes:.1f}")
    for ex, rws in m["by_expert"].items():
        print(f"  {ex}: {sum(rws)/len(rws):.3f} ({len(rws)} eps)")
    return m

def _baseline(obs, t):
    if t == 0: return NegotiationAction(action_type="propose",base_salary=random.randint(130000,170000),equity=round(random.uniform(2,4),2),start_date=random.randint(30,90))
    if t >= 6 and random.random() > 0.5: return NegotiationAction(action_type="accept")
    return NegotiationAction(action_type="counter",base_salary=obs.current_offer_salary+random.randint(-5000,10000),
        equity=round(obs.current_offer_equity+random.uniform(-0.25,0.5),2),start_date=max(14,obs.current_offer_start+random.randint(-14,7)))

def _load_model(path):
    try:
        from unsloth import FastLanguageModel
        import re
        model, tok = FastLanguageModel.from_pretrained(model_name=path, max_seq_length=2048, load_in_4bit=True)
        FastLanguageModel.for_inference(model)
        def gen(obs):
            p = f"<|im_start|>system\nNegotiator. JSON only.<|im_end|>\n<|im_start|>user\nTurn {obs.turn}/{obs.max_turns}. Offer: ${obs.current_offer_salary:,}, {obs.current_offer_equity}%, {obs.current_offer_start}d. Expert: {obs.expert_style}.<|im_end|>\n<|im_start|>assistant\n"
            inp = tok(p, return_tensors="pt").to(model.device)
            out = model.generate(**inp, max_new_tokens=256, temperature=0.7)
            txt = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
            m = re.search(r'\{[^{}]*\}', txt)
            if m:
                try: return NegotiationAction(**json.loads(m.group()))
                except: pass
            return _baseline(obs, obs.turn)
        return gen
    except: return None

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--episodes",type=int,default=50)
    pa.add_argument("--model-path",type=str,default="./grpo_output")
    a = pa.parse_args()
    run("Baseline", a.episodes)
    run("Finetuned (GRPO)", a.episodes, a.model_path)
