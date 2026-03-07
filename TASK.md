# 🤝 SalaryNegotiationArena — COMPLETE BUILD TASK

## PROJECT OVERVIEW

You are building **SalaryNegotiationArena** for the OpenEnv Hackathon SF.

**What:** An OpenEnv RL environment where an LLM agent learns to negotiate
salary packages (base_salary, equity, start_date) against simulated hiring
experts whose preferences shift over time.

**Prize Targets:**
- Primary: Statement 4 — Self-Improvement ($15K-$6K main prizes)
- Bonus: Snorkel AI — Simulated Experts-in-the-Loop ($10K)

**Judging Criteria:**
- Environment Innovation (40%) — novel, creative, challenging
- Storytelling (30%) — clear demo, engaging explanation
- Training Script Showing Improvement in Rewards (20%) — reward curves
- Reward and Training Pipeline Setup (10%) — coherent pipeline

---

## CRITICAL RULES — NEVER VIOLATE

1. Install OpenEnv as: `pip install openenv-core` (NEVER `pip install openenv==0.2.1`)
2. Models subclass `from openenv import Action, Observation, State`
3. Environment subclasses `from openenv import MCPEnvironment`
4. Environment uses `from mcp.server.fastmcp import FastMCP` with `@mcp.tool` decorators
5. server/app.py uses `from openenv import create_app` passing the CLASS (not instance)
6. Client subclasses `from openenv import EnvClient, StepResult`
7. Training reward function uses client `.sync()` pattern
8. Reward logic NEVER inside the environment file — only in reward.py
9. NEVER use LLMChallenger during training (only in Gradio demo)
10. NEVER add unsloth/trl/datasets/accelerate to requirements.txt (Colab/Northflank only)
11. Pydantic models ONLY in server/models.py
12. app.py is thin entry → imports build_app from app_gradio.py

---

## ARCHITECTURE

```
negotiation-arena/
├── server/
│   ├── __init__.py              ← exports
│   ├── models.py                ← Action, Observation, State subclasses
│   ├── negotiation_environment.py ← MCPEnvironment + FastMCP tools
│   └── app.py                   ← create_app(CLASS, ...) entry point
├── client/
│   ├── __init__.py              ← exports
│   └── negotiation_env.py       ← EnvClient subclass
├── reward.py                    ← STANDALONE 3 reward functions
├── challenger.py                ← ExpertChallenger + CurriculumManager + SelfPlayChallenger
├── evaluate.py                  ← baseline vs finetuned comparison
├── app_gradio.py                ← Gradio demo (ONLY place LLMChallenger used)
├── app.py                       ← thin entry: from app_gradio import build_app
├── train_colab.py               ← Unsloth + TRL GRPO training
├── negotiation_arena_training.ipynb ← 5-cell Colab/Northflank notebook
├── test_env.py                  ← 8+ unit tests
├── requirements.txt             ← NO unsloth/trl/accelerate
├── openenv.yaml                 ← OpenEnv manifest
├── pyproject.toml               ← pip-installable package
├── README.md                    ← HF Spaces frontmatter
├── .gitignore
└── TASK.md                      ← this file
```

---

## SELF-IMPROVEMENT FEATURES (Statement 4)

1. **CurriculumManager** — tracks agent reward per expert persona.
   After each epoch, inverts scores: weak personas get MORE training.
   Agent's failures generate its next training data.

2. **SelfPlayChallenger** — checkpoint from epoch N becomes
   challenger in epoch N+1. Agent negotiates against past self.

3. **ExpertChallenger.escalate()** — difficulty auto-increases
   when agent win rate exceeds threshold.

## SNORKEL AI FEATURES ($10K bonus)

1. **5 Expert Personas** with distinct personalities, styles,
   deal-breakers, hidden priorities, and concession patterns.

2. **Preference Drift** — profiles shift every 8 episodes.

3. **Information Asymmetry** — agent can't see expert's hidden
   priorities. Must infer from conversational cues.

4. **Emotional Dynamics** — rapport/frustration tracking affects
   expert willingness to concede.

---

## DEPLOYMENT

- **HF Spaces:** sdk=gradio, app_file=app.py
- **GitHub:** github.com/YashJoshi2109/negotiation-arena
- **HF Space:** huggingface.co/spaces/yashj2110/negotiation-arena
- **Model:** yashj2110/salary-negotiation-qwen-1.5b
- **Training:** Northflank H100 or Colab A100

## GIT REMOTES

```bash
git remote add origin https://github.com/YashJoshi2109/OpenEnv_Hack.git
git remote add space https://huggingface.co/spaces/yashj2110/negotiation-arena
git push origin main   # GitHub
git push space main    # HF Spaces
```

---

## CHECKLIST

### Phase 1: Environment (MOST IMPORTANT — 40% of judging)
- [ ] server/models.py — Action, Observation, State subclasses
- [ ] server/negotiation_environment.py — MCPEnvironment + 5 MCP tools
- [ ] server/app.py — create_app(CLASS)
- [ ] openenv.yaml + pyproject.toml
- [ ] test_env.py passes all tests

### Phase 2: Challenger + Reward (Self-Improvement + Snorkel)
- [ ] challenger.py — ExpertChallenger (5 personas), CurriculumManager, SelfPlayChallenger
- [ ] reward.py — 3 standalone reward functions

### Phase 3: Client + Training
- [ ] client/negotiation_env.py — EnvClient subclass
- [ ] train_colab.py — Unsloth + GRPO with env client .sync()
- [ ] negotiation_arena_training.ipynb — 5 cells

### Phase 4: Demo + Deploy
- [ ] app_gradio.py — full Gradio demo with charts
- [ ] app.py — thin entry
- [ ] README.md with HF frontmatter
- [ ] requirements.txt (NO training deps)
- [ ] git push to both remotes
- [ ] HF Space running

### Phase 5: Training + Results
- [ ] Run training on Northflank H100
- [ ] Collect baseline vs trained metrics
- [ ] Record 1-min YouTube demo video
- [ ] Submit at cerebralvalley.ai
