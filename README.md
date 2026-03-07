---
title: SalaryNegotiationArena
emoji: 🤝
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.20.0"
app_file: app.py
pinned: false
license: mit
---

# 🤝 SalaryNegotiationArena

**OpenEnv Hackathon SF — Self-Improvement (Statement 4) + Snorkel AI**

An RL environment where LLM agents learn to negotiate salary packages against 5 simulated hiring experts with hidden priorities that shift over time.

## Key Innovation

- **5 Expert Personas** with distinct personalities, deal-breakers, and hidden priorities
- **Information Asymmetry** — agent must infer expert priorities from conversational cues
- **CurriculumManager** — agent's weaknesses drive next epoch's training data
- **SelfPlayChallenger** — epoch N model becomes epoch N+1 opponent
- **Snorkel Drift** — preferences shift every 8 episodes

## Architecture

```
Agent (Qwen2.5-1.5B) ←→ OpenEnv MCPEnvironment ←→ Expert Challengers
                              ↓
                     3 Reward Functions
                     ├── Format compliance
                     ├── Negotiation outcome
                     └── Snorkel-weighted quality
```
## File Structure
OpenEnv_Hack/
├── server/
│   ├── __init__.py
│   ├── models.py
│   ├── negotiation_environment.py
│   └── app.py
├── client/
│   ├── __init__.py
│   └── negotiation_env.py
├── reward.py
├── challenger.py
├── app_gradio.py
├── app.py
├── train_colab.py
├── evaluate.py
├── test_env.py
├── requirements.txt
├── openenv.yaml
├── pyproject.toml
├── README.md
├── .gitignore
└── negotiation_arena_training.ipynb

## Resources Negotiated

| Resource | Range |
|----------|-------|
| Base Salary | $80K–$200K/yr |
| Equity | 0%–5% RSU |
| Start Date | 14–180 days |

## Training

Unsloth + TRL GRPO on Qwen2.5-1.5B (4-bit) using Northflank H100.
https://unsloth.ai/docs/get-startedunsloth-notebooks#grpo-reasoning-rl-notebooks

## Built by

**Yash Joshi** — Solo builder at OpenEnv Hackathon SF 2025
