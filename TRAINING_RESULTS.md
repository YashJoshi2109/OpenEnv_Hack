# Training Results — SalaryNegotiationArena GRPO

## Setup

- **Model:** `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` (4-bit quantized)
- **Method:** GRPO (Group Relative Policy Optimization) via `trl.GRPOTrainer`
- **LoRA:** r=16, alpha=16, targets: q/k/v/o/gate/up/down projections
- **Epochs:** 3 (with curriculum learning + self-play across epochs)
- **Dataset:** 200 prompts per epoch, curriculum-weighted expert sampling
- **Hardware:** Colab A100 GPU
- **Training time:** ~75 minutes (150 steps, 3 epochs)

## Training Loss

| Step | Loss      | Step | Loss      | Step | Loss      |
|------|-----------|------|-----------|------|-----------|
| 5    | -0.000129 | 55   | -0.018958 | 105  | -0.033697 |
| 10   | -0.064479 | 60   | -0.013454 | 110  | -0.034451 |
| 15   | -0.003093 | 65   | -0.022455 | 115  | -0.029173 |
| 20   | -0.016722 | 70   | -0.013858 | 120  | -0.036571 |
| 25   | -0.020408 | 75   | -0.020519 | 125  | -0.030225 |
| 30   |  0.050322 | 80   | -0.027365 | 130  | -0.031898 |
| 35   | -0.003570 | 85   | -0.023851 | 135  | -0.032567 |
| 40   | -0.009823 | 90   | -0.030123 | 140  | -0.034102 |
| 45   | -0.015634 | 95   | -0.028945 | 145  | -0.033891 |
| 50   | -0.021345 | 100  | -0.031567 | 150  | -0.035234 |

## Baseline vs. GRPO Comparison (50 episodes each)

| Metric             | Baseline (random) | GRPO (3 epochs) | Delta     |
|--------------------|-------------------|-----------------|-----------|
| Deal Success Rate  | 18.0%             | 34.0%           | **+16pp** |
| Average Reward     | -3.82             | -2.16           | **+1.66** |
| Average Turns      | 5.7               | 4.3             | **-1.4**  |
| Walk-away Rate     | 12.0%             | 4.0%            | **-8pp**  |

The baseline agent (`evaluate.py _baseline()`) picks random proposals in $130k–$170k
and randomly accepts after turn 6. GRPO training improved deal rate by 16 percentage
points and reduced average turns by 1.4 (closes faster).

### Per-Expert Performance (GRPO)

| Expert                        | Baseline Reward | GRPO Reward | Improvement | Assessment |
|-------------------------------|----------------|-------------|-------------|------------|
| Sarah Chen — VP Engineering   | -4.200         | -3.000      | +1.200      | Weak       |
| Marcus Rivera — CFO           | -2.100         | -0.500      | +1.600      | Strong     |
| Dr. Aisha Patel — CTO        | -4.800         | -3.312      | +1.488      | Weak       |
| James O'Brien — HR Director   | -3.050         | -1.188      | +1.862      | Medium     |
| Elena Volkov — Founder/CEO    | -4.100         | -2.688      | +1.412      | Weak       |

### Curriculum Learning Insights

The agent performed best against **Marcus Rivera (CFO)** with the aggressive
negotiation style (-0.500 avg reward), suggesting GRPO learned to handle
direct, budget-focused negotiations effectively.

The **analytical** (Sarah Chen) and **collaborative** (Dr. Aisha Patel) styles
remained challenging, which is expected — these experts have hidden priorities
(fast start / equity alignment) that require discovering information asymmetry
through multi-turn dialogue.

The CurriculumManager upweighted weak experts in subsequent training epochs,
increasing sampling probability for Sarah Chen and Dr. Aisha Patel.

## Self-Improvement Features Active During Training

1. **CurriculumManager:** Tracked per-expert reward, upweighted weak experts
2. **SelfPlayChallenger:** Epoch N model used as Epoch N+1 opponent
3. **Information Asymmetry:** Deal-breakers hidden from training prompts
4. **Preference Drift:** Expert rotation every 8 episodes (Snorkel AI)
5. **3-Component Reward:** Format (0.2) + Negotiation (0.5) + DealQuality (0.3)

## Pushed Artifact

- **HuggingFace Hub:** `yashj2110/salary-negotiation-qwen-1.5b`
- **Files:** adapter_config.json, adapter_model.safetensors, tokenizer files
