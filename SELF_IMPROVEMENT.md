# 🎯 Self-Improvement Features — Implementation Summary

## Statement 4: Self-Improvement Architecture

### 1. CurriculumManager (challenger.py)
**Purpose:** Agent's failures drive next training data

**Implementation:**
- Tracks reward per expert persona across all episodes
- After each epoch: inverts performance scores
- **Weak personas get MORE training weight**
- Maintains rolling window (last 50 episodes per expert)

**Key Methods:**
```python
def record(self, expert_name, reward):
    """Track reward for each expert encounter"""
    
def get_weights(self):
    """Returns curriculum weights - inverted performance scores"""
    # Low avg reward → High weight → More training
    
def sample_persona(self):
    """Weighted sampling based on agent's weaknesses"""
```

**Used In:**
- `train_colab.py` - Generation of training prompts (curriculum-weighted)
- `app_gradio.py` - Real-time curriculum visualization

---

### 2. SelfPlayChallenger (challenger.py)
**Purpose:** Epoch N model becomes Epoch N+1 opponent

**Implementation:**
- Loads checkpoint from previous epoch
- Acts as negotiation opponent using past model
- If loading fails, falls back to RuleBasedChallenger

**Key Methods:**
```python
def __init__(self, model_path):
    """Initialize with path to previous checkpoint"""
    
def respond(self, offer, turn, max_turns):
    """Generate response using past model's strategy"""
    # Uses Unsloth FastLanguageModel inference
    # Parses JSON actions from model output
```

**Integration Point:**
- Can be used in `train_colab.py` for self-play training iterations
- Each epoch's checkpoint becomes next epoch's opponent

---

### 3. ExpertChallenger with Escalation (challenger.py)
**Purpose:** Auto-increasing difficulty based on agent win rate

**Implementation:**
- 5 expert personas with distinct styles and hidden priorities
- Dynamic difficulty adjustment via `.escalate(win_rate)`
- Emotional dynamics (rapport/frustration) affect negotiation

**Key Features:**
```python
def escalate(self, win_rate):
    """Increase difficulty if agent is winning too often"""
    if win_rate > 0.6:
        self.difficulty = min(self.difficulty + 0.1, 1.0)
        self.concession_rate = max(0.02, self.concession_rate - 0.02)
    elif win_rate < 0.3:
        self.difficulty = max(self.difficulty - 0.05, 0.1)
```

**Expert Personas:**
1. Sarah Chen (Analytical) - Data-driven, balanced weights
2. Marcus Rivera (Aggressive) - Cash-focused, low equity
3. Dr. Aisha Patel (Collaborative) - Equity-heavy, mission-driven
4. James O'Brien (Bureaucratic) - Start date priority
5. Elena Volkov (Visionary) - Balanced with mission alignment

---

## Training Pipeline Integration

### train_colab.py - Multi-Epoch Curriculum Learning

**Architecture:**
```python
curriculum = CurriculumManager()  # Global curriculum tracker

for epoch in range(NUM_EPOCHS):
    # 1. Generate curriculum-weighted prompts
    prompts = gen_prompts(n=200, epoch=epoch)
    # Weak experts get MORE samples
    
    # 2. Train epoch with GRPO
    trainer.train()
    
    # 3. Save checkpoint for self-play
    trainer.save_model(f"{OUT}/epoch_{epoch+1}")
    
    # 4. Advance curriculum (rolling window)
    curriculum.advance()
```

**Curriculum-Weighted Sampling:**
```python
def gen_prompts(n=200, epoch=0):
    weights = curriculum.get_weights()
    expert_names = [p["name"] for p in EXPERT_PERSONAS]
    expert_weights = [weights.get(name, 1.0/5) for name in expert_names]
    
    for _ in range(n):
        # Sample expert based on curriculum (agent's weaknesses)
        expert_idx = random.choices(
            range(len(EXPERT_PERSONAS)), 
            weights=expert_weights, 
            k=1
        )[0]
        # Generate scenario with that expert...
```

---

## Snorkel AI Features (Simulated Experts-in-the-Loop)

### 1. 5 Expert Personas
- Distinct personalities, styles, deal-breakers
- Hidden priorities (agent must infer from cues)
- Emotional dynamics (rapport/frustration tracking)

### 2. Preference Drift
**Implementation:** `negotiation_environment.py`
```python
def reset(self):
    if self._ep > 0 and self._ep % self._shift == 0:
        self._pidx = (self._pidx + 1) % len(EXPERT_PERSONAS)
    # Shifts expert every 8 episodes
```

### 3. Information Asymmetry
- Agent sees: style hints ("analytical", "aggressive", etc.)
- Agent does NOT see: deal-breakers, hidden priorities, exact weights
- Must infer from: conversation patterns, concession behavior

### 4. Weighted Utility Calculation
**reward.py - Snorkel-weighted bonus:**
```python
def reward_deal_quality(completion, **kw):
    """Bonus if weighted utility matches expert's hidden priorities"""
    profile = PROFILES[profile_idx]
    weighted_utility = (
        profile["salary_wt"] * normalized_salary +
        profile["equity_wt"] * normalized_equity +
        profile["start_wt"] * normalized_start_date
    )
    return 0.3 if weighted_utility >= 0.5 else 0.0
```

---

## Gradio Demo - Curriculum Visualization

**app_gradio.py** displays real-time curriculum:
```python
def _curriculum_md(curr: CurriculumManager):
    """Show which experts need more training"""
    weights = curr.get_weights()
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    
    for name, weight in sorted_weights[:5]:
        avg_reward = sum(curr.perf[name]) / len(curr.perf[name])
        # Display: Higher weight = More training needed
```

**UI Components:**
- Real-time reward breakdown (format, negotiation, quality)
- Turn-by-turn reward curves (Plotly)
- Curriculum weights bar chart
- Expert style hints (visible) vs hidden priorities (invisible)

---

## Evaluation Pipeline

**evaluate.py** - Baseline vs Finetuned comparison:
- Runs 50+ episodes per model
- Tracks: deal rate, avg reward, avg turns, per-expert performance
- Baseline: Rule-based random strategy
- Finetuned: GRPO-trained model with curriculum learning

---

## Summary

✅ **Self-Improvement (Statement 4):**
1. CurriculumManager tracks weaknesses → drives next training
2. SelfPlayChallenger uses past checkpoints as opponents
3. ExpertChallenger.escalate() auto-adjusts difficulty

✅ **Snorkel AI ($10K bonus):**
1. 5 expert personas with hidden priorities
2. Preference drift every 8 episodes
3. Information asymmetry (style hints only)
4. Emotional dynamics affect concession patterns
5. Weighted utility calculations for quality rewards

✅ **Training Pipeline:**
- Multi-epoch curriculum learning
- Weighted expert sampling based on agent failures
- Checkpoint-based self-play capability
- Comprehensive evaluation metrics

✅ **Demo:**
- Real-time curriculum visualization
- Reward breakdown charts
- Turn-by-turn negotiation tracking
- Expert style inference gameplay
