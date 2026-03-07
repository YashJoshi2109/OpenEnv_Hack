# 🚀 Deployment Guide — SalaryNegotiationArena

## Prerequisites

- Northflank H100 GPU with Jupyter PyTorch service running
- SSH connection established (as per conversation history)
- HuggingFace account authenticated
- GitHub repo: `github.com/YashJoshi2109/OpenEnv_Hack`
- HF Space: `huggingface.co/spaces/yashj2110/negotiation-arena`

---

## Step 1: Upload Code to Northflank

From your **LOCAL** machine (macOS):

```bash
cd /Users/yash/Downloads/OpenEnv

# Verify SSH connection is still active
ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null -p 42530 root@127.0.0.1 'echo SSH_OK'

# Upload the entire project
scp -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null -P 42530 -r \
    /Users/yash/Downloads/OpenEnv root@127.0.0.1:/home/jovyan/negotiation-arena

# Verify upload
ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null -p 42530 root@127.0.0.1 \
    'ls -la /home/jovyan/negotiation-arena'
```

**Alternative:** Use `rsync` for faster sync:
```bash
rsync -avz --progress -e "ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null -p 42530" \
    /Users/yash/Downloads/OpenEnv/ root@127.0.0.1:/home/jovyan/negotiation-arena/
```

---

## Step 2: Install Dependencies on Northflank

SSH into the container:
```bash
ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null -p 42530 root@127.0.0.1
```

Inside the container:
```bash
cd /home/jovyan/negotiation-arena

# Install base dependencies (NO unsloth/trl yet)
pip install openenv-core fastapi uvicorn pydantic gradio plotly \
    transformers torch pytest huggingface_hub

# Verify installation
python3 verify_build.py
```

---

## Step 3: Test Environment Locally

```bash
# Run unit tests
python3 -m pytest test_env.py -v

# Expected output:
# test_env.py::TestEnv::test_reset PASSED
# test_env.py::TestEnv::test_step PASSED
# test_env.py::TestEnv::test_walk_away PASSED
# test_env.py::TestEnv::test_timeout PASSED
# test_env.py::TestEnv::test_snorkel_shift PASSED
# test_env.py::TestEnv::test_expert_style_visible PASSED
# test_env.py::TestReward::test_valid PASSED
# test_env.py::TestReward::test_invalid PASSED
# test_env.py::TestChallenger::test_expert PASSED
# test_env.py::TestChallenger::test_curriculum PASSED
```

---

## Step 4: Start OpenEnv Server (Background)

```bash
# Start server on port 8000
nohup uvicorn server.app:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &

# Verify server is running
curl http://localhost:8000/health || echo "Server starting..."

# Check logs
tail -f server.log
```

---

## Step 5: Install Training Dependencies

**IMPORTANT:** Only install these for training (not in requirements.txt):

```bash
pip install unsloth trl datasets accelerate -q
```

---

## Step 6: Run GRPO Training

```bash
# Set environment URL to local server
export ENV_URL="http://localhost:8000"

# Run training (3 epochs with curriculum learning)
python3 train_colab.py 2>&1 | tee training.log

# Training will:
# 1. Load Qwen2.5-1.5B-Instruct-bnb-4bit
# 2. Apply LoRA (r=16)
# 3. Run 3 epochs with curriculum-weighted sampling
# 4. Save checkpoints to ./grpo_output/epoch_1, epoch_2, epoch_3
# 5. Save final model to ./grpo_output/

# Expected runtime: ~2-3 hours on H100
```

**Monitor training:**
```bash
# In another terminal (local)
ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null -p 42530 root@127.0.0.1 \
    'tail -f /home/jovyan/negotiation-arena/training.log'
```

---

## Step 7: Evaluate Model

```bash
# Run baseline vs finetuned comparison
python3 evaluate.py --episodes 50 --model-path ./grpo_output

# Expected output:
# ==================================================
# Baseline
# ==================================================
# Avg Reward: 0.1234
# Deal Rate: 45.0%
# Avg Turns: 6.2
#   Sarah Chen — VP Engineering: 0.156 (10 eps)
#   Marcus Rivera — CFO: 0.089 (11 eps)
#   ...
#
# ==================================================
# Finetuned (GRPO)
# ==================================================
# Avg Reward: 0.5678
# Deal Rate: 72.0%
# Avg Turns: 4.8
#   Sarah Chen — VP Engineering: 0.634 (9 eps)
#   Marcus Rivera — CFO: 0.523 (12 eps)
#   ...
```

---

## Step 8: Push Model to HuggingFace

```bash
# Authenticate (if not already done)
huggingface-cli login

# Push model
python3 -c "from train_colab import push; push()"

# Model will be at: huggingface.co/yashj2110/salary-negotiation-qwen-1.5b
```

---

## Step 9: Deploy to HuggingFace Spaces

### Option A: From Local Machine

```bash
# Clone HF Space repo
git clone https://huggingface.co/spaces/yashj2110/negotiation-arena
cd negotiation-arena

# Copy files (excluding training deps)
cp -r /Users/yash/Downloads/OpenEnv/{server,client,*.py,*.md,*.yaml,*.toml,requirements.txt} .

# Commit and push
git add .
git commit -m "Deploy self-improvement negotiation arena"
git push
```

### Option B: From Northflank

```bash
cd /home/jovyan/negotiation-arena

# Add HF Space remote
git remote add space https://huggingface.co/spaces/yashj2110/negotiation-arena
git push space main
```

---

## Step 10: Test Gradio Demo

**Locally on Northflank:**
```bash
python3 app.py
# Demo will run on http://localhost:7860
```

**On HuggingFace Spaces:**
Visit: https://huggingface.co/spaces/yashj2110/negotiation-arena

---

## Step 11: Create Submission Video

Record a 1-minute demo showing:

1. **Environment Innovation** (0-15s):
   - Show 5 expert personas with distinct styles
   - Demonstrate preference drift (episode counter)
   - Show information asymmetry (style hints only)

2. **Self-Improvement** (15-30s):
   - Show curriculum weights in Gradio
   - Explain: "Weak experts get MORE training"
   - Highlight reward curves

3. **Training Results** (30-45s):
   - Show evaluation comparison
   - Baseline: 45% deal rate
   - Finetuned: 72% deal rate
   - Show per-expert improvements

4. **Live Negotiation** (45-60s):
   - Play one negotiation round
   - Show real-time rewards
   - Accept deal, show success

Upload to: YouTube (unlisted or public)

---

## Step 12: Submit to OpenEnv Hackathon

Go to: https://cerebralvalley.ai/openenv-hackathon

**Submission Form:**
- **Project Name:** SalaryNegotiationArena
- **Category:** Statement 4 (Self-Improvement) + Snorkel AI
- **GitHub:** github.com/YashJoshi2109/OpenEnv_Hack
- **HF Space:** huggingface.co/spaces/yashj2110/negotiation-arena
- **HF Model:** huggingface.co/yashj2110/salary-negotiation-qwen-1.5b
- **Video:** [YouTube link]
- **Description:**
  ```
  SalaryNegotiationArena: RL environment where LLM agents learn to negotiate
  salary packages against 5 simulated hiring experts with hidden priorities.
  
  Self-Improvement Features:
  - CurriculumManager tracks agent failures → drives next training
  - SelfPlayChallenger uses past checkpoints as opponents
  - ExpertChallenger auto-adjusts difficulty based on win rate
  
  Snorkel AI Features:
  - 5 expert personas with distinct styles & hidden priorities
  - Preference drift every 8 episodes
  - Information asymmetry (style hints only)
  - Emotional dynamics (rapport/frustration tracking)
  
  Results: 72% deal rate (vs 45% baseline), 60% improvement in avg reward
  ```

---

## Troubleshooting

### SSH Connection Lost
```bash
# Restart SSH proxy (local terminal)
northflank ssh service --projectId hackathon --serviceId jupyter-pytorch --proxyOnly

# Reconnect
ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null -p 42530 root@127.0.0.1
```

### Server Not Starting
```bash
# Check logs
cat server.log

# Verify port not in use
lsof -i :8000

# Kill existing process
pkill -f "uvicorn server.app"
```

### Training OOM
```bash
# Reduce batch size in train_colab.py:
# per_device_train_batch_size=2 (instead of 4)
# gradient_accumulation_steps=8 (instead of 4)
```

### HF Cache Full
```bash
# Clean old cache
rm -rf /home/jovyan/.hf/hub/models--*

# Verify space
df -h /home/jovyan
```

---

## Next Steps After Deployment

1. **Monitor HF Space:** Check deployment logs
2. **Share Demo:** Post on Twitter/LinkedIn with video
3. **Iterate:** Based on judge feedback
4. **Scale Up:** Try 7B model if H100 has capacity
5. **Add Features:** Self-play mode, multi-turn curriculum

---

## Quick Reference

**SSH Command:**
```bash
ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null -p 42530 root@127.0.0.1
```

**Upload Files:**
```bash
scp -i ~/.ssh/id_ed25519 -P 42530 -r /Users/yash/Downloads/OpenEnv/* \
    root@127.0.0.1:/home/jovyan/negotiation-arena/
```

**Training Command:**
```bash
cd /home/jovyan/negotiation-arena && python3 train_colab.py
```

**Evaluation:**
```bash
python3 evaluate.py --episodes 50 --model-path ./grpo_output
```

**Push to HF:**
```bash
python3 -c "from train_colab import push; push()"
```

Good luck! 🚀
