"""GRPO Training — Official OpenEnv + Unsloth pattern from slides page 41.
Uses env client .sync() for reward evaluation.
SELF-IMPROVEMENT: CurriculumManager + SelfPlayChallenger."""
import json, random, re, os
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from client.negotiation_env import NegotiationEnv
from server.models import NegotiationAction
from challenger import ExpertChallenger, CurriculumManager, SelfPlayChallenger, EXPERT_PERSONAS
from reward import compute_reward

MODEL = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
SEQ_LEN = 2048
OUT = "./grpo_output"
HF_REPO = "yashj2110/salary-negotiation-qwen-1.5b"
ENV_URL = "https://yashj2110-negotiation-arena.hf.space"  # or http://localhost:8000

# Global curriculum manager for self-improvement
curriculum = CurriculumManager()


def openenv_reward(completions, **kw):
    """Reward via env client .sync() + curriculum tracking."""
    rewards = []
    for c in completions:
        txt = c[0]["content"] if isinstance(c, list) else str(c)
        try:
            action = _parse(txt)
            with NegotiationEnv(base_url=ENV_URL).sync() as env:
                obs = env.reset()
                expert_name = obs.expert_name if hasattr(obs, 'expert_name') else "Unknown"
                r = env.step(action)
                # Track curriculum
                curriculum.record(expert_name, r.reward)
                rewards.append(r.reward)
        except Exception as e:
            print(f"Reward error: {e}")
            rewards.append(-0.5)
    return rewards



def _parse(text):
    m = re.search(r'\{[^{}]*\}', text)

def gen_prompts(n=200, epoch=0):
    """Generate prompts with curriculum-weighted expert sampling."""
    sys = ("You are a skilled salary negotiator (candidate). Respond ONLY with JSON: "
           "{action_type, base_salary, equity, start_date, message}. "
           "Actions: propose, counter, accept, reject, walk_away. Best deal, close early.")
    prompts = []
    
    # Get curriculum weights (weak experts get MORE training)
    weights = curriculum.get_weights()
    expert_names = [p["name"] for p in EXPERT_PERSONAS]
    expert_weights = [weights.get(name, 1.0/len(EXPERT_PERSONAS)) for name in expert_names]
    
    for _ in range(n):
        # Sample expert based on curriculum (agent's weaknesses)
        expert_idx = random.choices(range(len(EXPERT_PERSONAS)), weights=expert_weights, k=1)[0]
        expert = EXPERT_PERSONAS[expert_idx]

def train():
    print("="*60+"\nSalaryNegotiationArena — GRPO Training with Self-Improvement\n"+"="*60)
    print("[1/5] Loading model...")
    model, tok = FastLanguageModel.from_pretrained(model_name=MODEL, max_seq_length=SEQ_LEN, load_in_4bit=True)
    
    print("[2/5] LoRA...")
    model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], 
        lora_dropout=0, bias="none")
    
    NUM_EPOCHS = 3
    prev_checkpoint = None
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}\nEPOCH {epoch+1}/{NUM_EPOCHS}\n{'='*60}")
        
        print(f"[3/5] Dataset generation (curriculum-weighted)...")
        ds = gen_prompts(n=200, epoch=epoch)
        from datasets import Dataset
        train_ds = Dataset.from_dict({"prompt": [d["prompt"] for d in ds]})
        
        print(f"[4/5] Training epoch {epoch+1}...")
        epoch_out = f"{OUT}/epoch_{epoch+1}"
        os.makedirs(epoch_out, exist_ok=True)
        
        cfg = GRPOConfig(
            output_dir=epoch_out, 
            num_train_epochs=1,  # 1 epoch per curriculum iteration
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4, 
            learning_rate=5e-6, 
            lr_scheduler_type="cosine",
            num_generations=4, 
            max_completion_length=512, 
            logging_steps=10, 
            save_steps=100, 
            bf16=True, 
            report_to="none"
        )
        
        trainer = GRPOTrainer(
            model=model, 
            processing_class=tok, 
            config=cfg, 
            train_dataset=train_ds, 
            reward_funcs=openenv_reward
        )
        trainer.train()
        
        # Save checkpoint for this epoch
        trainer.save_model(epoch_out)
        tok.save_pretrained(epoch_out)
        print(f"Epoch {epoch+1} checkpoint saved to {epoch_out}")
        
        # Advance curriculum (keeps last 50 per expert)
        curriculum.advance()
        print(f"Curriculum advanced. Current weights:")
        for name, weight in list(curriculum.get_weights().items())[:5]:
            print(f"  {name}: {weight:.3f}")
        
        prev_checkpoint = epoch_out
    
    print(f"\n[5/5] Final model save...")
    trainer.save_model(OUT)
    tok.save_pretrained(OUT)
    print(f"="*60)
    print(f"Training complete! Final model: {OUT}")
    print(f"Self-Improvement features used:")
    print(f"  ✓ CurriculumManager: {len(curriculum.perf)} experts tracked")
    print(f"  ✓ Curriculum iterations: {NUM_EPOCHS}")
    print(f"="*60)


def train():
    print("="*60+"\nSalaryNegotiationArena — GRPO Training\n"+"="*60)
    print("[1/4] Loading model...")
    model, tok = FastLanguageModel.from_pretrained(model_name=MODEL, max_seq_length=SEQ_LEN, load_in_4bit=True)
    print("[2/4] LoRA...")
    model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], lora_dropout=0, bias="none")
    print("[3/4] Dataset...")
    ds = gen_prompts(200)
    from datasets import Dataset
    train_ds = Dataset.from_dict({"prompt": [d["prompt"] for d in ds]})
    print("[4/4] Training...")
    cfg = GRPOConfig(output_dir=OUT, num_train_epochs=3, per_device_train_batch_size=4,
        gradient_accumulation_steps=4, learning_rate=5e-6, lr_scheduler_type="cosine",
        num_generations=4, max_completion_length=512, logging_steps=10, save_steps=100, bf16=True, report_to="none")
    trainer = GRPOTrainer(model=model, processing_class=tok, config=cfg, train_dataset=train_ds, reward_funcs=openenv_reward)
    trainer.train()
    trainer.save_model(OUT); tok.save_pretrained(OUT)
    print(f"Done! Model at {OUT}")

def push():
    from huggingface_hub import HfApi
    HfApi().upload_folder(folder_path=OUT, repo_id=HF_REPO, repo_type="model")
    print(f"Pushed to hf.co/{HF_REPO}")

if __name__ == "__main__": train()
