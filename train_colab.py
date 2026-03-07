"""GRPO Training Script for SalaryNegotiationArena.

Uses the OFFICIAL OpenEnv + Unsloth + TRL pattern from hackathon slides (page 41):
  1. Load model via FastLanguageModel (4-bit)
  2. Apply LoRA adapters
  3. Define reward function using env client .sync()
  4. Train with GRPOTrainer

Run on Northflank H100 or Colab A100.
"""

import json
import random

# ---- Unsloth ----
from unsloth import FastLanguageModel

# ---- TRL ----
from trl import GRPOTrainer, GRPOConfig

# ---- OpenEnv client (official pattern from slides page 41) ----
from client.negotiation_env import NegotiationEnv, NegotiationAction

# =====================================================================
# CONFIG
# =====================================================================
MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
MAX_SEQ_LEN = 2048
OUTPUT_DIR = "./grpo_output"
HF_REPO = "yashj2110/salary-negotiation-qwen-1.5b"

# Environment URL — your HF Space or local server
ENV_URL = "https://yashj2110-negotiation-arena.hf.space"
# For local dev: ENV_URL = "http://localhost:8000"


# =====================================================================
# REWARD FUNCTION (official pattern from slides pages 40-41)
# =====================================================================
def openenv_reward(completions, **kwargs):
    """
    OpenEnv reward function for GRPOTrainer.

    For each completion, connect to the env via .sync() client,
    reset, step with the completion, and collect the reward.
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        try:
            # Parse agent's action from completion
            action = _parse_action(text)

            # Connect to env and evaluate
            with NegotiationEnv(base_url=ENV_URL).sync() as env:
                env.reset()
                result = env.step(action)
                rewards.append(result.reward)
        except Exception:
            rewards.append(-0.5)  # penalty for unparseable output
    return rewards


def _parse_action(text: str) -> NegotiationAction:
    """Parse a JSON action from model output."""
    import re
    match = re.search(r'\{[^{}]*\}', text)
    if match:
        try:
            data = json.loads(match.group())
            return NegotiationAction(**data)
        except (json.JSONDecodeError, Exception):
            pass
    # Default: propose something reasonable
    return NegotiationAction(
        action_type="propose",
        base_salary=random.randint(120_000, 160_000),
        equity=round(random.uniform(1.5, 3.5), 2),
        start_date=random.randint(30, 90),
        message=text[:200],
    )


# =====================================================================
# DATASET
# =====================================================================
def generate_prompts(num: int = 200) -> list[dict]:
    """Generate negotiation scenario prompts."""
    system_msg = (
        "You are a skilled salary negotiator (the candidate). "
        "Respond ONLY with a JSON: {action_type, base_salary, equity, start_date, message}. "
        "Actions: propose, counter, accept, reject, walk_away. "
        "Aim for the best deal while closing early."
    )

    prompts = []
    for _ in range(num):
        salary = random.randint(80_000, 120_000)
        equity = round(random.uniform(0.5, 1.5), 2)
        start = random.randint(14, 60)
        turn = random.randint(0, 7)
        max_turns = 10

        scenario = (
            f"Turn {turn}/{max_turns}. "
            f"Employer offers: ${salary:,}/yr, {equity}% equity, {start} days start. "
            f"Negotiate the best package. Respond with JSON."
        )

        prompts.append({
            "prompt": (
                f"<|im_start|>system\n{system_msg}<|im_end|>\n"
                f"<|im_start|>user\n{scenario}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            ),
        })

    random.shuffle(prompts)
    print(f"Generated {len(prompts)} training prompts")
    return prompts


# =====================================================================
# MAIN
# =====================================================================
def train():
    print("=" * 60)
    print("SalaryNegotiationArena — GRPO Training (Official Pattern)")
    print("=" * 60)

    # 1. Load model
    print("\n[1/4] Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )

    # 2. LoRA
    print("[2/4] Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0, bias="none",
    )

    # 3. Dataset
    print("[3/4] Generating prompts...")
    dataset = generate_prompts(200)
    from datasets import Dataset
    train_dataset = Dataset.from_dict({"prompt": [d["prompt"] for d in dataset]})

    # 4. Train with GRPO
    print("[4/4] Training...")
    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        num_generations=4,
        max_completion_length=512,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        config=config,
        train_dataset=train_dataset,
        reward_funcs=openenv_reward,  # single reward fn using env
    )

    trainer.train()

    print("\nSaving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Done! Model at {OUTPUT_DIR}")
    return model, tokenizer


def push_to_hub():
    from huggingface_hub import HfApi
    HfApi().upload_folder(folder_path=OUTPUT_DIR, repo_id=HF_REPO, repo_type="model")
    print(f"Pushed to https://huggingface.co/{HF_REPO}")


if __name__ == "__main__":
    model, tokenizer = train()
    # push_to_hub()
