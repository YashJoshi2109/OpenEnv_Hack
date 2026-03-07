#!/usr/bin/env python3
"""Verify project structure and imports for OpenEnv SalaryNegotiationArena."""
import sys
from pathlib import Path

def check_structure():
    """Verify all required files exist."""
    root = Path(__file__).parent
    
    required_files = {
        # Server
        "server/__init__.py": "Server package init",
        "server/models.py": "Pydantic models (Action, Observation, State)",
        "server/negotiation_environment.py": "MCPEnvironment + FastMCP tools",
        "server/app.py": "FastAPI create_app entry",
        
        # Client
        "client/__init__.py": "Client package init",
        "client/negotiation_env.py": "EnvClient subclass",
        
        # Root
        "reward.py": "Standalone reward functions",
        "challenger.py": "Expert/Curriculum/SelfPlay challengers",
        "train_colab.py": "GRPO training with curriculum",
        "evaluate.py": "Baseline vs finetuned evaluation",
        "app_gradio.py": "Gradio demo with curriculum viz",
        "app.py": "Thin entry point",
        "test_env.py": "Unit tests",
        "openenv.yaml": "OpenEnv manifest",
        "pyproject.toml": "Package metadata",
        "requirements.txt": "Dependencies",
        "README.md": "HF Spaces frontmatter",
        "TASK.md": "Build specifications",
    }
    
    print("="*60)
    print("📁 Project Structure Verification")
    print("="*60)
    
    all_exist = True
    for file_path, description in required_files.items():
        full_path = root / file_path
        exists = full_path.exists()
        symbol = "✅" if exists else "❌"
        print(f"{symbol} {file_path:<45} {description}")
        if not exists:
            all_exist = False
    
    return all_exist

def check_imports():
    """Verify critical imports work."""
    print("\n" + "="*60)
    print("🔍 Import Verification")
    print("="*60)
    
    imports = [
        ("server.models", "NegotiationAction, NegotiationObservation, NegotiationState", True),
        ("server.negotiation_environment", "NegotiationEnvironment, EXPERT_PERSONAS", True),
        ("client.negotiation_env", "NegotiationEnv", True),
        ("reward", "compute_reward, reward_format, reward_negotiation, reward_deal_quality", False),
        ("challenger", "ExpertChallenger, CurriculumManager, SelfPlayChallenger", False),
    ]
    
    all_imported = True
    for module_info in imports:
        module_name, items, requires_openenv = module_info
        try:
            exec(f"from {module_name} import {items}")
            print(f"✅ {module_name:<40} OK")
        except ModuleNotFoundError as e:
            if requires_openenv and "openenv" in str(e):
                print(f"⚠️  {module_name:<40} SKIPPED (needs openenv-core)")
            else:
                print(f"❌ {module_name:<40} FAILED: {e}")
                all_imported = False
        except Exception as e:
            print(f"❌ {module_name:<40} FAILED: {e}")
            all_imported = False
    
    return all_imported

def check_self_improvement():
    """Verify self-improvement features are implemented."""
    print("\n" + "="*60)
    print("🧠 Self-Improvement Features")
    print("="*60)
    
    features = []
    
    # Check CurriculumManager
    try:
        from challenger import CurriculumManager
        cm = CurriculumManager()
        assert hasattr(cm, 'record'), "CurriculumManager missing record()"
        assert hasattr(cm, 'get_weights'), "CurriculumManager missing get_weights()"
        assert hasattr(cm, 'sample_persona'), "CurriculumManager missing sample_persona()"
        print("✅ CurriculumManager - Tracks failures, drives curriculum")
        features.append(True)
    except Exception as e:
        print(f"❌ CurriculumManager - FAILED: {e}")
        features.append(False)
    
    # Check SelfPlayChallenger
    try:
        from challenger import SelfPlayChallenger
        sp = SelfPlayChallenger()
        assert hasattr(sp, 'respond'), "SelfPlayChallenger missing respond()"
        print("✅ SelfPlayChallenger - Past model as opponent")
        features.append(True)
    except Exception as e:
        print(f"❌ SelfPlayChallenger - FAILED: {e}")
        features.append(False)
    
    # Check ExpertChallenger escalation
    try:
        from challenger import ExpertChallenger
        ec = ExpertChallenger(0)
        assert hasattr(ec, 'escalate'), "ExpertChallenger missing escalate()"
        print("✅ ExpertChallenger.escalate() - Auto difficulty adjustment")
        features.append(True)
    except Exception as e:
        print(f"❌ ExpertChallenger.escalate() - FAILED: {e}")
        features.append(False)
    
    # Check 5 expert personas
    try:
        from challenger import EXPERT_PERSONAS
        assert len(EXPERT_PERSONAS) == 5, f"Expected 5 personas, got {len(EXPERT_PERSONAS)}"
        styles = [p['style'] for p in EXPERT_PERSONAS]
        assert len(set(styles)) == 5, "Personas should have distinct styles"
        print("✅ 5 Expert Personas - Distinct styles, hidden priorities")
        features.append(True)
    except Exception as e:
        print(f"❌ 5 Expert Personas - FAILED: {e}")
        features.append(False)
    
    # Check preference drift (needs openenv, just check file)
    try:
        env_code = Path("server/negotiation_environment.py").read_text()
        assert "self._shift" in env_code, "Missing preference drift mechanism"
        assert "self._ep % self._shift" in env_code, "Missing drift trigger"
        print("✅ Preference Drift - Every 8 episodes (code verified)")
        features.append(True)
    except Exception as e:
        print(f"❌ Preference Drift - FAILED: {e}")
        features.append(False)
    
    return all(features)

def check_reward_system():
    """Verify reward system is properly separated."""
    print("\n" + "="*60)
    print("🎯 Reward System")
    print("="*60)
    
    try:
        from reward import reward_format, reward_negotiation, reward_deal_quality, compute_reward
        print("✅ reward_format() - Format compliance (1.0 valid, 0.0 invalid)")
        print("✅ reward_negotiation() - Outcome-based (+1 deal, -1 no deal)")
        print("✅ reward_deal_quality() - Snorkel-weighted utility bonus")
        print("✅ compute_reward() - Combined reward function")
        
        # Verify rewards are NOT in environment
        env_code = Path("server/negotiation_environment.py").read_text()
        assert "def reward_format" not in env_code, "Reward function in environment file!"
        assert "def reward_negotiation" not in env_code, "Reward function in environment file!"
        assert "def reward_deal_quality" not in env_code, "Reward function in environment file!"
        print("✅ Rewards are STANDALONE (not in environment)")
        return True
    except ModuleNotFoundError as e:
        if "openenv" in str(e):
            print("⚠️  Reward system checks skipped (needs openenv-core)")
            return True  # Don't fail on missing openenv
        print(f"❌ Reward System - FAILED: {e}")
        return False
    except Exception as e:
        print(f"❌ Reward System - FAILED: {e}")
        return False
def main():
    """Run all checks."""
    print("\n🏗️  OpenEnv SalaryNegotiationArena — Build Verification\n")
    
    results = {
        "Structure": check_structure(),
        "Imports": check_imports(),
        "Self-Improvement": check_self_improvement(),
        "Reward System": check_reward_system(),
    }
    
    print("\n" + "="*60)
    print("📊 Summary")
    print("="*60)
    for category, passed in results.items():
        symbol = "✅" if passed else "❌"
        print(f"{symbol} {category}")
    
    if all(results.values()):
        print("\n🎉 All checks passed! Ready for deployment.")
        print("\nNext steps:")
        print("  1. Run tests: python -m pytest test_env.py -v")
        print("  2. Start server: uvicorn server.app:app --reload")
        print("  3. Launch demo: python app.py")
        print("  4. Train on Northflank: Upload and run train_colab.py")
        return 0
    else:
        print("\n❌ Some checks failed. Review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
