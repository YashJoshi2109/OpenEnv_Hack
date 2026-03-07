"""Gradio demo for SalaryNegotiationArena. ONLY place LLMChallenger used."""
import json
import gradio as gr
import plotly.graph_objects as go
from server.negotiation_environment import NegotiationEnvironment, EXPERT_PERSONAS
from server.models import NegotiationAction
from reward import compute_reward, reward_format, reward_negotiation, reward_deal_quality
from challenger import CurriculumManager

def build_app() -> gr.Blocks:
    env = NegotiationEnvironment()
    curriculum = CurriculumManager()  # Track performance for curriculum learning

    with gr.Blocks(title="SalaryNegotiationArena", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🤝 SalaryNegotiationArena\n**OpenEnv Hackathon SF — Self-Improvement + Snorkel AI**\n\nNegotiate salary with AI experts who have hidden priorities. Preferences shift every 8 episodes. Your failures drive the curriculum.")


        with gr.Row():
            with gr.Column(scale=2):
                status_md = gr.Markdown("**Status:** Ready")
                expert_md = gr.Markdown("**Expert:** —")
                chatbot = gr.Chatbot(label="Negotiation", height=400, type="messages")
                with gr.Row():
                    action_dd = gr.Dropdown(["propose","accept","reject","counter","walk_away"], value="propose", label="Action")
                with gr.Row():
                    salary_sl = gr.Slider(80000,200000,value=140000,step=5000,label="Salary ($/yr)")
                    equity_sl = gr.Slider(0.0,5.0,value=2.0,step=0.25,label="Equity (%)")
                    start_sl = gr.Slider(14,180,value=60,step=7,label="Start (days)")
                msg_tb = gr.Textbox(label="Message", placeholder="Your negotiation message...")
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    reset_btn = gr.Button("New Episode", variant="secondary")
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Metrics")
                turn_md = gr.Markdown("**Turn:** 0/10")
                reward_md = gr.Markdown("**Reward:** —")
                ep_md = gr.Markdown("**Episode:** 0")
                reward_plot = gr.Plot(label="Rewards")
                gr.Markdown("### 🧠 Self-Improvement")
                curriculum_md = gr.Markdown("Curriculum weights will appear after episodes.")

        ep_rewards = gr.State([])

        def do_reset():
            obs = env.reset()
            p = env._persona()
            chat = [{"role":"assistant","content":f"🏢 **{p['name']}** ({p['style']}): {obs.challenger_message}"}]
            return (chat, f"**Status:** 🟡 Negotiating", f"**Expert:** {p['name']} ({p['style']})",
                    "**Turn:** 0/10", "**Reward:** —", f"**Episode:** {env._ep}", [], _empty_plot(),
                    f"*Expert style: {p['style']}. Infer their hidden priorities from cues.*")
        def do_send(action, salary, equity, start, msg, chat, ep_rw):
            act = NegotiationAction(action_type=action, base_salary=int(salary),
                                    equity=float(equity), start_date=int(start), message=msg or "")
            obs = env.step(act)
            # Agent message
            if action == "walk_away": agent_msg = "🚶 **You:** Walking away."
            elif action == "accept": agent_msg = "✅ **You:** I accept!"
            else: agent_msg = f"💼 **You** ({action}): ${int(salary):,}/yr, {equity}%, {int(start)}d" + (f' — "{msg}"' if msg else "")
            chat = chat + [{"role":"user","content":agent_msg}]
            # Challenger
            p = env._persona()
            emoji = {"deal_reached":"✅","no_deal":"❌","walked_away":"🚶","negotiating":"🔄"}.get(obs.phase,"🔄")
            chat = chat + [{"role":"assistant","content":f"🏢 **{p['name']}:** {obs.challenger_message} {emoji}"}]
            # Rewards
            es = {"phase":obs.phase,"turn":obs.turn,"max_turns":obs.max_turns,
                  "current_offer":{"base_salary":obs.current_offer_salary,"equity":obs.current_offer_equity,"start_date":obs.current_offer_start},
                  "profile_idx":env._pidx}
            rf = reward_format(json.dumps(act.model_dump()))
            rn = reward_negotiation(json.dumps(act.model_dump()), env_state=es)
            rq = reward_deal_quality(json.dumps(act.model_dump()), env_state=es)
            rt = compute_reward(json.dumps(act.model_dump()), env_state=es)
            ep_rw.append({"turn":obs.turn,"format":rf,"negotiation":rn,"quality":rq,"total":rt})
            
            # Track in curriculum when episode ends
            if obs.done:
                curriculum.record(p["name"], rt)
            
            phase_d = {"negotiating":"🟡 Negotiating","deal_reached":"🟢 Deal!","no_deal":"🔴 No Deal","walked_away":"🔴 Walked Away"}
            curr_md = _curriculum_md(curriculum) if obs.done else f"*Expert style: {p['style']}. Infer their hidden priorities from cues.*"
            
            return (chat, f"**Status:** {phase_d.get(obs.phase,obs.phase)}",
                    f"**Turn:** {obs.turn}/{obs.max_turns}",
                    f"**Reward:** Fmt={rf:.1f} | Neg={rn:.1f} | Qual={rq:.1f} | **Total={rt:.2f}**",
                    _reward_plot(ep_rw), ep_rw, curr_md)| Neg={rn:.1f} | Qual={rq:.1f} | **Total={rt:.2f}**",
                    _reward_plot(ep_rw), ep_rw)

        def _reward_plot(rws):
            if not rws: return _empty_plot()
            turns = [r["turn"] for r in rws]
            fig = go.Figure()
        def _empty_plot():
            fig = go.Figure(); fig.update_layout(title="Reward Breakdown",height=300,margin=dict(l=40,r=20,t=40,b=30)); return fig
        
        def _curriculum_md(curr: CurriculumManager):
            """Display curriculum weights - weak experts get MORE training."""
            if not curr.perf:
                return "*Complete episodes to see curriculum weights. Weaker personas will get more training.*"
            weights = curr.get_weights()
            lines = ["**📚 Curriculum Weights** (Higher = More Training Needed):"]
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            for name, weight in sorted_weights[:5]:  # Top 5
                avg_reward = sum(curr.perf[name])/len(curr.perf[name]) if name in curr.perf else 0
                bar = "█" * int(weight * 20)
                lines.append(f"- {name.split('—')[0].strip()}: `{bar}` ({weight:.2f}) | Avg Reward: {avg_reward:.2f}")
            return "\n".join(lines)

        reset_btn.click(do_reset, outputs=[chatbot,status_md,expert_md,turn_md,reward_md,ep_md,ep_rewards,reward_plot,curriculum_md])
        send_btn.click(do_send, inputs=[action_dd,salary_sl,equity_sl,start_sl,msg_tb,chatbot,ep_rewards],
                       outputs=[chatbot,status_md,turn_md,reward_md,reward_plot,ep_rewards,curriculum_md])
    return app

        def _empty_plot():
            fig = go.Figure(); fig.update_layout(title="Reward Breakdown",height=300,margin=dict(l=40,r=20,t=40,b=30)); return fig

        reset_btn.click(do_reset, outputs=[chatbot,status_md,expert_md,turn_md,reward_md,ep_md,ep_rewards,reward_plot,curriculum_md])
        send_btn.click(do_send, inputs=[action_dd,salary_sl,equity_sl,start_sl,msg_tb,chatbot,ep_rewards],
                       outputs=[chatbot,status_md,turn_md,reward_md,reward_plot,ep_rewards])
    return app
