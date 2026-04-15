# llm-alignment

> An Implementation of RLHF and DPO alignment techniques, built on GPT-2 using the Anthropic HH-RLHF dataset.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers%20%7C%20TRL-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This repository demonstrates two dominant approaches to aligning large language models with human preferences:

1. **RLHF** (Reinforcement Learning from Human Feedback) — the original three-phase pipeline pioneered by InstructGPT.
2. **DPO** (Direct Preference Optimization) — a more recent, RL-free alternative that achieves comparable alignment without an explicit reward model or PPO loop.

Both pipelines are trained on the **Anthropic HH-RLHF helpfulness subset**, which provides human-written preference pairs: for each prompt, a chosen (preferred) response and a rejected (dispreferred) response.

The base model throughout is **GPT-2** (`gpt2`). It is small enough to train on a single GPU yet large enough to observe meaningful alignment behaviour.

---

## The 3-Phase RLHF Pipeline

![RLHF Pipeline](assets/RLHFPipeline.jpeg)

**Phase 1 — Supervised Fine-Tuning (SFT)**
- Start from a pretrained GPT-2 checkpoint
- Fine-tune on the chosen (preferred) responses using standard next-token prediction
- This gives a well-formatted starting point before any RL training begins

**Phase 2 — Reward Model Training**
- Take the SFT model and add a small linear head on top that outputs a single scalar score
- Train it on preference pairs using the Bradley-Terry ranking loss: the loss pushes the model to give a higher score to the chosen response than the rejected one, by taking the negative log sigmoid of the score difference
- The trained reward model translates noisy human preferences into a smooth, differentiable signal

**Phase 3 — PPO Fine-Tuning**
- Load the SFT model twice: one copy is the active policy being trained, the other is a frozen reference policy that never updates
- For each prompt, sample a response from the active policy and score it with the reward model
- Subtract a KL penalty (scaled by beta) from the reward to prevent the active policy from drifting too far from the reference — this stops reward hacking
- Run a PPO gradient update on the active policy using the penalised reward

---

## Learning Rate Schedule

All three RLHF phases use a warmup period followed by cosine annealing. The learning rate rises linearly from zero during warmup, then decays smoothly following a cosine curve down to near zero by the end of training. This avoids large early updates (which can destabilise a pretrained model) while still allowing the rate to stay high through most of training before tapering off.

![Cosine Annealing LR](assets/CosineAnnhealingLR.jpeg)

---

