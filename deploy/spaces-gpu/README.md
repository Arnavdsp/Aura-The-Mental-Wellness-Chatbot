---
title: Aura Wellness Coach
emoji: 🌿
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.27.0
python_version: "3.12.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: A multimodal wellness coach running real Gemma 3n
startup_duration_timeout: 1h
---

# Aura

A multimodal mental wellness coach on **Gemma 3n E4B**. Type, speak, or show it
an image. It reads tone as well as words, tracks the themes you keep returning
to, and replies with reflection and one open question rather than advice.

Source: <https://github.com/Arnavdsp/Gemma-3n-Hackathon>

## What is actually running

Real Gemma 3n on a ZeroGPU worker — not a fallback, not an API proxy. The model
takes text, images and audio natively through one chat template.

The design decision worth looking at is *where* each part runs:

| Main process | GPU worker |
|---|---|
| crisis screen, affect estimate, topic graph, prompt assembly | `model.generate` |

The crisis screen runs **before** anything reaches the model, and on a crisis
match the model is never invoked — so no prompt can route around it, because
there is no prompt. Keeping the screen out of the GPU function is what makes
that structural rather than a matter of instruction-following.

## Not a medical device

Aura is a wellness companion, not a clinician. It does not diagnose, and it is
not a substitute for a person. On a crisis match it stops coaching and shows
real helplines.
