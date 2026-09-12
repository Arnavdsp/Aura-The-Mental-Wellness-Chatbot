---
title: Aura — Multimodal Wellness Coach
emoji: 🌿
colorFrom: green
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
license: apache-2.0
short_description: A Gemma 3n wellness coach that listens in text, voice and images.
---

# Aura

A multimodal mental wellness coach. Type, speak, or show it an image; it reads
tone as well as words, notices the themes you keep returning to, and replies
with reflection and one open question. When someone describes real danger it
stops coaching and puts helplines in front of them.

Source and full documentation:
<https://github.com/Arnavdsp/Gemma-3n-Hackathon>

## What is running here

**The echo engine, not Gemma 3n.** This Space runs on a free CPU tier, and
Gemma 3n needs an accelerator to be usable — on CPU a single reply takes
minutes. What you get instead is the fallback reflective-listening coach: the
real UI, the real API, the real safety layer, with a much simpler mind behind
them.

To run the actual model, either upgrade this Space to a GPU tier and set
`AURA_ENGINE=gemma`, or run the same container on any GPU host:

```bash
docker build --build-arg EXTRAS='[ml]' -t aura:gpu .
docker run --gpus all -p 8000:8000 -e AURA_ENGINE=gemma aura:gpu
```

Two further consequences of the free tier, neither of which is a bug:

- Sessions live in the container's memory, so a restart forgets past
  conversations. Attachments are unaffected — the client sends image and audio
  bytes inline with each turn.
- Spoken replies are off (`AURA_TTS_BACKEND=none`); they need torch.

**Not a medical device.** Aura is a wellness companion, not a clinician.
