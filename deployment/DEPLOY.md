# Deployment Guide

## Overview

This app runs the Dubai Building Code RAG API on Render (free tier).
The local Ollama dependency is replaced with two free cloud APIs:

| Component | Service | Purpose |
|-----------|---------|---------|
| LLM | Groq | Generates answers from retrieved context |
| Embeddings | Hugging Face Inference API | Converts questions into vectors for search |
| Hosting | Render | Runs the FastAPI server |

---

## Step 1 — Copy the index files

The FAISS index and metadata must be in this folder before deploying.
Copy them from `rag_api_test_app/`:

```
rag_api_test_app/index.faiss     → deployment/index.faiss
rag_api_test_app/index_meta.pkl  → deployment/index_meta.pkl
```

These files were built with `nomic-embed-text` and do not need to be
rebuilt — the HuggingFace Inference API serves the same model.

---

## Step 2 — Get API keys

### Groq (LLM)
1. Sign up at https://console.groq.com
2. Go to **API Keys** → **Create API Key**
3. Copy the key (starts with `gsk_...`)

### Hugging Face (Embeddings)
1. Sign up at https://huggingface.co
2. Go to **Settings → Access Tokens** → **New token**
3. Select **Read** role — that is sufficient
4. Copy the token (starts with `hf_...`)

---

## Step 3 — Push to GitHub

The `deployment/` folder must be in a GitHub repository.
If you don't have one yet:

```bash
cd deployment
git init
git add .
git commit -m "initial deployment"
gh repo create dubai-building-code-rag --public --push --source .
```

Or push to an existing repo and note the folder path.

> **Note:** `index.faiss` (~6.5 MB) and `index_meta.pkl` (~2 MB) must be
> committed too. Add them explicitly if your `.gitignore` excludes `*.pkl`
> or `*.faiss`.

---

## Step 4 — Deploy on Render

1. Go to https://render.com and sign in
2. Click **New → Web Service**
3. Connect your GitHub account and select the repository
4. If the `deployment/` folder is not the repo root, set **Root Directory** to `deployment`
5. Render will detect `render.yaml` automatically — the build and start
   commands are already configured inside it
6. Click **Advanced** → **Add Environment Variable** and add:

   | Key | Value |
   |-----|-------|
   | `GROQ_API_KEY` | your Groq key (`gsk_...`) |
   | `HF_API_KEY` | your HuggingFace token (`hf_...`) |

7. Click **Create Web Service**

Render will install dependencies and start the server.
The public URL will appear at the top of the dashboard (e.g. `https://dubai-rag.onrender.com`).

---

## Step 5 — Verify

Open the public URL in a browser. You should see the chat UI.
The green dot in the header confirms the API is live.

Test with a question like:
> *What is the minimum ceiling height for residential rooms?*

---

## Changing the LLM model

Edit the `GROQ_MODEL` constant in `api.py`:

```python
GROQ_MODEL = "llama-3.3-70b-versatile"   # default — best quality
GROQ_MODEL = "llama-3.1-8b-instant"      # faster, higher rate limit
GROQ_MODEL = "gemma2-9b-it"              # closest to local gemma4:e4b
```

Commit and push — Render redeploys automatically.

---

## Free tier limits

| Service | Limit | Impact |
|---------|-------|--------|
| Groq (`llama-3.3-70b`) | ~100K tokens/day | ~30–50 questions/day |
| Groq (`llama-3.1-8b`) | ~500K tokens/day | ~200+ questions/day |
| HuggingFace Inference API | Shared, rate-limited | May be slow under load |
| Render free tier | Spins down after 15 min idle | First request after idle takes ~30s |

---

## Troubleshooting

**502 Embedding request failed**
- HuggingFace model may be loading (cold start). Wait 20s and retry.
- Check `HF_API_KEY` is set correctly in Render environment variables.

**502 Groq API error**
- Check `GROQ_API_KEY` is set correctly.
- You may have hit the daily token limit — check https://console.groq.com/usage.

**App not starting / index not found**
- Confirm `index.faiss` and `index_meta.pkl` are committed to the repo.
- Check the Render build logs for the exact error.

**Render spins down (slow first response)**
- This is normal on the free tier after 15 minutes of no traffic.
- Upgrade to a paid instance ($7/month) to keep it always-on.
