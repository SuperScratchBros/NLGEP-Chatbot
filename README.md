# AI Chatbot Models: Groq+Tavily+Gemini+Openrouter

A small web chatbot. Models include OpenAI Chat GPT 4.0, Gemini 3.8 Flash, and mixed. Tavily provides web search when the model needs current info. Built to run on Render's free tier.

## Run locally
```bash
pip install -r requirements.txt
cp .env.example .env     # then put your real keys in .env
uvicorn main:app --reload
```
Open http://localhost:8000

## Deploy on Render
1. Push this folder to a public GitHub repo (never commit `.env`).
2. In Render: **New > Blueprint**, pick the repo. It reads `render.yaml`.
   (Or **New > Web Service**, runtime Python, instance type **Free**, build `pip install -r requirements.txt`, start `uvicorn main:app --host 0.0.0.0 --port $PORT`. Add the two API keys under Environment. The `.python-version` file pins Python 3.12.)
3. When prompted, set `GROQ_API_KEY` and `TAVILY_API_KEY`.
4. Deploy. The free service sleeps after ~15 min idle; the first request afterward is slow.

## Notes
- Chat history lives in the browser (localStorage), so the server stays stateless.
- Change the model with the optional `GROQ_MODEL` env var.
- Basic per-IP rate limit (20/min) protects your keys on a public URL. Edit `RATE_LIMIT` in `main.py`.
