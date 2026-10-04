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

- These are the API keys you need to put in on the enviorment section
- # AI Chatbot Models: Groq + Tavily + Gemini + OpenRouter

A web chatbot featuring model support across OpenAI, Gemini, Groq, and OpenRouter, integrated with Tavily live web search for real-time information retrieval. Designed to be run locally or deployed seamlessly on Render.

---

## 🔑 Environment Variables & API Keys

Below are all environment variables used by the application:

| Key | Description / Example Value |
| :--- | :--- |
| `ACCESS_CODE` | Password/code required to access the chatbot interface. |
| `DAILY_BUDGET_GEMINI` | Daily budget limit for Gemini API usage (e.g., `10.00`). |
| `DAILY_BUDGET_GROQ` | Daily budget limit for Groq API usage (e.g., `10.00`). |
| `DAILY_LIMIT_MAX` | Maximum daily usage/request cap (e.g., `100`). |
| `DAILY_LIMIT_MIN` | Minimum daily usage threshold (e.g., `10`). |
| `GEMINI_API_KEY` | API key from [Google AI Studio](https://aistudio.google.com/). |
| `GEMINI_MODEL` | Default Gemini model (e.g., `gemini-1.5-flash`). |
| `GROQ_API_KEY` | API key from [Groq Console](https://console.groq.com/). |
| `GROQ_MODEL` | Default Groq model (e.g., `llama3-70b-8192`). |
| `MISTRAL_API_KEY` | API key from Mistral AI Studio. |
| `MISTRAL_MODEL` | Default Mistral chat model (default: `mistral-large-latest`). |
| `MISTRAL_IMAGE_MODEL` | Mistral model used by the image-generation agent (default: `mistral-medium-latest`). |
| `MISTRAL_IMAGE_AGENT_ID` | Optional existing Mistral image-generation agent ID; leave blank to let the app create/cache one. |
| `OPENROUTER_API_KEY` | API key from [OpenRouter](https://openrouter.ai/). |
| `PYTHON_VERSION` | Python runtime version (e.g., `3.12.0`). |
| `TAVILY_API_KEY` | API key for web search capabilities from [Tavily](https://tavily.com/). |
| `UPSTASH_REDIS_REST_TOKEN` | REST API token from [Upstash Redis](https://upstash.com/). |
| `UPSTASH_REDIS_REST_URL` | REST API URL from [Upstash Redis](https://upstash.com/). |

---

## ⚙️ Local Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>


### Mistral
Set `MISTRAL_API_KEY` in Render. Mistral appears in the model selector and participates in compatibility fallback. To disable it, add `mistral` to `DISABLED_FEATURES`. The Image toggle uses Mistral's built-in image-generation agent tool.
