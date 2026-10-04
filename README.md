# Infinity Network / InfiNet

Infinity Network is the home of **NLGEP AI** and community chatting.

## Services

**NLGEP AI** — the AI assistant with Groq, Gemini, and optional OpenRouter model selection, plus web tools, attachments, and voice transcription.

**Chatting** — a Supabase-backed global InfiNet lobby where users choose unique usernames and message each other.

## Render environment

Required:
- `GROQ_API_KEY`
- `TAVILY_API_KEY`

Optional AI providers:
- `GEMINI_API_KEY`
- `GEMINI_MODEL`
- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL`

InfiNet chatting:
- `SUPABASE_URL`
- `SUPABASE_SECRET_KEY`

Supabase recommends the newer `sb_secret_...` server key for backend code. Keep it only on the server. citeturn653662search1

Run `supabase/schema.sql` once in the Supabase SQL Editor before using Chatting.

The current chat UI uses lightweight polling through the FastAPI backend. Supabase Realtime can provide WebSocket database changes and is documented for live chat; Broadcast is the recommended approach for larger-scale real-time applications. citeturn653662search0turn653662search2

Important: you need **both** your Supabase project URL and server key. The API key alone is not enough to identify the project endpoint.

Never commit real API keys or a `.env` file.
