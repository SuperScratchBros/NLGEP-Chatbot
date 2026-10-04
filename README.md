# NLGEP AI Chatbot

A small FastAPI web chatbot that supports Groq, Gemini, OpenRouter, web search, tools, voice transcription, image generation, and optional Supabase cloud chat history.

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://localhost:8000.

## Render

The repository includes `render.yaml`. Set these environment variables in Render:

| Key | Purpose |
| --- | --- |
| `GROQ_API_KEY` | Groq chat models and voice transcription |
| `TAVILY_API_KEY` | Web search |
| `GEMINI_API_KEY` | Optional Gemini chat provider |
| `GEMINI_MODEL` | Optional Gemini model |
| `OPENROUTER_API_KEY` | Optional OpenRouter chat provider |
| `OPENROUTER_MODEL` | Optional OpenRouter model |
| `HF_TOKEN` | Hugging Face token with the Inference Providers permission |
| `HF_IMAGE_MODEL` | Image model; default is `black-forest-labs/FLUX.1-schnell` |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_SECRET_KEY` | Supabase server secret key; use an `sb_secret_...` key when available |
| `UPSTASH_REDIS_REST_URL` | Optional Upstash REST URL |
| `UPSTASH_REDIS_REST_TOKEN` | Optional Upstash REST token |

Never commit real API keys or a `.env` file.

## Image generation

The Image button uses Hugging Face Inference Providers and the configured `HF_IMAGE_MODEL`. The current default is FLUX.1-schnell. Hugging Face currently provides a small monthly free-user credit for Inference Providers; it is a limited free tier, not unlimited image generation. citeturn845468search4turn736274search0

Create a Hugging Face token with permission to make Inference Providers calls and store it in Render as `HF_TOKEN`. Hugging Face documents `InferenceClient.text_to_image()` for this task. citeturn845468search3turn736274search0

## Cloud chat history

Run `supabase/schema.sql` once in the Supabase SQL Editor.

Then set `SUPABASE_URL` and `SUPABASE_SECRET_KEY` in Render. The secret key is used only on the server and is never sent to the browser. Supabase currently recommends the `sb_secret_...` server key, replacing the older `service_role` key. citeturn569393search9turn569393search12

The browser keeps an opaque chat ID in localStorage and the server stores the associated message history in Supabase.

## Notes

- Chat still works from browser localStorage even when Supabase cloud history is not configured.
- `DISABLED_FEATURES=image` disables the Image button.
- `DISABLED_FEATURES=cloudchat` can be used to disable cloud chat behavior at the application level if needed.
- Basic per-IP rate limiting remains enabled in the server.
