# NLGEP Chat / InfiNet

NLGEP Chat is a Supabase-backed global lobby where users pick a unique username and message each other. It uses the same dark-green glass look as the NLGEP browser.

## Render environment

Required:
- `SUPABASE_URL`
- `SUPABASE_SECRET_KEY`

Supabase recommends the newer `sb_secret_...` server key for backend code. Keep it only on the server.

Run `supabase/schema.sql` once in the Supabase SQL Editor before using Chat.

The chat UI uses lightweight polling through the FastAPI backend.

Important: you need **both** your Supabase project URL and server key. The API key alone is not enough to identify the project endpoint.

Optional:
- `DISABLED_FEATURES=cloudchat` turns chat off
- `TRUST_PROXY_HEADERS=true` when running behind a proxy that sets `X-Forwarded-For`

Never commit real API keys or a `.env` file.
