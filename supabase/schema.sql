-- Cloud chat history for the NLGEP Chatbot.
-- Run this once in the Supabase SQL Editor.

create table if not exists public.chats (
  chat_id text primary key,
  messages jsonb not null default '[]'::jsonb,
  updated_at timestamptz not null default now()
);

alter table public.chats enable row level security;

-- The chatbot server uses the Supabase secret key, so the API is intentionally
-- not exposed to the browser. No anon/authenticated policies are required.
grant all on table public.chats to service_role;
