-- ONE-TIME RESET: deletes every existing account, session and chat message.
-- Run this in the Supabase SQL Editor BEFORE deploying the username+password update.
-- (Do not re-run it later unless you want to wipe everything again.)

-- 1. Add the new columns the password/guest login needs.
alter table public.chat_users add column if not exists password_hash text;
alter table public.chat_users add column if not exists is_guest boolean not null default false;
create index if not exists chat_users_guest_idx on public.chat_users(is_guest, created_at);

-- 2. Wipe all accounts (sessions and messages are removed with them).
truncate table public.chat_messages, public.chat_sessions, public.chat_users restart identity cascade;
