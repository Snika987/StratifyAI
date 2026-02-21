CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS semantic_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    embedding JSONB NOT NULL,
    original_query TEXT NOT NULL,
    ai_response TEXT NOT NULL,
    model_name TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tool_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tool_name TEXT NOT NULL,
    args_hash TEXT NOT NULL,
    output JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(tool_name, args_hash)
);

CREATE TABLE IF NOT EXISTS logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id TEXT,
    user_id TEXT,
    raw_input TEXT,
    model_name TEXT,
    token_usage JSONB,
    tool_calls INTEGER,
    latency FLOAT,
    cache_hit BOOLEAN DEFAULT FALSE,
    retry_count INTEGER DEFAULT 0,
    status TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
