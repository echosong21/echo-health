-- 记忆层表结构（PostgreSQL 18+）
--
-- **为什么是两张表而不是一张**
-- 情景记忆与语义记忆的失效策略完全不同：
--   · episodes 按时间累积，可滚动裁剪（只保留最近 N 轮）
--   · facts    是"稳定事实"，写了就该一直有效，直到被显式更新
-- 混在一张表里会导致"裁剪对话时把用户过敏史也删了"。
--
-- **为什么 facts 用扁平 KV 而不是固定列**
-- 固定列（allergies/chronic/diet_goal...）每加一种事实就要 ALTER TABLE，
-- 而事实类型会随领域变化（健康领域有过敏，法律领域就没有）。
-- 扁平 KV 加一个 key 就完事，这是"未来不大改"的关键。
--
-- **为什么 meta 用 JSONB 而不是 TEXT**
-- 只放可枚举的元信息（命中形态、上下文条数），但需要能按条件查
-- （例：找出所有 tables 形态零命中的请求）。JSONB 支持索引与运算符，
-- TEXT 只能模糊匹配。
--
-- ⚠️ **meta 里不许放原文** —— 与 observability/audit.py 的 args_summary
-- 同一条纪律：日志/元数据里出现用户原文就是隐私泄露。

BEGIN;

-- ── 情景记忆：发生过什么 ─────────────────────────────────
CREATE TABLE IF NOT EXISTS episodes (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT        NOT NULL,
    user_id     TEXT        NOT NULL DEFAULT 'u_default',
    turn        INTEGER     NOT NULL,
    role        TEXT        NOT NULL CHECK (role IN ('user', 'assistant')),
    text        TEXT        NOT NULL,
    meta        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- 同一会话内轮次不可重复：写入重试时不会产生重复记录
    UNIQUE (session_id, turn, role)
);

-- 按会话取最近 N 轮（主查询路径）
CREATE INDEX IF NOT EXISTS idx_episodes_session_turn
    ON episodes (session_id, turn DESC);

-- 跨会话按时间看某用户的历史（可选路径，为将来准备）
CREATE INDEX IF NOT EXISTS idx_episodes_user_ts
    ON episodes (user_id, ts DESC);

-- ── 语义记忆：关于用户的稳定事实 ─────────────────────────
CREATE TABLE IF NOT EXISTS facts (
    user_id     TEXT        NOT NULL DEFAULT 'u_default',
    key         TEXT        NOT NULL,
    value       TEXT        NOT NULL,
    -- 这个事实是从哪句对话记下来的（session#turn）
    -- **没有溯源就无法撤销**：用户说"我不过敏了"，得知道当初依据哪句
    evidence    TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, key)
);

COMMENT ON TABLE  episodes IS '情景记忆：按时间累积的交互记录，可滚动裁剪';
COMMENT ON TABLE  facts    IS '语义记忆：用户稳定事实，扁平 KV + 溯源';
COMMENT ON COLUMN facts.evidence IS '来源 session_id#turn，用于撤销与审计';

COMMIT;