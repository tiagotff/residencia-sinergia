-- =====================================================================
-- Migração 002: Adicionar metadados JSONB na tabela rag.trecho
-- =====================================================================

ALTER TABLE rag.trecho
ADD COLUMN IF NOT EXISTS metadados JSONB NOT NULL DEFAULT '{}'::jsonb;

-- Índice GIN para busca e filtros rápidos por chaves em metadados (ex: página, seções)
CREATE INDEX IF NOT EXISTS ix_trecho_metadados ON rag.trecho USING gin (metadados);

