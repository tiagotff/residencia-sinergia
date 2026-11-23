-- =====================================================================
-- Migração 003: Alinhamento do schema afastamento ao Guia A1 v0.7
-- Preserva 100% de compatibilidade com 001_schema_inicial.sql (3FN),
-- adicionando as tabelas de arquivo (SHA-256), auditoria por campo,
-- baixa física dedicada, CID restrito com RLS e views de governança.
-- =====================================================================

-- ---------------------------------------------------------------------
-- 1. Extensão aditiva de afastamento.atestado
-- ---------------------------------------------------------------------
ALTER TABLE afastamento.atestado
    ADD COLUMN IF NOT EXISTS id_caso BIGINT,
    ADD COLUMN IF NOT EXISTS tipo_documento VARCHAR(30) NOT NULL DEFAULT 'atestado_medico',
    ADD COLUMN IF NOT EXISTS prazos_legais TEXT[] NOT NULL DEFAULT '{}';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'ck_atestado_tipo_documento'
          AND conrelid = 'afastamento.atestado'::regclass
    ) THEN
        ALTER TABLE afastamento.atestado
            ADD CONSTRAINT ck_atestado_tipo_documento
            CHECK (tipo_documento IN ('atestado_medico', 'declaracao_comparecimento', 'outro'));
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'ck_atestado_prazos_legais'
          AND conrelid = 'afastamento.atestado'::regclass
    ) THEN
        ALTER TABLE afastamento.atestado
            ADD CONSTRAINT ck_atestado_prazos_legais
            CHECK (prazos_legais <@ ARRAY['inss_16_dias', 'exame_retorno']::text[]);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS ix_atestado_caso
    ON afastamento.atestado (id_caso)
    WHERE id_caso IS NOT NULL;

-- ---------------------------------------------------------------------
-- 2. afastamento.atestado_arquivo (integridade e duplicidade por SHA-256)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS afastamento.atestado_arquivo (
    id_arquivo      BIGINT       GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    id_atestado     BIGINT       NOT NULL REFERENCES afastamento.atestado (id_atestado) ON DELETE CASCADE,
    id_caso         BIGINT,
    id_colaborador  BIGINT       NOT NULL REFERENCES cadastro.colaborador (id_colaborador),
    arquivo_uri     VARCHAR(500) NOT NULL,
    mime            VARCHAR(100),
    sha256          CHAR(64)     NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    enviado_em      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_atestado_arquivo_sha256
    ON afastamento.atestado_arquivo (sha256);

CREATE INDEX IF NOT EXISTS ix_atestado_arquivo_colab_sha256
    ON afastamento.atestado_arquivo (id_colaborador, sha256);

CREATE INDEX IF NOT EXISTS ix_atestado_arquivo_atestado
    ON afastamento.atestado_arquivo (id_atestado);

-- ---------------------------------------------------------------------
-- 3. afastamento.atestado_campo (memória stateless e auditoria IA vs MANUAL)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS afastamento.atestado_campo (
    id_campo                    BIGINT       GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    id_atestado                 BIGINT       NOT NULL REFERENCES afastamento.atestado (id_atestado) ON DELETE CASCADE,
    id_caso                     BIGINT,
    nome_campo                  VARCHAR(60)  NOT NULL,
    valor_extraido              TEXT,
    valor_confirmado            TEXT,
    confianca                   NUMERIC(5,4) CHECK (confianca IS NULL OR (confianca >= 0 AND confianca <= 1)),
    origem                      VARCHAR(10)  NOT NULL DEFAULT 'IA' CHECK (origem IN ('IA', 'MANUAL')),
    vazio_por_baixa_confianca   BOOLEAN      NOT NULL DEFAULT FALSE,
    atualizado_em               TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (id_atestado, nome_campo)
);

CREATE INDEX IF NOT EXISTS ix_atestado_campo_caso
    ON afastamento.atestado_campo (id_caso)
    WHERE id_caso IS NOT NULL;

-- ---------------------------------------------------------------------
-- 4. afastamento.baixa_fisica (recebimento da via original pelo RH)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS afastamento.baixa_fisica (
    id_baixa       BIGINT       GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    id_atestado    BIGINT       NOT NULL UNIQUE REFERENCES afastamento.atestado (id_atestado) ON DELETE CASCADE,
    id_usuario_rh  BIGINT       NOT NULL REFERENCES cadastro.usuario (id_usuario),
    recebido_por   VARCHAR(160) NOT NULL,
    data_hora      TIMESTAMPTZ  NOT NULL,
    registrado_em  TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_baixa_fisica_data_hora
    ON afastamento.baixa_fisica (data_hora);

-- ---------------------------------------------------------------------
-- 5. afastamento.cid_restrito (isolamento LGPD Art. 11 com RLS)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS afastamento.cid_restrito (
    id_atestado    BIGINT      PRIMARY KEY REFERENCES afastamento.atestado (id_atestado) ON DELETE CASCADE,
    codigo_cid     VARCHAR(8)  NOT NULL REFERENCES afastamento.cid (codigo_cid),
    registrado_em  TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE afastamento.cid_restrito ENABLE ROW LEVEL SECURITY;

-- Isolamento LGPD Art. 11: permite leitura/escrita apenas ao papel proprietário
-- do serviço ('sinergia') ou sessões explicitamente autenticadas como A1/RH/ADMIN,
-- impedindo acesso indevido de agentes sem privilégio clínico (ex.: A2 / gestores).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_policies
        WHERE schemaname = 'afastamento'
          AND tablename = 'cid_restrito'
          AND policyname = 'p_cid_restrito_acesso_autorizado'
    ) THEN
        CREATE POLICY p_cid_restrito_acesso_autorizado
            ON afastamento.cid_restrito
            USING (
                current_user = 'sinergia'
                OR current_setting('sinergia.perfil_acesso', true) IN ('A1', 'RH', 'ADMIN')
            )
            WITH CHECK (
                current_user = 'sinergia'
                OR current_setting('sinergia.perfil_acesso', true) IN ('A1', 'RH', 'ADMIN')
            );
    END IF;
END $$;

-- ---------------------------------------------------------------------
-- 6. Views de governança e compatibilidade (A0 / A2 / A3)
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW afastamento.vw_atestado_sem_cid AS
SELECT a.id_atestado,
       a.id_caso,
       a.id_colaborador,
       a.id_profissional,
       a.id_municipio,
       a.tipo_documento,
       a.data_emissao,
       a.inicio_afastamento,
       a.fim_afastamento,
       a.arquivo_uri,
       a.origem_dados,
       a.status,
       a.postado_em,
       a.prazo_entrega_fisica,
       a.prazos_legais,
       a.recebido_fisico_em,
       a.recebido_fisico_por
FROM afastamento.atestado a;

CREATE OR REPLACE VIEW afastamento.vw_baixa_fisica AS
SELECT bf.id_baixa,
       bf.id_atestado,
       a.id_caso,
       a.id_colaborador,
       a.prazo_entrega_fisica,
       bf.id_usuario_rh,
       bf.recebido_por,
       bf.data_hora,
       bf.registrado_em,
       (a.prazo_entrega_fisica IS NOT NULL AND (bf.data_hora AT TIME ZONE 'UTC')::date <= a.prazo_entrega_fisica) AS entregue_no_prazo
FROM afastamento.baixa_fisica bf
JOIN afastamento.atestado a ON a.id_atestado = bf.id_atestado;
