-- =====================================================================
-- Migração 004: Adiciona coluna nome_doenca em afastamento.cid,
-- afastamento.cid_restrito e afastamento.atestado, com backfill e
-- sincronização automática (trigger) entre nome_doenca e descricao.
-- Idempotente: segura para executar múltiplas vezes.
-- =====================================================================

-- ---------------------------------------------------------------------
-- 1. afastamento.cid
-- ---------------------------------------------------------------------
ALTER TABLE afastamento.cid
    ADD COLUMN IF NOT EXISTS nome_doenca VARCHAR(255);

ALTER TABLE afastamento.cid
    ALTER COLUMN descricao TYPE TEXT;

UPDATE afastamento.cid
SET nome_doenca = LEFT(COALESCE(NULLIF(btrim(descricao), ''), 'CID ' || codigo_cid), 255)
WHERE nome_doenca IS NULL OR btrim(nome_doenca) = '';

ALTER TABLE afastamento.cid
    ALTER COLUMN nome_doenca SET NOT NULL;

-- Trigger de sincronização nome_doenca <-> descricao
CREATE OR REPLACE FUNCTION afastamento.fn_sincronizar_nome_doenca_cid()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.nome_doenca IS NULL OR btrim(NEW.nome_doenca) = '' THEN
        NEW.nome_doenca := LEFT(
            COALESCE(NULLIF(btrim(NEW.descricao), ''), 'CID ' || NEW.codigo_cid),
            255
        );
    END IF;
    IF NEW.descricao IS NULL OR btrim(NEW.descricao) = '' THEN
        NEW.descricao := NEW.nome_doenca;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_sincronizar_nome_doenca_cid ON afastamento.cid;
CREATE TRIGGER trg_sincronizar_nome_doenca_cid
BEFORE INSERT OR UPDATE ON afastamento.cid
FOR EACH ROW
EXECUTE FUNCTION afastamento.fn_sincronizar_nome_doenca_cid();

-- ---------------------------------------------------------------------
-- 2. afastamento.cid_restrito
-- ---------------------------------------------------------------------
ALTER TABLE afastamento.cid_restrito
    ADD COLUMN IF NOT EXISTS nome_doenca VARCHAR(255);

UPDATE afastamento.cid_restrito cr
SET nome_doenca = COALESCE(NULLIF(btrim(c.nome_doenca), ''), c.descricao)
FROM afastamento.cid c
WHERE cr.codigo_cid = c.codigo_cid
  AND (cr.nome_doenca IS NULL OR btrim(cr.nome_doenca) = '');

-- ---------------------------------------------------------------------
-- 3. afastamento.atestado (backfill de codigo_cid e nome_doenca)
-- ---------------------------------------------------------------------
ALTER TABLE afastamento.atestado
    ADD COLUMN IF NOT EXISTS nome_doenca VARCHAR(255);

UPDATE afastamento.atestado a
SET codigo_cid = cr.codigo_cid
FROM afastamento.cid_restrito cr
WHERE cr.id_atestado = a.id_atestado
  AND a.codigo_cid IS NULL;

UPDATE afastamento.atestado a
SET codigo_cid = c.codigo_cid
FROM afastamento.atestado_campo ac
JOIN afastamento.cid c
  ON UPPER(c.codigo_cid) = UPPER(btrim(COALESCE(NULLIF(btrim(ac.valor_confirmado), ''), ac.valor_extraido)))
WHERE ac.id_atestado = a.id_atestado
  AND ac.nome_campo = 'cid'
  AND a.codigo_cid IS NULL;

-- Garante que atestados confirmados com CID em atestado_campo ou codigo_cid
-- também possuam registro correspondente em afastamento.cid_restrito.
INSERT INTO afastamento.cid_restrito (id_atestado, codigo_cid, nome_doenca)
SELECT a.id_atestado,
       a.codigo_cid,
       COALESCE(NULLIF(btrim(c.nome_doenca), ''), c.descricao)
FROM afastamento.atestado a
JOIN afastamento.cid c ON c.codigo_cid = a.codigo_cid
WHERE a.codigo_cid IS NOT NULL
  AND (
      a.status = 'CONFIRMADO'
      OR EXISTS (
          SELECT 1
          FROM afastamento.atestado_campo ac
          WHERE ac.id_atestado = a.id_atestado
            AND ac.nome_campo = 'cid'
      )
  )
ON CONFLICT (id_atestado) DO UPDATE
SET codigo_cid = EXCLUDED.codigo_cid,
    nome_doenca = COALESCE(
        NULLIF(btrim(afastamento.cid_restrito.nome_doenca), ''),
        EXCLUDED.nome_doenca
    );

UPDATE afastamento.atestado a
SET nome_doenca = COALESCE(NULLIF(btrim(c.nome_doenca), ''), c.descricao)
FROM afastamento.cid c
WHERE a.codigo_cid = c.codigo_cid
  AND (a.nome_doenca IS NULL OR btrim(a.nome_doenca) = '');

-- ---------------------------------------------------------------------
-- 4. Preserva view de isolamento LGPD Art. 11 sem codigo_cid / nome_doenca
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
