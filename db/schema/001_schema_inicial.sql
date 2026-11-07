-- =====================================================================
-- Sinergia — Modelo físico (PostgreSQL 16 + pgvector)
-- Normalização: 3ª Forma Normal
-- Convenção: data/hora gerada automaticamente com DEFAULT now()
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS vector;     -- pgvector (bases RAG)

CREATE SCHEMA IF NOT EXISTS cadastro;
CREATE SCHEMA IF NOT EXISTS afastamento;
CREATE SCHEMA IF NOT EXISTS bem_estar;
CREATE SCHEMA IF NOT EXISTS apoio;
CREATE SCHEMA IF NOT EXISTS rag;
CREATE SCHEMA IF NOT EXISTS orquestracao;

-- ---------------------------------------------------------------------
-- SCHEMA cadastro — identidade e estrutura organizacional (A1)
-- ---------------------------------------------------------------------
CREATE TABLE cadastro.uf (
    sigla_uf   CHAR(2)      PRIMARY KEY,
    nome       VARCHAR(40)  NOT NULL UNIQUE
);

CREATE TABLE cadastro.municipio (
    id_municipio  INTEGER      PRIMARY KEY,              -- código IBGE
    nome          VARCHAR(80)  NOT NULL,
    sigla_uf      CHAR(2)      NOT NULL REFERENCES cadastro.uf (sigla_uf),
    UNIQUE (nome, sigla_uf)
);

CREATE TABLE cadastro.area (
    id_area  INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    nome     VARCHAR(80) NOT NULL UNIQUE
);

CREATE TABLE cadastro.setor (
    id_setor  INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    id_area   INTEGER     NOT NULL REFERENCES cadastro.area (id_area),
    nome      VARCHAR(80) NOT NULL,
    UNIQUE (id_area, nome)
);

CREATE TABLE cadastro.cargo (
    id_cargo  INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    nome      VARCHAR(80) NOT NULL UNIQUE,
    cbo       VARCHAR(10)                                  -- Classificação Brasileira de Ocupações
);

CREATE TABLE cadastro.colaborador (
    id_colaborador  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    matricula       VARCHAR(20)  NOT NULL UNIQUE,
    nome            VARCHAR(120) NOT NULL,
    email           VARCHAR(160) NOT NULL UNIQUE,
    id_cargo        INTEGER      NOT NULL REFERENCES cadastro.cargo (id_cargo),
    id_setor        INTEGER      NOT NULL REFERENCES cadastro.setor (id_setor),
    id_municipio    INTEGER      NOT NULL REFERENCES cadastro.municipio (id_municipio),
    id_gestor       BIGINT       REFERENCES cadastro.colaborador (id_colaborador),
    data_admissao   DATE         NOT NULL,
    data_nascimento DATE         NOT NULL,
    qtd_dependentes SMALLINT     NOT NULL DEFAULT 0 CHECK (qtd_dependentes >= 0),
    jornada_inicio  TIME         NOT NULL,
    jornada_fim     TIME         NOT NULL,
    ativo           BOOLEAN      NOT NULL DEFAULT TRUE,
    criado_em       TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CHECK (jornada_fim > jornada_inicio)
);

CREATE TABLE cadastro.usuario (
    id_usuario      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    id_colaborador  BIGINT       NOT NULL UNIQUE REFERENCES cadastro.colaborador (id_colaborador),
    login           VARCHAR(160) NOT NULL UNIQUE,
    senha_hash      VARCHAR(255) NOT NULL,                 -- Argon2id; nunca texto puro
    perfil          VARCHAR(20)  NOT NULL
                    CHECK (perfil IN ('COLABORADOR','GESTOR','RH','TEC_SEG_TRABALHO','AUDITOR','ADMIN')),
    ativo           BOOLEAN      NOT NULL DEFAULT TRUE,
    criado_em       TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------
-- SCHEMA afastamento — atestados e afastamentos (A1)
-- ---------------------------------------------------------------------
CREATE TABLE afastamento.conselho_profissional (
    id_conselho  SMALLINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    sigla        VARCHAR(10)  NOT NULL UNIQUE,              -- CRM, CRO, CRP...
    descricao    VARCHAR(120) NOT NULL
);

CREATE TABLE afastamento.profissional_saude (
    id_profissional  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    id_conselho      SMALLINT     NOT NULL REFERENCES afastamento.conselho_profissional (id_conselho),
    numero_registro  VARCHAR(20)  NOT NULL,
    sigla_uf         CHAR(2)      NOT NULL REFERENCES cadastro.uf (sigla_uf),
    nome             VARCHAR(120),
    verificado       BOOLEAN      NOT NULL DEFAULT FALSE,
    UNIQUE (id_conselho, numero_registro, sigla_uf)
);

CREATE TABLE afastamento.cid (
    codigo_cid  VARCHAR(8)   PRIMARY KEY,                   -- ex.: M54.5
    descricao   VARCHAR(255) NOT NULL,
    capitulo    VARCHAR(10)  NOT NULL
);

CREATE TABLE afastamento.atestado (
    id_atestado         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    id_colaborador      BIGINT       NOT NULL REFERENCES cadastro.colaborador (id_colaborador),
    id_profissional     BIGINT       NOT NULL REFERENCES afastamento.profissional_saude (id_profissional),
    codigo_cid          VARCHAR(8)   REFERENCES afastamento.cid (codigo_cid),
    id_municipio        INTEGER      NOT NULL REFERENCES cadastro.municipio (id_municipio),
    data_emissao        DATE         NOT NULL,
    inicio_afastamento  TIMESTAMPTZ  NOT NULL,
    fim_afastamento     TIMESTAMPTZ  NOT NULL,
    arquivo_uri         VARCHAR(500) NOT NULL,
    origem_dados        VARCHAR(10)  NOT NULL CHECK (origem_dados IN ('IA','MANUAL')),
    status              VARCHAR(12)  NOT NULL DEFAULT 'PENDENTE'
                        CHECK (status IN ('PENDENTE','CONFIRMADO','REJEITADO')),
    postado_em          TIMESTAMPTZ  NOT NULL DEFAULT now(),
    prazo_entrega_fisica DATE,
    recebido_fisico_em  TIMESTAMPTZ,
    recebido_fisico_por BIGINT       REFERENCES cadastro.usuario (id_usuario),
    CHECK (fim_afastamento >= inicio_afastamento),
    CHECK ((recebido_fisico_em IS NULL) = (recebido_fisico_por IS NULL))
);

-- ---------------------------------------------------------------------
-- SCHEMA orquestracao — canais e execução de agentes (A0)
-- ---------------------------------------------------------------------
CREATE TABLE orquestracao.canal (
    id_canal  SMALLINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    nome      VARCHAR(20) NOT NULL UNIQUE                   -- CHAT, EMAIL, MENSAGEM
);

CREATE TABLE orquestracao.execucao_agente (
    id_execucao     UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id       VARCHAR(64)  NOT NULL,                  -- thread do LangGraph
    agente          VARCHAR(4)   NOT NULL CHECK (agente IN ('A0','A1','A2','A3')),
    no_grafo        VARCHAR(60)  NOT NULL,                  -- nó do LangGraph / tarefa CrewAI
    id_usuario      BIGINT       REFERENCES cadastro.usuario (id_usuario),
    modelo_llm      VARCHAR(80),
    tokens_entrada  INTEGER,
    tokens_saida    INTEGER,
    latencia_ms     INTEGER,
    status          VARCHAR(12)  NOT NULL CHECK (status IN ('SUCESSO','ERRO','INTERROMPIDO')),
    erro            TEXT,
    iniciado_em     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    finalizado_em   TIMESTAMPTZ
);

-- ---------------------------------------------------------------------
-- SCHEMA bem_estar — Agente A2
-- ---------------------------------------------------------------------
CREATE TABLE bem_estar.tipo_pausa (
    id_tipo_pausa  SMALLINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    nome           VARCHAR(40)  NOT NULL UNIQUE,            -- alongamento, hidratação, descanso visual...
    descricao      VARCHAR(255) NOT NULL
);

CREATE TABLE bem_estar.preferencia_pausa (
    id_preferencia     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    id_colaborador     BIGINT    NOT NULL REFERENCES cadastro.colaborador (id_colaborador),
    id_tipo_pausa      SMALLINT  NOT NULL REFERENCES bem_estar.tipo_pausa (id_tipo_pausa),
    id_canal           SMALLINT  NOT NULL REFERENCES orquestracao.canal (id_canal),
    intervalo_minutos  SMALLINT  NOT NULL DEFAULT 90 CHECK (intervalo_minutos BETWEEN 30 AND 240),
    duracao_minutos    SMALLINT  NOT NULL DEFAULT 5 CHECK (duracao_minutos BETWEEN 1 AND 30),
    dias_semana        SMALLINT[] NOT NULL DEFAULT '{1,2,3,4,5}',
    ativo              BOOLEAN   NOT NULL DEFAULT TRUE,
    atualizado_em      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (id_colaborador, id_tipo_pausa)
);

CREATE TABLE bem_estar.beneficio (
    id_beneficio  SMALLINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    nome          VARCHAR(80)  NOT NULL UNIQUE,
    descricao     VARCHAR(500) NOT NULL
);

CREATE TABLE bem_estar.colaborador_beneficio (
    id_colaborador  BIGINT   NOT NULL REFERENCES cadastro.colaborador (id_colaborador),
    id_beneficio    SMALLINT NOT NULL REFERENCES bem_estar.beneficio (id_beneficio),
    data_inicio     DATE     NOT NULL,
    data_fim        DATE,
    PRIMARY KEY (id_colaborador, id_beneficio, data_inicio)
);

CREATE TABLE bem_estar.treinamento (
    id_treinamento   INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    titulo           VARCHAR(120) NOT NULL UNIQUE,
    carga_horaria_h  NUMERIC(5,1) NOT NULL,
    validade_meses   SMALLINT,                               -- NULL = não vence
    id_epi           INTEGER                                 -- FK criada após apoio.epi
);

CREATE TABLE bem_estar.cargo_treinamento (
    id_cargo        INTEGER NOT NULL REFERENCES cadastro.cargo (id_cargo),
    id_treinamento  INTEGER NOT NULL REFERENCES bem_estar.treinamento (id_treinamento),
    obrigatorio     BOOLEAN NOT NULL DEFAULT TRUE,
    PRIMARY KEY (id_cargo, id_treinamento)
);

CREATE TABLE bem_estar.colaborador_treinamento (
    id_colaborador  BIGINT  NOT NULL REFERENCES cadastro.colaborador (id_colaborador),
    id_treinamento  INTEGER NOT NULL REFERENCES bem_estar.treinamento (id_treinamento),
    data_conclusao  DATE    NOT NULL,
    PRIMARY KEY (id_colaborador, id_treinamento, data_conclusao)
);

CREATE TABLE bem_estar.tipo_exame (
    id_tipo_exame        SMALLINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    nome                 VARCHAR(80) NOT NULL UNIQUE,       -- periódico, audiometria, acuidade visual...
    periodicidade_meses  SMALLINT    NOT NULL
);

CREATE TABLE bem_estar.cargo_exame (
    id_cargo       INTEGER  NOT NULL REFERENCES cadastro.cargo (id_cargo),
    id_tipo_exame  SMALLINT NOT NULL REFERENCES bem_estar.tipo_exame (id_tipo_exame),
    PRIMARY KEY (id_cargo, id_tipo_exame)
);

CREATE TABLE bem_estar.exame_colaborador (
    id_exame          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    id_colaborador    BIGINT   NOT NULL REFERENCES cadastro.colaborador (id_colaborador),
    id_tipo_exame     SMALLINT NOT NULL REFERENCES bem_estar.tipo_exame (id_tipo_exame),
    data_realizacao   DATE     NOT NULL,
    UNIQUE (id_colaborador, id_tipo_exame, data_realizacao)
);

CREATE TABLE bem_estar.vacina (
    id_vacina        SMALLINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    nome             VARCHAR(80) NOT NULL UNIQUE,
    doses_previstas  SMALLINT    NOT NULL DEFAULT 1,
    reforco_meses    SMALLINT
);

CREATE TABLE bem_estar.vacinacao_colaborador (
    id_colaborador  BIGINT   NOT NULL REFERENCES cadastro.colaborador (id_colaborador),
    id_vacina       SMALLINT NOT NULL REFERENCES bem_estar.vacina (id_vacina),
    dose            SMALLINT NOT NULL,
    data_aplicacao  DATE     NOT NULL,
    PRIMARY KEY (id_colaborador, id_vacina, dose)
);

CREATE TABLE bem_estar.evento (
    id_evento    INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    titulo       VARCHAR(120) NOT NULL,
    tipo         VARCHAR(20)  NOT NULL CHECK (tipo IN ('CAMPANHA_VACINACAO','PALESTRA','SIPAT','GINASTICA_LABORAL','OUTRO')),
    inicio       TIMESTAMPTZ  NOT NULL,
    fim          TIMESTAMPTZ  NOT NULL,
    id_area      INTEGER      REFERENCES cadastro.area (id_area),   -- NULL = toda a empresa
    CHECK (fim >= inicio)
);

CREATE TABLE bem_estar.lembrete (
    id_lembrete     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    id_colaborador  BIGINT       NOT NULL REFERENCES cadastro.colaborador (id_colaborador),
    id_canal        SMALLINT     NOT NULL REFERENCES orquestracao.canal (id_canal),
    id_execucao     UUID         REFERENCES orquestracao.execucao_agente (id_execucao),
    categoria       VARCHAR(20)  NOT NULL
                    CHECK (categoria IN ('PAUSA','DIREITO_BENEFICIO','PROCEDIMENTO','SAUDE_OCUPACIONAL','EPI_ORIENTACAO','EPI_TROCA')),
    titulo          VARCHAR(120) NOT NULL,
    conteudo        TEXT         NOT NULL,
    agendado_para   TIMESTAMPTZ  NOT NULL,
    status          VARCHAR(10)  NOT NULL DEFAULT 'GERADO'
                    CHECK (status IN ('GERADO','ENVIADO','REAGENDADO','CANCELADO')),
    gerado_em       TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE TABLE bem_estar.versao_modelo (
    id_modelo     SMALLINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    versao        VARCHAR(20)  NOT NULL UNIQUE,
    artefato_uri  VARCHAR(500) NOT NULL,                     -- arquivo ONNX
    auc_roc       NUMERIC(4,3) NOT NULL,
    precisao_at3  NUMERIC(4,3) NOT NULL,
    ativo         BOOLEAN      NOT NULL DEFAULT FALSE,
    treinado_em   TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX ux_versao_modelo_ativo ON bem_estar.versao_modelo (ativo) WHERE ativo;  -- uma versão ativa

CREATE TABLE bem_estar.recomendacao_beneficio (
    id_recomendacao  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    id_colaborador   BIGINT       NOT NULL REFERENCES cadastro.colaborador (id_colaborador),
    id_beneficio     SMALLINT     NOT NULL REFERENCES bem_estar.beneficio (id_beneficio),
    id_modelo        SMALLINT     NOT NULL REFERENCES bem_estar.versao_modelo (id_modelo),
    probabilidade    NUMERIC(4,3) NOT NULL CHECK (probabilidade BETWEEN 0 AND 1),
    posicao          SMALLINT     NOT NULL CHECK (posicao BETWEEN 1 AND 3),
    status           VARCHAR(10)  NOT NULL DEFAULT 'GERADA'
                     CHECK (status IN ('GERADA','ENVIADA','ACEITA','RECUSADA')),
    gerada_em        TIMESTAMPTZ  NOT NULL DEFAULT now(),
    respondida_em    TIMESTAMPTZ
);

-- ---------------------------------------------------------------------
-- SCHEMA apoio — Agente A3
-- ---------------------------------------------------------------------
CREATE TABLE apoio.epi (
    id_epi            INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    nome              VARCHAR(120) NOT NULL,
    numero_ca         VARCHAR(10)  NOT NULL UNIQUE,          -- Certificado de Aprovação (NR-6)
    validade_ca       DATE         NOT NULL,
    vida_util_dias    SMALLINT     NOT NULL
);

ALTER TABLE bem_estar.treinamento
    ADD CONSTRAINT fk_treinamento_epi FOREIGN KEY (id_epi) REFERENCES apoio.epi (id_epi);

CREATE TABLE apoio.cargo_epi (
    id_cargo     INTEGER NOT NULL REFERENCES cadastro.cargo (id_cargo),
    id_epi       INTEGER NOT NULL REFERENCES apoio.epi (id_epi),
    obrigatorio  BOOLEAN NOT NULL DEFAULT TRUE,
    PRIMARY KEY (id_cargo, id_epi)
);

CREATE TABLE apoio.entrega_epi (
    id_entrega      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    id_colaborador  BIGINT  NOT NULL REFERENCES cadastro.colaborador (id_colaborador),
    id_epi          INTEGER NOT NULL REFERENCES apoio.epi (id_epi),
    data_entrega    DATE    NOT NULL,
    quantidade      SMALLINT NOT NULL DEFAULT 1 CHECK (quantidade > 0),
    registrado_em   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE apoio.solicitacao_epi (
    id_solicitacao  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    id_colaborador  BIGINT      NOT NULL REFERENCES cadastro.colaborador (id_colaborador),
    id_epi          INTEGER     NOT NULL REFERENCES apoio.epi (id_epi),
    motivo          VARCHAR(20) NOT NULL
                    CHECK (motivo IN ('AVARIA','PERDA','NAO_RECEBIDO','TAMANHO_INADEQUADO')),
    descricao       TEXT,
    status          VARCHAR(16) NOT NULL DEFAULT 'ABERTA'
                    CHECK (status IN ('ABERTA','EM_ATENDIMENTO','ATENDIDA','CANCELADA')),
    id_entrega      BIGINT      UNIQUE REFERENCES apoio.entrega_epi (id_entrega),
    aberta_em       TIMESTAMPTZ NOT NULL DEFAULT now(),
    atendida_em     TIMESTAMPTZ,
    CHECK ((status = 'ATENDIDA') = (id_entrega IS NOT NULL))
);

CREATE TABLE apoio.registro_ponto (
    id_registro     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    id_colaborador  BIGINT      NOT NULL REFERENCES cadastro.colaborador (id_colaborador),
    tipo_marcacao   VARCHAR(16) NOT NULL
                    CHECK (tipo_marcacao IN ('ENTRADA','SAIDA','INICIO_INTERVALO','FIM_INTERVALO')),
    marcado_em      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE apoio.periodo_aquisitivo (
    id_periodo      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    id_colaborador  BIGINT   NOT NULL REFERENCES cadastro.colaborador (id_colaborador),
    inicio          DATE     NOT NULL,
    fim             DATE     NOT NULL,
    dias_direito    SMALLINT NOT NULL DEFAULT 30,
    UNIQUE (id_colaborador, inicio)
);

CREATE TABLE apoio.ferias (
    id_ferias     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    id_periodo    BIGINT      NOT NULL REFERENCES apoio.periodo_aquisitivo (id_periodo),
    data_inicio   DATE        NOT NULL,
    data_fim      DATE        NOT NULL,
    data_retorno  DATE        NOT NULL,
    status        VARCHAR(12) NOT NULL DEFAULT 'PROGRAMADA'
                  CHECK (status IN ('PROGRAMADA','APROVADA','EM_GOZO','CONCLUIDA','CANCELADA')),
    CHECK (data_fim >= data_inicio AND data_retorno > data_fim)
);

CREATE TABLE apoio.tipo_alerta (
    id_tipo_alerta  SMALLINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    codigo          VARCHAR(30)  NOT NULL UNIQUE,  -- INCONSISTENCIA_AFASTAMENTO, EPIDEMIOLOGIA, FERIAS, EPI, ATESTADO_FISICO_PENDENTE
    descricao       VARCHAR(255) NOT NULL,
    perfil_destino  VARCHAR(20)  NOT NULL
);

CREATE TABLE apoio.alerta (
    id_alerta       BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    id_tipo_alerta  SMALLINT    NOT NULL REFERENCES apoio.tipo_alerta (id_tipo_alerta),
    escopo          VARCHAR(12) NOT NULL CHECK (escopo IN ('COLABORADOR','SETOR','EMPRESA')),
    id_colaborador  BIGINT      REFERENCES cadastro.colaborador (id_colaborador),
    id_setor        INTEGER     REFERENCES cadastro.setor (id_setor),
    id_execucao     UUID        REFERENCES orquestracao.execucao_agente (id_execucao),
    severidade      VARCHAR(8)  NOT NULL CHECK (severidade IN ('BAIXA','MEDIA','ALTA')),
    descricao       TEXT        NOT NULL,
    status          VARCHAR(12) NOT NULL DEFAULT 'ABERTO'
                    CHECK (status IN ('ABERTO','EM_ANALISE','RESOLVIDO','DESCARTADO')),
    gerado_em       TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolvido_em    TIMESTAMPTZ,
    resolvido_por   BIGINT      REFERENCES cadastro.usuario (id_usuario),
    CHECK ((escopo = 'COLABORADOR' AND id_colaborador IS NOT NULL)
        OR (escopo = 'SETOR' AND id_setor IS NOT NULL)
        OR (escopo = 'EMPRESA' AND id_colaborador IS NULL AND id_setor IS NULL))
);

CREATE TABLE apoio.alerta_atestado (
    id_alerta    BIGINT NOT NULL REFERENCES apoio.alerta (id_alerta),
    id_atestado  BIGINT NOT NULL REFERENCES afastamento.atestado (id_atestado),
    PRIMARY KEY (id_alerta, id_atestado)
);

-- ---------------------------------------------------------------------
-- SCHEMA rag — bases de conhecimento do A2
-- ---------------------------------------------------------------------
CREATE TABLE rag.base_conhecimento (
    id_base  SMALLINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    nome     VARCHAR(40) NOT NULL UNIQUE    -- PAUSAS, DIREITOS_BENEFICIOS, PROCEDIMENTOS, SAUDE_OCUPACIONAL, EPI
);

CREATE TABLE rag.documento (
    id_documento  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    id_base       SMALLINT     NOT NULL REFERENCES rag.base_conhecimento (id_base),
    titulo        VARCHAR(200) NOT NULL,
    fonte         VARCHAR(500) NOT NULL,
    versao        VARCHAR(20)  NOT NULL,
    vigente       BOOLEAN      NOT NULL DEFAULT TRUE,
    indexado_em   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (titulo, versao)
);

CREATE TABLE rag.trecho (
    id_trecho     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    id_documento  BIGINT      NOT NULL REFERENCES rag.documento (id_documento) ON DELETE CASCADE,
    ordem         INTEGER     NOT NULL,
    conteudo      TEXT        NOT NULL,
    embedding     vector(768) NOT NULL,
    UNIQUE (id_documento, ordem)
);

-- ---------------------------------------------------------------------
-- SCHEMA orquestracao — comunicação e logs (A0)
-- ---------------------------------------------------------------------
CREATE TABLE orquestracao.status_mensagem (
    id_status  SMALLINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    codigo     VARCHAR(20)  NOT NULL UNIQUE,   -- ENFILEIRADA, ENVIADA, ENTREGUE, LIDA, ERRO, REENVIO
    descricao  VARCHAR(120) NOT NULL
);

CREATE TABLE orquestracao.mensagem (
    id_mensagem       UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    id_destinatario   BIGINT       NOT NULL REFERENCES cadastro.colaborador (id_colaborador),
    id_canal          SMALLINT     NOT NULL REFERENCES orquestracao.canal (id_canal),
    agente_origem     VARCHAR(4)   NOT NULL CHECK (agente_origem IN ('A0','A1','A2','A3')),
    id_lembrete       BIGINT       REFERENCES bem_estar.lembrete (id_lembrete),
    id_alerta         BIGINT       REFERENCES apoio.alerta (id_alerta),
    id_atestado       BIGINT       REFERENCES afastamento.atestado (id_atestado),
    assunto           VARCHAR(200),
    conteudo          TEXT         NOT NULL,
    criado_em         TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CHECK (num_nonnulls(id_lembrete, id_alerta, id_atestado) <= 1)
);

CREATE TABLE orquestracao.log_envio (
    id_log        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    id_mensagem   UUID        NOT NULL REFERENCES orquestracao.mensagem (id_mensagem),
    id_status     SMALLINT    NOT NULL REFERENCES orquestracao.status_mensagem (id_status),
    tentativa     SMALLINT    NOT NULL DEFAULT 1,
    detalhe       TEXT,
    registrado_em TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE orquestracao.interacao_chat (
    id_interacao  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    id_usuario    BIGINT      NOT NULL REFERENCES cadastro.usuario (id_usuario),
    thread_id     VARCHAR(64) NOT NULL,
    papel         VARCHAR(10) NOT NULL CHECK (papel IN ('USUARIO','AGENTE')),
    id_execucao   UUID        REFERENCES orquestracao.execucao_agente (id_execucao),
    conteudo      TEXT        NOT NULL,
    criado_em     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE orquestracao.avaliacao_judge (
    id_avaliacao  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    id_execucao   UUID         NOT NULL REFERENCES orquestracao.execucao_agente (id_execucao),
    criterio      VARCHAR(30)  NOT NULL
                  CHECK (criterio IN ('FUNDAMENTACAO','ADERENCIA_PERFIL','SEGURANCA','TOM','CLAREZA')),
    nota          SMALLINT     NOT NULL CHECK (nota BETWEEN 1 AND 5),
    justificativa TEXT         NOT NULL,
    avaliado_em   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (id_execucao, criterio)
);

CREATE TABLE orquestracao.log_acesso (
    id_acesso    BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    id_usuario   BIGINT      REFERENCES cadastro.usuario (id_usuario),
    login_informado VARCHAR(160) NOT NULL,
    evento       VARCHAR(16) NOT NULL CHECK (evento IN ('LOGIN','LOGOUT','FALHA_LOGIN','BLOQUEIO','TOKEN_RENOVADO')),
    ip_origem    INET,
    ocorrido_em  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------
-- Índices de apoio às consultas dos agentes
-- ---------------------------------------------------------------------
CREATE INDEX ix_atestado_colab_periodo ON afastamento.atestado (id_colaborador, inicio_afastamento, fim_afastamento);
CREATE INDEX ix_atestado_cid           ON afastamento.atestado (codigo_cid, data_emissao);
CREATE INDEX ix_ponto_colab_data       ON apoio.registro_ponto (id_colaborador, marcado_em);
CREATE INDEX ix_lembrete_agenda        ON bem_estar.lembrete (status, agendado_para);
CREATE INDEX ix_alerta_status          ON apoio.alerta (status, gerado_em);
CREATE INDEX ix_solicitacao_epi_status ON apoio.solicitacao_epi (status, aberta_em);
CREATE INDEX ix_atestado_fisico_prazo  ON afastamento.atestado (prazo_entrega_fisica) WHERE recebido_fisico_em IS NULL;
CREATE INDEX ix_log_envio_msg          ON orquestracao.log_envio (id_mensagem, registrado_em);
CREATE INDEX ix_exec_thread            ON orquestracao.execucao_agente (thread_id, iniciado_em);
CREATE INDEX ix_trecho_embedding       ON rag.trecho USING hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------------
-- Visões: atributos derivados ficam fora das tabelas (3FN)
-- ---------------------------------------------------------------------
CREATE VIEW apoio.vw_saldo_ferias AS
SELECT p.id_periodo,
       p.id_colaborador,
       p.dias_direito,
       p.dias_direito - COALESCE(SUM(f.data_fim - f.data_inicio + 1)
                                 FILTER (WHERE f.status <> 'CANCELADA'), 0) AS dias_remanescentes
FROM apoio.periodo_aquisitivo p
LEFT JOIN apoio.ferias f ON f.id_periodo = p.id_periodo
GROUP BY p.id_periodo, p.id_colaborador, p.dias_direito;

CREATE VIEW apoio.vw_epi_pendente AS
SELECT c.id_colaborador, c.id_setor, ce.id_epi, e.nome AS epi,
       MAX(en.data_entrega) AS ultima_entrega,
       MAX(en.data_entrega) + e.vida_util_dias AS troca_prevista,
       e.validade_ca
FROM cadastro.colaborador c
JOIN apoio.cargo_epi ce ON ce.id_cargo = c.id_cargo AND ce.obrigatorio
JOIN apoio.epi e        ON e.id_epi = ce.id_epi
LEFT JOIN apoio.entrega_epi en ON en.id_colaborador = c.id_colaborador AND en.id_epi = ce.id_epi
WHERE c.ativo
GROUP BY c.id_colaborador, c.id_setor, ce.id_epi, e.nome, e.vida_util_dias, e.validade_ca
HAVING MAX(en.data_entrega) IS NULL
    OR MAX(en.data_entrega) + e.vida_util_dias <= CURRENT_DATE + 15
    OR e.validade_ca <= CURRENT_DATE + 30;

CREATE VIEW bem_estar.vw_proximo_exame AS
SELECT c.id_colaborador, ce.id_tipo_exame, te.nome AS exame,
       MAX(ex.data_realizacao) AS ultimo_exame,
       (MAX(ex.data_realizacao) + make_interval(months => te.periodicidade_meses))::date AS proximo_exame
FROM cadastro.colaborador c
JOIN bem_estar.cargo_exame ce ON ce.id_cargo = c.id_cargo
JOIN bem_estar.tipo_exame te  ON te.id_tipo_exame = ce.id_tipo_exame
LEFT JOIN bem_estar.exame_colaborador ex
       ON ex.id_colaborador = c.id_colaborador AND ex.id_tipo_exame = ce.id_tipo_exame
WHERE c.ativo
GROUP BY c.id_colaborador, ce.id_tipo_exame, te.nome, te.periodicidade_meses;

-- Variáveis e rótulo da rede neural de recomendação de benefícios
CREATE VIEW bem_estar.vw_features_beneficio AS
SELECT c.id_colaborador,
       b.id_beneficio,
       c.id_cargo,
       s.id_area,
       c.id_setor,
       m.sigla_uf,
       (EXTRACT(YEAR FROM age(CURRENT_DATE, c.data_admissao)) * 12
        + EXTRACT(MONTH FROM age(CURRENT_DATE, c.data_admissao)))::int        AS tempo_empresa_meses,
       EXTRACT(YEAR FROM age(CURRENT_DATE, c.data_nascimento))::int           AS idade,
       c.qtd_dependentes,
       EXTRACT(EPOCH FROM (c.jornada_fim - c.jornada_inicio)) / 3600.0        AS jornada_horas,
       (SELECT COUNT(*) FROM bem_estar.colaborador_beneficio o
         WHERE o.id_colaborador = c.id_colaborador
           AND o.id_beneficio <> b.id_beneficio
           AND (o.data_fim IS NULL OR o.data_fim >= CURRENT_DATE))            AS outros_beneficios_ativos,
       (SELECT AVG(CASE r.status WHEN 'ACEITA' THEN 1.0 WHEN 'RECUSADA' THEN 0.0 END)
          FROM bem_estar.recomendacao_beneficio r
         WHERE r.id_colaborador = c.id_colaborador)                           AS taxa_aceite,
       EXISTS (SELECT 1 FROM bem_estar.colaborador_beneficio a
                WHERE a.id_colaborador = c.id_colaborador
                  AND a.id_beneficio = b.id_beneficio)                        AS aderiu
FROM cadastro.colaborador c
CROSS JOIN bem_estar.beneficio b
JOIN cadastro.setor s     ON s.id_setor = c.id_setor
JOIN cadastro.municipio m ON m.id_municipio = c.id_municipio
WHERE c.ativo;

CREATE VIEW orquestracao.vw_status_atual_mensagem AS
SELECT DISTINCT ON (l.id_mensagem)
       l.id_mensagem, s.codigo AS status_atual, l.tentativa, l.registrado_em
FROM orquestracao.log_envio l
JOIN orquestracao.status_mensagem s ON s.id_status = l.id_status
ORDER BY l.id_mensagem, l.registrado_em DESC;

-- ---------------------------------------------------------------------
-- Logs somente inserção
-- ---------------------------------------------------------------------
REVOKE UPDATE, DELETE ON orquestracao.log_envio, orquestracao.interacao_chat,
       orquestracao.avaliacao_judge, orquestracao.log_acesso
       FROM PUBLIC;
