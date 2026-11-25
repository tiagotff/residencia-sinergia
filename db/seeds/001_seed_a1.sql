-- Dados canonicos minimos para desenvolvimento e testes do A1.
-- Seguro para executar mais de uma vez.

INSERT INTO cadastro.uf (sigla_uf, nome)
VALUES ('SP', 'São Paulo'), ('RJ', 'Rio de Janeiro')
ON CONFLICT (sigla_uf) DO NOTHING;

INSERT INTO cadastro.municipio (id_municipio, nome, sigla_uf)
VALUES
    (3509502, 'Campinas', 'SP'),
    (3550308, 'São Paulo', 'SP'),
    (3304557, 'Rio de Janeiro', 'RJ')
ON CONFLICT (id_municipio) DO UPDATE
SET nome = EXCLUDED.nome, sigla_uf = EXCLUDED.sigla_uf;

INSERT INTO cadastro.area (id_area, nome)
OVERRIDING SYSTEM VALUE
VALUES (2, 'Operações')
ON CONFLICT (nome) DO NOTHING;

INSERT INTO cadastro.setor (id_setor, id_area, nome)
OVERRIDING SYSTEM VALUE
VALUES (7, 2, 'Logística')
ON CONFLICT (id_setor) DO UPDATE
SET id_area = EXCLUDED.id_area, nome = EXCLUDED.nome;

INSERT INTO cadastro.cargo (id_cargo, nome, cbo)
OVERRIDING SYSTEM VALUE
VALUES (15, 'Analista de Operações', '1421')
ON CONFLICT (id_cargo) DO UPDATE
SET nome = EXCLUDED.nome, cbo = EXCLUDED.cbo;

INSERT INTO cadastro.colaborador (
    id_colaborador, matricula, nome, email, id_cargo, id_setor,
    id_municipio, id_gestor, data_admissao, data_nascimento,
    qtd_dependentes, jornada_inicio, jornada_fim, ativo
)
OVERRIDING SYSTEM VALUE
VALUES (
    42, '00123', 'Carla Souza', 'carla.souza@sinergia.local',
    15, 7, 3509502, NULL, DATE '2024-03-01', DATE '1995-06-15',
    0, TIME '08:00', TIME '17:00', TRUE
)
ON CONFLICT (id_colaborador) DO UPDATE
SET matricula = EXCLUDED.matricula,
    nome = EXCLUDED.nome,
    email = EXCLUDED.email,
    id_cargo = EXCLUDED.id_cargo,
    id_setor = EXCLUDED.id_setor,
    id_municipio = EXCLUDED.id_municipio,
    data_admissao = EXCLUDED.data_admissao,
    data_nascimento = EXCLUDED.data_nascimento,
    ativo = EXCLUDED.ativo;

INSERT INTO cadastro.colaborador (
    id_colaborador, matricula, nome, email, id_cargo, id_setor,
    id_municipio, id_gestor, data_admissao, data_nascimento,
    qtd_dependentes, jornada_inicio, jornada_fim, ativo
)
OVERRIDING SYSTEM VALUE
VALUES (
    7, 'RH0007', 'Ana RH', 'rh.ana@sinergia.local',
    15, 7, 3509502, NULL, DATE '2023-01-10', DATE '1990-04-20',
    0, TIME '08:00', TIME '17:00', TRUE
)
ON CONFLICT (id_colaborador) DO UPDATE
SET matricula = EXCLUDED.matricula,
    nome = EXCLUDED.nome,
    email = EXCLUDED.email,
    id_cargo = EXCLUDED.id_cargo,
    id_setor = EXCLUDED.id_setor,
    id_municipio = EXCLUDED.id_municipio,
    data_admissao = EXCLUDED.data_admissao,
    data_nascimento = EXCLUDED.data_nascimento,
    ativo = EXCLUDED.ativo;

INSERT INTO cadastro.usuario (
    id_usuario, id_colaborador, login, senha_hash, perfil, ativo
)
OVERRIDING SYSTEM VALUE
VALUES (
    42, 42, 'carla.souza',
    '$argon2id$seed-nao-utilizavel-para-autenticacao', 'COLABORADOR', TRUE
)
ON CONFLICT (id_usuario) DO UPDATE
SET id_colaborador = EXCLUDED.id_colaborador,
    login = EXCLUDED.login,
    perfil = EXCLUDED.perfil,
    ativo = EXCLUDED.ativo;

INSERT INTO cadastro.usuario (
    id_usuario, id_colaborador, login, senha_hash, perfil, ativo
)
OVERRIDING SYSTEM VALUE
VALUES (
    7, 7, 'rh.ana',
    '$argon2id$seed-nao-utilizavel-para-autenticacao', 'RH', TRUE
)
ON CONFLICT (id_usuario) DO UPDATE
SET id_colaborador = EXCLUDED.id_colaborador,
    login = EXCLUDED.login,
    perfil = EXCLUDED.perfil,
    ativo = EXCLUDED.ativo;

INSERT INTO afastamento.conselho_profissional (id_conselho, sigla, descricao)
OVERRIDING SYSTEM VALUE
VALUES
    (1, 'CRM', 'Conselho Regional de Medicina'),
    (2, 'CRO', 'Conselho Regional de Odontologia'),
    (3, 'CRP', 'Conselho Regional de Psicologia')
ON CONFLICT (id_conselho) DO UPDATE
SET sigla = EXCLUDED.sigla, descricao = EXCLUDED.descricao;

INSERT INTO afastamento.cid (codigo_cid, nome_doenca, descricao, capitulo)
VALUES
    ('M54.5', 'Dor lombar baixa', 'Dor localizada na região lombar inferior (lombalgia não especificada).', 'M'),
    ('F41.1', 'Ansiedade generalizada', 'Transtorno caracterizado por ansiedade e preocupação persistentes e excessivas.', 'F'),
    ('J06.9', 'Infecção aguda das vias aéreas superiores', 'Infecção aguda das vias aéreas superiores não especificada (resfriado comum / IVAS).', 'J'),
    ('Z00.0', 'Exame médico geral', 'Consulta médica para exame clínico geral ou rastreamento sem queixas específicas.', 'Z')
ON CONFLICT (codigo_cid) DO UPDATE
SET nome_doenca = EXCLUDED.nome_doenca,
    descricao = EXCLUDED.descricao,
    capitulo = EXCLUDED.capitulo;
