-- Seed idempotente com as 27 Unidades Federativas do Brasil e principais capitais.
-- Permite que o OCR e o formulário manual aceitem registros CRM/CRO/CRP de qualquer estado.

INSERT INTO cadastro.uf (sigla_uf, nome)
VALUES
    ('AC', 'Acre'),
    ('AL', 'Alagoas'),
    ('AP', 'Amapá'),
    ('AM', 'Amazonas'),
    ('BA', 'Bahia'),
    ('CE', 'Ceará'),
    ('DF', 'Distrito Federal'),
    ('ES', 'Espírito Santo'),
    ('GO', 'Goiás'),
    ('MA', 'Maranhão'),
    ('MT', 'Mato Grosso'),
    ('MS', 'Mato Grosso do Sul'),
    ('MG', 'Minas Gerais'),
    ('PA', 'Pará'),
    ('PB', 'Paraíba'),
    ('PR', 'Paraná'),
    ('PE', 'Pernambuco'),
    ('PI', 'Piauí'),
    ('RJ', 'Rio de Janeiro'),
    ('RN', 'Rio Grande do Norte'),
    ('RS', 'Rio Grande do Sul'),
    ('RO', 'Rondônia'),
    ('RR', 'Roraima'),
    ('SC', 'Santa Catarina'),
    ('SP', 'São Paulo'),
    ('SE', 'Sergipe'),
    ('TO', 'Tocantins')
ON CONFLICT (sigla_uf) DO NOTHING;

INSERT INTO cadastro.municipio (id_municipio, nome, sigla_uf)
VALUES
    (1200401, 'Rio Branco', 'AC'),
    (2704302, 'Maceió', 'AL'),
    (1600303, 'Macapá', 'AP'),
    (1302603, 'Manaus', 'AM'),
    (2927408, 'Salvador', 'BA'),
    (2304400, 'Fortaleza', 'CE'),
    (5300108, 'Brasília', 'DF'),
    (3205309, 'Vitória', 'ES'),
    (5208707, 'Goiânia', 'GO'),
    (2111300, 'São Luís', 'MA'),
    (5103403, 'Cuiabá', 'MT'),
    (5002704, 'Campo Grande', 'MS'),
    (3106200, 'Belo Horizonte', 'MG'),
    (1501402, 'Belém', 'PA'),
    (2507507, 'João Pessoa', 'PB'),
    (4106902, 'Curitiba', 'PR'),
    (2611606, 'Recife', 'PE'),
    (2211001, 'Teresina', 'PI'),
    (3304557, 'Rio de Janeiro', 'RJ'),
    (2408102, 'Natal', 'RN'),
    (4314902, 'Porto Alegre', 'RS'),
    (1100205, 'Porto Velho', 'RO'),
    (1400100, 'Boa Vista', 'RR'),
    (4205407, 'Florianópolis', 'SC'),
    (3550308, 'São Paulo', 'SP'),
    (3509502, 'Campinas', 'SP'),
    (2800308, 'Aracaju', 'SE'),
    (1721000, 'Palmas', 'TO')
ON CONFLICT (id_municipio) DO UPDATE
SET nome = EXCLUDED.nome, sigla_uf = EXCLUDED.sigla_uf;
