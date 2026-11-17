"""CLI e pipeline de ingestão atômica e versionamento de documentos para o subsistema RAG."""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from sinergia.rag.chunking import (
    MAX_CHARS_PADRAO,
    OVERLAP_PADRAO,
    chunking_documento_pdf,
    extrair_metadados_cabecalho_pdf,
)
from sinergia.rag.embeddings import GeradorEmbeddings, get_gerador_embeddings

MAPA_MOCKS_PADRAO: Dict[str, Dict[str, Any]] = {
    "rag_ergonomia.pdf": {
        "id_base": 1,
        "codigo": "KB-RH-ERG-001",
        "titulo": "Base de Conhecimento RH: Pausas e Ergonomia",
        "versao": "2.0",
        "fonte": "Recursos Humanos e Segurança do Trabalho (RH/TST) — SinergIA; NR-17 (Ergonomia)",
    },
    "rag_trabalho.pdf": {
        "id_base": 2,
        "codigo": "KB-RH-BEN-002",
        "titulo": "Base de Conhecimento RH: Benefícios e Direitos Trabalhistas",
        "versao": "1.0",
        "fonte": "Recursos Humanos (RH) — SinergIA; CLT (Decreto-Lei nº 5.452/1943), Convenção Coletiva e Políticas Internas",
    },
    "rag_pcmso_saude_ocupacional.pdf": {
        "id_base": 4,
        "codigo": "KB-SO-004",
        "titulo": "Base de Conhecimento Saúde Ocupacional",
        "versao": "1.0",
        "fonte": "Serviço de Saúde Ocupacional (médico do trabalho e SESMT) — SinergIA; NR-7 (PCMSO)",
    },
    "rag_epi.pdf": {
        "id_base": 5,
        "codigo": "KB-EPI-005",
        "titulo": "Base de Conhecimento EPI",
        "versao": "1.2",
        "fonte": "Segurança do Trabalho (TST) — SinergIA; NR-6 (Portaria MTP nº 2.175/2022 e ajustes), NR-1, CLT e Súmula 289 do TST",
    },
}


def mapear_mock_para_metadados(nome_arquivo: str) -> Dict[str, Any]:
    """Retorna os metadados padrão associados a um arquivo PDF de Base de Conhecimento."""
    base_nome = os.path.basename(nome_arquivo)
    if base_nome in MAPA_MOCKS_PADRAO:
        return dict(MAPA_MOCKS_PADRAO[base_nome])

    nome_lower = base_nome.lower()
    if any(p in nome_lower for p in ("base1", "nr17", "ergonomia")):
        return dict(MAPA_MOCKS_PADRAO["rag_ergonomia.pdf"])
    if any(p in nome_lower for p in ("base2", "clt", "trabalho", "beneficio")):
        return dict(MAPA_MOCKS_PADRAO["rag_trabalho.pdf"])
    if any(p in nome_lower for p in ("base3", "pop", "procedimento", "treinamento")):
        return {
            "id_base": 3,
            "codigo": "KB-POP-003",
            "titulo": "POP - Procedimentos e Treinamentos Corporativos",
            "versao": "1.0",
            "fonte": "Diretoria de Operações e Recursos Humanos",
        }
    if any(p in nome_lower for p in ("base4", "pcmso", "saude")):
        return dict(MAPA_MOCKS_PADRAO["rag_pcmso_saude_ocupacional.pdf"])
    if any(p in nome_lower for p in ("base5", "epi")):
        return dict(MAPA_MOCKS_PADRAO["rag_epi.pdf"])

    return {"id_base": 1, "titulo": base_nome, "versao": "1.0", "fonte": base_nome}


def _texto_para_embedding(chunk: Any) -> str:
    """Constrói o texto limpo para vetorização sem poluição de paginação."""
    meta = getattr(chunk, "metadados", {}) or {}
    if meta.get("id_pergunta"):
        secao_macro = (meta.get("secoes_map") or {}).get("h1")
        if secao_macro:
            return f"[{secao_macro}]\n{chunk.conteudo}"
        return chunk.conteudo
    return f"[{chunk.breadcrumb}]\n{chunk.conteudo}" if chunk.breadcrumb else chunk.conteudo


def ingerir_documento(
    caminho_pdf: str,
    titulo: str = "",
    id_base: int = 1,
    versao: Optional[str] = None,
    fonte: Optional[str] = None,
    database_url: Optional[str] = None,
    gerador_embeddings: Optional[GeradorEmbeddings] = None,
    dry_run: bool = False,
    max_chars: int = MAX_CHARS_PADRAO,
    overlap: int = OVERLAP_PADRAO,
) -> Dict[str, Any]:
    """Executa a ingestão atômica de um PDF no layout KB: particiona, vetoriza e persiste em transação."""
    if not os.path.exists(caminho_pdf):
        raise FileNotFoundError(f"Arquivo PDF não encontrado: {caminho_pdf}")

    cabecalho = extrair_metadados_cabecalho_pdf(caminho_pdf)
    titulo_efetivo = titulo.strip() or (cabecalho.titulo if cabecalho and cabecalho.titulo else os.path.basename(caminho_pdf))
    versao_efetiva = versao or (cabecalho.versao if cabecalho and cabecalho.versao else "1.0")
    fonte_efetiva = fonte or (cabecalho.fonte if cabecalho and cabecalho.fonte else os.path.basename(caminho_pdf))

    gerador = gerador_embeddings or get_gerador_embeddings()

    # 1. Chunking estrutural no layout KB com limite de caracteres e overlap
    chunks = chunking_documento_pdf(
        caminho_pdf=caminho_pdf,
        titulo_documento=titulo_efetivo,
        max_chars=max_chars,
        overlap=overlap,
    )
    if not chunks:
        return {
            "status": "aviso",
            "mensagem": "Nenhum chunk extraído do documento",
            "chunks_ingeridos": 0,
        }

    # 2. Geração dos embeddings densos (768 dimensões) com textos já sanitizados
    textos_embedding = [_texto_para_embedding(c) for c in chunks]
    embeddings = gerador.gerar_embeddings(textos_embedding)

    if dry_run:
        return {
            "status": "sucesso",
            "modo": "dry-run",
            "titulo": titulo_efetivo,
            "versao": versao_efetiva,
            "id_base": id_base,
            "codigo": cabecalho.codigo if cabecalho else None,
            "max_chars": max_chars,
            "overlap": overlap,
            "chunks_ingeridos": len(chunks),
        }

    # 3. Transação atômica no banco de dados
    db_url = database_url or os.getenv(
        "DATABASE_URL", "postgresql://sinergia:sinergia@localhost:5432/sinergia"
    )

    try:
        import psycopg
    except ImportError:
        psycopg = None

    if psycopg is None:
        raise RuntimeError("Driver 'psycopg' não está instalado no ambiente.")

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            # 3.1 Invalidar versão anterior do mesmo título e base (vigente = false)
            sql_invalida = """
                UPDATE rag.documento
                SET vigente = false
                WHERE id_base = %s AND titulo = %s AND vigente = true;
            """
            cur.execute(sql_invalida, (id_base, titulo_efetivo))

            # 3.2 Inserir novo documento ativo (vigente = true)
            sql_insere_doc = """
                INSERT INTO rag.documento (id_base, titulo, fonte, versao, vigente)
                VALUES (%s, %s, %s, %s, true)
                RETURNING id_documento;
            """
            cur.execute(sql_insere_doc, (id_base, titulo_efetivo, fonte_efetiva, versao_efetiva))
            id_documento = cur.fetchone()[0]

            # 3.3 Inserção em lote (bulk insert) dos trechos, embeddings e metadados JSONB
            sql_insere_trecho = """
                INSERT INTO rag.trecho (id_documento, ordem, conteudo, embedding, metadados)
                VALUES (%s, %s, %s, %s::vector, %s::jsonb);
            """
            for chunk, emb in zip(chunks, embeddings):
                vetor_str = "[" + ",".join(str(x) for x in emb) + "]"
                meta_json = json.dumps(chunk.metadados, ensure_ascii=False)
                cur.execute(
                    sql_insere_trecho,
                    (id_documento, chunk.ordem, chunk.conteudo, vetor_str, meta_json),
                )

        conn.commit()

    return {
        "status": "sucesso",
        "id_documento": id_documento,
        "titulo": titulo_efetivo,
        "versao": versao_efetiva,
        "id_base": id_base,
        "chunks_ingeridos": len(chunks),
    }


def ingerir_diretorio_mocks(
    diretorio: str,
    database_url: Optional[str] = None,
    dry_run: bool = False,
    max_chars: int = MAX_CHARS_PADRAO,
    overlap: int = OVERLAP_PADRAO,
) -> List[Dict[str, Any]]:
    """Ingere todos os arquivos PDF de um diretório utilizando o mapa de metadados padrão."""
    if not os.path.exists(diretorio):
        raise FileNotFoundError(f"Diretório não encontrado: {diretorio}")

    resultados = []
    arquivos = sorted([f for f in os.listdir(diretorio) if f.lower().endswith(".pdf")])

    for arq in arquivos:
        caminho_completo = os.path.join(diretorio, arq)
        meta = mapear_mock_para_metadados(arq)
        res = ingerir_documento(
            caminho_pdf=caminho_completo,
            titulo=meta["titulo"],
            id_base=meta["id_base"],
            versao=meta["versao"],
            fonte=meta["fonte"],
            database_url=database_url,
            dry_run=dry_run,
            max_chars=max_chars,
            overlap=overlap,
        )
        resultados.append(res)

    return resultados


def main():
    """Ponto de entrada CLI para ingestão de documentos RAG."""
    parser = argparse.ArgumentParser(description="CLI de Ingestão do Subsistema RAG SinergIA")
    parser.add_argument("--arquivo", type=str, help="Caminho para um arquivo PDF individual")
    parser.add_argument("--diretorio", type=str, help="Caminho para diretório com arquivos PDF")
    parser.add_argument("--id-base", type=int, help="ID da base de conhecimento (1 a 5)")
    parser.add_argument("--titulo", type=str, help="Título do documento")
    parser.add_argument("--versao", type=str, default=None, help="Versão do documento")
    parser.add_argument("--fonte", type=str, help="Fonte ou ato normativo")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=MAX_CHARS_PADRAO,
        help=f"Limite máximo de caracteres por chunk (padrão: {MAX_CHARS_PADRAO})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=OVERLAP_PADRAO,
        help=f"Sobreposição em caracteres quando um bloco excede max-chars (padrão: {OVERLAP_PADRAO})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Executa sem persistir no banco de dados")

    args = parser.parse_args()

    if args.diretorio:
        print(f"Iniciando ingestão do diretório: {args.diretorio}")
        resultados = ingerir_diretorio_mocks(
            args.diretorio,
            dry_run=args.dry_run,
            max_chars=args.max_chars,
            overlap=args.overlap,
        )
        print(f"Ingestão concluída. Total de documentos: {len(resultados)}")
        for r in resultados:
            print(f" - {r.get('titulo')}: {r.get('chunks_ingeridos')} chunks ({r.get('status')})")
    elif args.arquivo:
        meta = mapear_mock_para_metadados(args.arquivo)
        id_base = args.id_base or meta["id_base"]
        titulo = args.titulo or ""
        versao = args.versao
        fonte = args.fonte

        print(f"Ingerindo documento: {args.arquivo} (Base: {id_base})")
        res = ingerir_documento(
            caminho_pdf=args.arquivo,
            titulo=titulo,
            id_base=id_base,
            versao=versao,
            fonte=fonte,
            dry_run=args.dry_run,
            max_chars=args.max_chars,
            overlap=args.overlap,
        )
        print(f"Resultado: {res}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
