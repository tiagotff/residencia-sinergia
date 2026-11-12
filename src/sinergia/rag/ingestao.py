"""CLI e pipeline de ingestão atômica e versionamento de documentos para o subsistema RAG."""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

from sinergia.rag.chunking import chunking_documento_pdf
from sinergia.rag.embeddings import GeradorEmbeddings, get_gerador_embeddings

MAPA_MOCKS_PADRAO = {
    "base1_nr17_ergonomia.pdf": {
        "id_base": 1,
        "titulo": "NR-17 Ergonomia",
        "versao": "2024.1",
        "fonte": "Portaria MTP nº 423 / Ministério do Trabalho e Emprego",
    },
    "base2_clt_direitos_beneficios.pdf": {
        "id_base": 2,
        "titulo": "CLT - Consolidação das Leis do Trabalho",
        "versao": "2024.1",
        "fonte": "Decreto-Lei nº 5.452 / Presidência da República",
    },
    "base3_pop_procedimentos_treinamentos.pdf": {
        "id_base": 3,
        "titulo": "POP - Procedimentos e Treinamentos Corporativos",
        "versao": "1.0",
        "fonte": "Diretoria de Operações e Recursos Humanos",
    },
    "base4_pcmso_saude_ocupacional.pdf": {
        "id_base": 4,
        "titulo": "PCMSO - Programa de Controle Médico de Saúde Ocupacional",
        "versao": "2024",
        "fonte": "NR-7 / Coordenação de Medicina do Trabalho",
    },
    "base5_epi_fichas_ca.pdf": {
        "id_base": 5,
        "titulo": "Fichas Técnicas de EPIs e Catálogo de CA",
        "versao": "2024",
        "fonte": "NR-6 / SESMT - Segurança do Trabalho",
    },
}


def mapear_mock_para_metadados(nome_arquivo: str) -> Dict[str, Any]:
    """Retorna os metadados padrão associados a um arquivo de mock conhecido."""
    base_nome = os.path.basename(nome_arquivo)
    if base_nome in MAPA_MOCKS_PADRAO:
        return MAPA_MOCKS_PADRAO[base_nome]

    # Inferência por padrão no nome
    if "base1" in base_nome or "nr17" in base_nome:
        return {"id_base": 1, "titulo": "NR-17 Ergonomia", "versao": "1.0", "fonte": "NR-17"}
    elif "base2" in base_nome or "clt" in base_nome:
        return {"id_base": 2, "titulo": "CLT", "versao": "1.0", "fonte": "CLT"}
    elif "base3" in base_nome or "pop" in base_nome:
        return {"id_base": 3, "titulo": "POP Treinamentos", "versao": "1.0", "fonte": "POP"}
    elif "base4" in base_nome or "pcmso" in base_nome:
        return {"id_base": 4, "titulo": "PCMSO", "versao": "1.0", "fonte": "PCMSO"}
    elif "base5" in base_nome or "epi" in base_nome:
        return {"id_base": 5, "titulo": "Catálogo EPI", "versao": "1.0", "fonte": "EPI"}

    return {"id_base": 1, "titulo": base_nome, "versao": "1.0", "fonte": base_nome}


def ingerir_documento(
    caminho_pdf: str,
    titulo: str,
    id_base: int,
    versao: str,
    fonte: Optional[str] = None,
    database_url: Optional[str] = None,
    gerador_embeddings: Optional[GeradorEmbeddings] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Executa a ingestão atômica de um PDF: particiona, vetoriza e persiste em transação."""
    if not os.path.exists(caminho_pdf):
        raise FileNotFoundError(f"Arquivo PDF não encontrado: {caminho_pdf}")

    fonte = fonte or os.path.basename(caminho_pdf)
    gerador = gerador_embeddings or get_gerador_embeddings()

    # 1. Chunking híbrido com metadados estruturais
    chunks = chunking_documento_pdf(caminho_pdf=caminho_pdf, titulo_documento=titulo)
    if not chunks:
        return {
            "status": "aviso",
            "mensagem": "Nenhum chunk extraído do documento",
            "chunks_ingeridos": 0,
        }

    # 2. Geração dos embeddings densos (768 dimensões)
    textos = [c.conteudo for c in chunks]
    embeddings = gerador.gerar_embeddings(textos)

    if dry_run:
        return {
            "status": "sucesso",
            "modo": "dry-run",
            "titulo": titulo,
            "versao": versao,
            "id_base": id_base,
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
            cur.execute(sql_invalida, (id_base, titulo))

            # 3.2 Inserir novo documento ativo (vigente = true)
            sql_insere_doc = """
                INSERT INTO rag.documento (id_base, titulo, fonte, versao, vigente)
                VALUES (%s, %s, %s, %s, true)
                RETURNING id_documento;
            """
            cur.execute(sql_insere_doc, (id_base, titulo, fonte, versao))
            id_documento = cur.fetchone()[0]

            # 3.3 Inserção em lote (bulk insert) dos trechos e embeddings
            sql_insere_trecho = """
                INSERT INTO rag.trecho (id_documento, ordem, conteudo, embedding)
                VALUES (%s, %s, %s, %s::vector);
            """
            for chunk, emb in zip(chunks, embeddings):
                vetor_str = "[" + ",".join(str(x) for x in emb) + "]"
                cur.execute(sql_insere_trecho, (id_documento, chunk.ordem, chunk.conteudo, vetor_str))

        conn.commit()

    return {
        "status": "sucesso",
        "id_documento": id_documento,
        "titulo": titulo,
        "versao": versao,
        "id_base": id_base,
        "chunks_ingeridos": len(chunks),
    }


def ingerir_diretorio_mocks(
    diretorio: str,
    database_url: Optional[str] = None,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Ingere todos os arquivos PDF de um diretório utilizando o mapa de metadados padrão."""
    if not os.path.exists(diretorio):
        raise FileNotFoundError(f"Diretório não encontrado: {diretorio}")

    resultados = []
    arquivos = sorted([f for f in os.listdir(diretorio) if f.endswith(".pdf")])

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
    parser.add_argument("--versao", type=str, default="1.0", help="Versão do documento")
    parser.add_argument("--fonte", type=str, help="Fonte ou ato normativo")
    parser.add_argument("--dry-run", action="store_true", help="Executa sem persistir no banco de dados")

    args = parser.parse_args()

    if args.diretorio:
        print(f"Iniciando ingestão do diretório: {args.diretorio}")
        resultados = ingerir_diretorio_mocks(args.diretorio, dry_run=args.dry_run)
        print(f"Ingestão concluída. Total de documentos: {len(resultados)}")
        for r in resultados:
            print(f" - {r.get('titulo')}: {r.get('chunks_ingeridos')} chunks ({r.get('status')})")
    elif args.arquivo:
        if not args.id_base or not args.titulo:
            meta = mapear_mock_para_metadados(args.arquivo)
            id_base = args.id_base or meta["id_base"]
            titulo = args.titulo or meta["titulo"]
            versao = args.versao if args.versao != "1.0" else meta["versao"]
            fonte = args.fonte or meta["fonte"]
        else:
            id_base = args.id_base
            titulo = args.titulo
            versao = args.versao
            fonte = args.fonte

        print(f"Ingerindo documento: {args.arquivo} (Base: {id_base}, Título: {titulo})")
        res = ingerir_documento(
            caminho_pdf=args.arquivo,
            titulo=titulo,
            id_base=id_base,
            versao=versao,
            fonte=fonte,
            dry_run=args.dry_run,
        )
        print(f"Resultado: {res}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
