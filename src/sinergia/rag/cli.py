"""CLI interativa para testes manuais das ferramentas RAG do SinergIA."""

import sys
import time
from typing import Any, Callable, Dict

from sinergia.rag.tools import (
    ferramenta_busca_direitos,
    ferramenta_busca_epis,
    ferramenta_busca_pausas,
    ferramenta_busca_procedimentos,
    ferramenta_busca_saude_ocupacional,
)

FERRAMENTAS: Dict[str, Dict[str, Any]] = {
    "1": {
        "codigo": "F1",
        "id_base": 1,
        "nome": "Pausas e Ergonomia (NR-17)",
        "funcao": ferramenta_busca_pausas,
        "descricao": "Pausas, descansos, limites de digitação e ergonomia (NR-17)",
    },
    "2": {
        "codigo": "F2",
        "id_base": 2,
        "nome": "Direitos e Benefícios (CLT)",
        "funcao": ferramenta_busca_direitos,
        "descricao": "Legislação trabalhista (CLT), direitos, benefícios e faltas justificadas",
    },
    "3": {
        "codigo": "F3",
        "id_base": 3,
        "nome": "Procedimentos e Treinamentos (POPs)",
        "funcao": ferramenta_busca_procedimentos,
        "descricao": "Procedimentos operacionais padrão (POPs) e cronogramas de capacitação",
    },
    "4": {
        "codigo": "F4",
        "id_base": 4,
        "nome": "Saúde Ocupacional (PCMSO)",
        "funcao": ferramenta_busca_saude_ocupacional,
        "descricao": "Normas e diretrizes do PCMSO, exames periódicos e programas de saúde",
    },
    "5": {
        "codigo": "F5",
        "id_base": 5,
        "nome": "EPIs e Fichas de CA (NR-6)",
        "funcao": ferramenta_busca_epis,
        "descricao": "Fichas técnicas de EPIs, números de CA (NR-6), orientações de uso e substituição",
    },
}


def _executar_ferramenta(funcao: Callable[..., str], query: str) -> str:
    """Invoca a ferramenta tratando tanto objetos Tool do CrewAI quanto funções normais."""
    if hasattr(funcao, "run"):
        return funcao.run(query=query)
    return funcao(query=query)


def exibir_cabecalho() -> None:
    """Exibe o cabeçalho e menu de opções disponíveis."""
    print("\n" + "=" * 55)
    print("   🤖 SINERGIA - RAG Tools CLI (Console de Testes)")
    print("=" * 55)
    print("Escolha uma ferramenta:")
    for chave, info in sorted(FERRAMENTAS.items()):
        print(f"  [{chave}] {info['codigo']} - {info['nome']}")
    print("  [0] Sair")
    print("=" * 55)


def modo_interativo() -> None:
    """Executa o loop interativo da CLI de testes."""
    exibir_cabecalho()
    ferramenta_atual = None

    while True:
        try:
            prompt_texto = (
                f"\nOpção (1-5, 'm' menu, '0' sair) [{ferramenta_atual or '1'}]: "
                if ferramenta_atual
                else "\nOpção: "
            )
            entrada = input(prompt_texto).strip()

            # Usar padrão anterior se pressionar Enter vazio
            if not entrada:
                if ferramenta_atual:
                    entrada = ferramenta_atual
                else:
                    continue

            # Comandos de saída
            if entrada in ("0", "q", "quit", "exit", "sair"):
                print("\nEncerrando CLI de testes do RAG. Até logo!\n")
                break

            # Reexibir menu
            if entrada.lower() in ("m", "menu", "help", "?"):
                exibir_cabecalho()
                continue

            if entrada not in FERRAMENTAS:
                print("⚠️ Opção inválida. Digite um número de 1 a 5, 'm' para menu ou '0' para sair.")
                continue

            ferramenta_atual = entrada
            info = FERRAMENTAS[ferramenta_atual]

            print(f"\n📌 Selecionada: [{info['codigo']}] {info['nome']}")
            print(f"   ({info['descricao']})")

            query = input("Pergunta / Consulta (ou Enter para voltar): ").strip()
            if not query:
                print("↩️  Retornando à seleção de opções...")
                continue

            print(f"\n[Buscando na Base {info['id_base']}...]")
            inicio = time.perf_counter()

            try:
                resultado = _executar_ferramenta(info["funcao"], query)
                duracao = time.perf_counter() - inicio

                print(f"\n>>> Resultado (em {duracao:.2f}s):")
                print("-" * 55)
                print(resultado)
                print("-" * 55)
            except Exception as erro:
                print(f"❌ Erro ao consultar a base: {erro}")

        except (KeyboardInterrupt, EOFError):
            print("\n\nOperação cancelada pelo usuário. Até logo!\n")
            break


def main() -> None:
    """Ponto de entrada da CLI."""
    modo_interativo()


if __name__ == "__main__":
    main()

