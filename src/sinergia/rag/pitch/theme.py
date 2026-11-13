"""Configuração visual e helpers HTML em light theme monocromático para o pitch Streamlit."""

import html

import streamlit as st

PALETA_MONOCROMATICA: dict[str, str] = {
    "bg_principal": "#FFFFFF",
    "bg_secundario": "#FAFAFA",
    "bg_superficie": "#F4F4F5",
    "borda_sutil": "#E4E4E7",
    "texto_secundario": "#71717A",
    "texto_principal": "#09090B",
    "destaque_primario": "#18181B",
}


def obter_css_monocromatico_light() -> str:
    """Retorna bloco <style> forçando light theme monocromático e ocultando a sidebar."""
    p = PALETA_MONOCROMATICA
    return f"""
<style>
  :root {{
    color-scheme: light !important;
    --sinergia-bg: {p["bg_principal"]};
    --sinergia-panel: {p["bg_secundario"]};
    --sinergia-surface: {p["bg_superficie"]};
    --sinergia-border: {p["borda_sutil"]};
    --sinergia-muted: {p["texto_secundario"]};
    --sinergia-fg: {p["texto_principal"]};
    --sinergia-primary: {p["destaque_primario"]};
  }}

  /* Ocultar sidebar, controles laterais e barra decorativa superior */
  [data-testid="stSidebar"],
  [data-testid="stSidebarCollapsedControl"],
  [data-testid="collapsedControl"],
  [data-testid="stDecoration"] {{
    display: none !important;
  }}

  /* Forçar fundo claro monocromático */
  .stApp,
  [data-testid="stAppViewContainer"],
  [data-testid="stHeader"] {{
    background-color: {p["bg_principal"]} !important;
    color: {p["texto_principal"]} !important;
  }}

  /* Cartões e componentes customizados */
  .sinergia-header-title {{
    font-size: 1.35rem;
    font-weight: 700;
    color: {p["texto_principal"]};
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.25;
  }}

  .sinergia-header-subtitle {{
    font-size: 0.85rem;
    color: {p["texto_secundario"]};
    margin: 0.25rem 0 0 0;
  }}

  .sinergia-badge-base {{
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.2rem 0.65rem;
    border-radius: 9999px;
    background-color: {p["bg_superficie"]};
    border: 1px solid {p["borda_sutil"]};
    color: {p["destaque_primario"]};
    font-size: 0.76rem;
    font-weight: 600;
    line-height: 1.3;
  }}

  .sinergia-badge-base--ativa {{
    background-color: {p["destaque_primario"]};
    border-color: {p["destaque_primario"]};
    color: {p["bg_principal"]};
  }}

  .sinergia-kpi-card {{
    background-color: {p["bg_secundario"]};
    border: 1px solid {p["borda_sutil"]};
    border-radius: 0.5rem;
    padding: 0.75rem 0.9rem;
    transition: border-color 150ms ease;
  }}

  .sinergia-kpi-card:hover {{
    border-color: {p["texto_secundario"]};
  }}

  .sinergia-kpi-label {{
    font-size: 0.75rem;
    color: {p["texto_secundario"]};
    font-weight: 500;
  }}

  .sinergia-kpi-value {{
    font-size: 1.2rem;
    font-weight: 700;
    color: {p["texto_principal"]};
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.15rem;
  }}

  .sinergia-kpi-sub {{
    font-size: 0.72rem;
    color: {p["texto_secundario"]};
    margin-top: 0.1rem;
  }}

  .sinergia-dist-row {{
    margin-bottom: 0.6rem;
  }}

  .sinergia-dist-head {{
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: {p["texto_principal"]};
    margin-bottom: 0.25rem;
  }}

  .sinergia-dist-count {{
    font-family: 'JetBrains Mono', monospace;
    color: {p["texto_secundario"]};
  }}

  .sinergia-dist-track {{
    width: 100%;
    height: 0.45rem;
    border-radius: 9999px;
    background-color: {p["bg_superficie"]};
    border: 1px solid {p["borda_sutil"]};
    overflow: hidden;
  }}

  .sinergia-dist-fill {{
    height: 100%;
    background-color: {p["destaque_primario"]};
    border-radius: 9999px;
  }}

  .sinergia-citations-wrap {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin-top: 0.55rem;
    padding-top: 0.5rem;
    border-top: 1px solid {p["borda_sutil"]};
  }}

  .sinergia-citation-pill {{
    display: inline-block;
    font-size: 0.73rem;
    padding: 0.15rem 0.5rem;
    border-radius: 0.375rem;
    background-color: {p["bg_superficie"]};
    border: 1px solid {p["borda_sutil"]};
    color: {p["texto_secundario"]};
    font-family: 'JetBrains Mono', monospace;
  }}

  .sinergia-emb-bars {{
    display: flex;
    gap: 0.35rem;
    align-items: flex-end;
    padding: 0.6rem;
    border-radius: 0.5rem;
    background-color: {p["bg_secundario"]};
    border: 1px solid {p["borda_sutil"]};
  }}

  .sinergia-emb-col {{
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.25rem;
  }}

  .sinergia-emb-track {{
    width: 100%;
    height: 3.75rem;
    display: flex;
    align-items: flex-end;
    background-color: {p["bg_superficie"]};
    border-radius: 0.25rem;
    overflow: hidden;
  }}

  .sinergia-emb-bar-pos {{
    width: 100%;
    background-color: {p["destaque_primario"]};
  }}

  .sinergia-emb-bar-neg {{
    width: 100%;
    background-color: {p["texto_secundario"]};
  }}

  .sinergia-emb-idx {{
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    color: {p["texto_secundario"]};
  }}

  @media (prefers-reduced-motion: reduce) {{
    * {{
      transition: none !important;
      animation: none !important;
    }}
  }}
</style>
"""


def aplicar_tema_monocromatico() -> None:
    """Injeta as regras CSS monocromáticas no app Streamlit via st.html."""
    st.html(obter_css_monocromatico_light())


def renderizar_badge_base(codigo: str, nome: str, ativo: bool = False) -> str:
    """Gera o HTML do indicativo visual (badge) da base consultada (F1..F5)."""
    classe = "sinergia-badge-base sinergia-badge-base--ativa" if ativo else "sinergia-badge-base"
    texto = f"Base {html.escape(codigo)} · {html.escape(nome)}"
    return f'<span class="{classe}">{texto}</span>'


def renderizar_card_kpi(titulo: str, valor: str, subtitulo: str = "") -> str:
    """Gera o HTML de um cartão KPI monocromático."""
    sub_html = f'<div class="sinergia-kpi-sub">{html.escape(subtitulo)}</div>' if subtitulo else ""
    return (
        '<div class="sinergia-kpi-card">'
        f'<div class="sinergia-kpi-label">{html.escape(titulo)}</div>'
        f'<div class="sinergia-kpi-value">{html.escape(valor)}</div>'
        f"{sub_html}"
        "</div>"
    )


def renderizar_barra_distribuicao(
    codigo: str, nome: str, contagem: int, percentual: float
) -> str:
    """Gera o HTML da barra monocromática de acionamento por base (F1..F5)."""
    pct_clamped = max(0.0, min(100.0, float(percentual)))
    return (
        '<div class="sinergia-dist-row">'
        '<div class="sinergia-dist-head">'
        f"<span>{html.escape(codigo)} · {html.escape(nome)}</span>"
        f'<span class="sinergia-dist-count">{contagem} ({pct_clamped:.0f}%)</span>'
        "</div>"
        '<div class="sinergia-dist-track">'
        f'<div class="sinergia-dist-fill" style="width: {pct_clamped:.1f}%;"></div>'
        "</div>"
        "</div>"
    )


def renderizar_citacoes_html(citacoes: list[str]) -> str:
    """Gera o bloco HTML das pílulas de citação abaixo de uma resposta."""
    if not citacoes:
        return ""
    pills = "".join(
        f'<span class="sinergia-citation-pill">{html.escape(c)}</span>' for c in citacoes
    )
    return f'<div class="sinergia-citations-wrap">{pills}</div>'


def renderizar_grafico_embedding_html(amostra: list[float]) -> str:
    """Gera barras monocromáticas proporcionais para as primeiras dimensões do vetor 768d."""
    if not amostra:
        return ""
    colunas = []
    for idx, val in enumerate(amostra):
        altura_pct = min(100.0, max(12.0, abs(float(val)) * 900.0))
        classe_barra = "sinergia-emb-bar-pos" if val >= 0 else "sinergia-emb-bar-neg"
        colunas.append(
            '<div class="sinergia-emb-col">'
            '<div class="sinergia-emb-track">'
            f'<div class="{classe_barra}" style="height: {altura_pct:.1f}%;"></div>'
            "</div>"
            f'<span class="sinergia-emb-idx">{idx}</span>'
            "</div>"
        )
    return f'<div class="sinergia-emb-bars">{"".join(colunas)}</div>'
