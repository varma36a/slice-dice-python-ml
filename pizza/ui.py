"""Shared Streamlit chrome for the Slice & Dice course."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

from pizza.data import Shop, build_shop

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"]  {
  font-family: "IBM Plex Sans", sans-serif;
}
h1, h2, h3, .hero-title {
  font-family: "Fraunces", Georgia, serif !important;
  letter-spacing: -0.02em;
}
.stApp {
  background:
    radial-gradient(1200px 500px at 10% -10%, rgba(232, 93, 4, 0.16), transparent 50%),
    radial-gradient(900px 400px at 100% 0%, rgba(250, 163, 7, 0.08), transparent 45%),
    #0E1117;
}
.hero {
  border: 1px solid rgba(232, 93, 4, 0.35);
  background: linear-gradient(135deg, rgba(22,27,34,0.95), rgba(14,17,23,0.92));
  padding: 1.4rem 1.6rem 1.2rem;
  border-radius: 18px;
  margin-bottom: 1.1rem;
}
.hero-kicker {
  color: #FAA307;
  font-weight: 600;
  font-size: 0.82rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  margin-bottom: 0.35rem;
}
.hero-title { font-size: 2.05rem; margin: 0 0 0.35rem 0; color: #fff; }
.hero-sub { color: #9BA4B0; font-size: 1.02rem; margin: 0; }
.card {
  border: 1px solid #30363d;
  background: #161B22;
  border-radius: 14px;
  padding: 0.95rem 1.05rem;
  margin: 0.4rem 0 0.85rem;
}
.card h4 { margin: 0 0 0.35rem 0; color: #FAA307; font-size: 0.92rem; letter-spacing: 0.04em; text-transform: uppercase;}
.why { border-left: 3px solid #E85D04; padding-left: 0.85rem; margin: 0.6rem 0 1rem; color: #D0D7DE; }
.pitfall { border-left: 3px solid #F85149; padding-left: 0.85rem; margin: 0.6rem 0 1rem; color: #D0D7DE; }
.ok { border-left: 3px solid #3FB950; padding-left: 0.85rem; margin: 0.6rem 0 1rem; color: #D0D7DE; }
code, pre, .stCode { font-family: "IBM Plex Mono", ui-monospace, monospace !important; }
div[data-testid="stMetric"] {
  background: #161B22;
  border: 1px solid #30363d;
  padding: 0.6rem 0.8rem;
  border-radius: 12px;
}
.module-chip {
  display: inline-block;
  background: rgba(232, 93, 4, 0.15);
  color: #FAA307;
  border: 1px solid rgba(232, 93, 4, 0.4);
  padding: 0.15rem 0.55rem;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 600;
  margin-right: 0.35rem;
}
</style>
"""


def inject() -> None:
    st.set_page_config(
        page_title="Slice & Dice · Python for ML",
        page_icon="🍕",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("**Slice & Dice Pizzeria**")
        st.caption("Python for ML · pizza-shop lab")
        st.markdown("---")


def header(kicker: str, title: str, sub: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-kicker">{kicker}</div>
          <h1 class="hero-title">{title}</h1>
          <p class="hero-sub">{sub}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def why(text: str) -> None:
    st.markdown(f'<div class="why"><strong>Why it matters in ML.</strong> {text}</div>', unsafe_allow_html=True)


def pitfall(text: str) -> None:
    st.markdown(f'<div class="pitfall"><strong>Pitfall.</strong> {text}</div>', unsafe_allow_html=True)


def ok(text: str) -> None:
    st.markdown(f'<div class="ok"><strong>Do this.</strong> {text}</div>', unsafe_allow_html=True)


def card(title: str, body: str) -> None:
    st.markdown(f'<div class="card"><h4>{title}</h4><div>{body}</div></div>', unsafe_allow_html=True)


@st.cache_data(show_spinner="Firing the ovens — generating 8 weeks of shop data…")
def load_shop(days: int = 56) -> Shop:
    return build_shop(days=days)
