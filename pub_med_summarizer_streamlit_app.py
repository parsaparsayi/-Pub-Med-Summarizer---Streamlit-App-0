"""
Streamlit GUI for PubMed search + article summarization (English-only UI)

How to run locally:
1) Create a virtual env (optional) and install deps:
   pip install streamlit requests beautifulsoup4 transformers torch --upgrade
   # If transformers pulls a heavy model, first run will download weights.
2) Save this file as app.py and launch:
   streamlit run app.py

Notes:
- Default summarization model is a lightweight DistilBART; you can switch to BART-Large from the sidebar.
- PubMed markup can change; parsing is defensive but not guaranteed forever.
- English-only UI.
"""

from __future__ import annotations

import io
import re
import base64
from typing import List, Tuple, Optional, Dict
import pathlib

import requests
from bs4 import BeautifulSoup
import streamlit as st
from transformers import pipeline
import pathlib

# ----------------------------- Page & Styling ----------------------------- #

def apply_space_theme(image_bytes: bytes | None = None, image_url: str | None = None, overlay_alpha: float = 0.6):
    """Inject CSS for a NASA/space look. Use either image bytes or a URL.
    If neither is provided, falls back to a dark starry gradient.
    """
    if image_bytes:
        encoded = base64.b64encode(image_bytes).decode()
        bg_css = f"url('data:image/jpeg;base64,{encoded}')"
    elif image_url:
        bg_css = f"url('{image_url}')"
    else:
        bg_css = (
            "radial-gradient(ellipse at 20% 10%, rgba(255,255,255,.08), rgba(0,0,0,0) 40%), "
            "radial-gradient(ellipse at 80% 50%, rgba(255,255,255,.06), rgba(0,0,0,0) 40%), #000014"
        )

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0,0,20,{overlay_alpha}), rgba(0,0,20,{overlay_alpha})), {bg_css};
            background-attachment: fixed;
            background-size: cover;
        }}
        /* Glass panels */
        .panel {{
            background: rgba(6, 10, 28, 0.55);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1rem 1.25rem;
            box-shadow: 0 10px 40px rgba(0,0,0,.35);
            backdrop-filter: blur(6px);
        }}
        /* Headings/colors */
        h1, h2, h3, .stMarkdown p strong {{ color: #e6f0ff; }}
        /* Primary buttons */
        .stButton>button {{
            background: #0B3D91; /* NASA blue */
            color: #ffffff;
            border-radius: 999px;
            border: 1px solid #113a8a;
        }}
        .stButton>button:hover {{ background: #1e56b3; }}
        /* Downloads */
        .stDownloadButton>button {{ border-radius: 999px; }}
        /* Radios, expanders */
        .stRadio label {{ color: #d9e6ff; }}
        .streamlit-expanderHeader {{ color: #d9e6ff; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
st.set_page_config(
    page_title="NASA‑style PubMed Summarizer",
    page_icon="🛰️",
    layout="wide",
)

# UI text (English only)
T: Dict[str, str] = {
    "title": "🧪 PubMed Search & Summarizer",
    "desc": "Enter a query, pick a paper, and generate a concise summary of the abstract using a Transformers model.",
    "query": "Search query",
    "search": "Search",
    "num_results": "Max results",
    "results": "Search results",
    "select": "Select an article",
    "fetch_details": "Fetch details",
    "title_lbl": "Title",
    "abstract_lbl": "Abstract",
    "link_lbl": "PubMed link",
    "summarize": "Summarize",
    "summary": "AI Summary",
    "model": "Summarization model",
    "min_len": "Min summary length",
    "max_len": "Max summary length",
    "copypaste": "Or paste a PubMed link directly",
    "nothing": "No results found.",
    "parsing_err": "Couldn't parse this page.",
    "download_sum": "Download summary (.txt)",
    "download_abs": "Download abstract (.txt)",
    "clear": "Clear results",
    "lang": "Language",
}

# Sidebar controls
st.sidebar.markdown("### ⚙️ Settings")

# --- Space theme controls ---
ASSETS = pathlib.Path(__file__).parent / "assets"
bg_upl = st.sidebar.file_uploader("Background image (space)", type=["png", "jpg", "jpeg"])
bg_url = st.sidebar.text_input("Background image URL (optional)")
overlay = st.sidebar.slider("Background overlay", 0.0, 0.9, 0.0, 0.01)

# Priority: uploaded > URL > local asset > default NASA URL
if bg_upl is not None:
    data = bg_upl.read()
    try:
        ASSETS.mkdir(parents=True, exist_ok=True)
        (ASSETS / "bg.jpg").write_bytes(data)  # persist for next runs
        st.sidebar.success("Saved as assets/bg.jpg — will load by default next time.")
    except Exception:
        st.sidebar.warning("Couldn't save to assets/. Using in-memory background only.")
    apply_space_theme(image_bytes=data, image_url=None, overlay_alpha=overlay)
elif bg_url:
    apply_space_theme(image_bytes=None, image_url=bg_url.strip(), overlay_alpha=overlay)
elif (ASSETS / "bg.jpg").exists():
    apply_space_theme(image_bytes=(ASSETS / "bg.jpg").read_bytes(), image_url=None, overlay_alpha=overlay)
else:
    default_url = "https://www.nasa.gov/wp-content/uploads/2015/06/huDF2014-20140603a.jpg"
    apply_space_theme(image_bytes=None, image_url=default_url, overlay_alpha=overlay)


# Model choice & lengths
model_name = st.sidebar.selectbox(
    T["model"],
    (
        "sshleifer/distilbart-cnn-12-6",  # small & fast
        "facebook/bart-large-cnn",        # higher quality, heavier
        "google/pegasus-xsum",            # abstractive, may be heavy
    ),
    index=0,
)
min_len = st.sidebar.slider(T["min_len"], min_value=30, max_value=200, value=60, step=10)
max_len = st.sidebar.slider(T["max_len"], min_value=80, max_value=400, value=180, step=10)
max_results = st.sidebar.slider(T["num_results"], min_value=5, max_value=50, value=20, step=5)

st.sidebar.caption(
    "GPU not required; CPU works but first run downloads model weights."
)

# ----------------------------- Helpers ----------------------------- #
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/119.0 Safari/537.36"
    )
}

@st.cache_data(show_spinner=False, ttl=1800)
def search_pub(query: str, limit: int = 20) -> List[Tuple[str, str]]:
    if not query:
        return []
    url = "https://pubmed.ncbi.nlm.nih.gov/"
    params = {"term": query}
    r = requests.get(url, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    links = soup.find_all("a", class_="docsum-title")
    results: List[Tuple[str, str]] = []
    for a in links[:limit]:
        title = a.get_text(strip=True)
        href = a.get("href", "").strip()
        if href and not href.startswith("http"):
            href = f"https://pubmed.ncbi.nlm.nih.gov{href}"
        if title and href:
            results.append((title, href))
    return results

@st.cache_data(show_spinner=False, ttl=3600)
def get_article_details(link: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not link:
        return None, None, None
    r = requests.get(link, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Title
    title_el = soup.select_one("h1.heading-title") or soup.select_one(".heading-title")
    title = title_el.get_text(" ", strip=True) if title_el else None

    # Abstract (robust across minor DOM changes)
    abstract_text = None
    candidates = [
        soup.select_one("div.abstract"),
        soup.select_one("section#abstract"),
        soup.select_one("div#enc-abstract"),
    ]
    for cand in candidates:
        if cand:
            # Join paragraph children if present; fallback to raw text
            parts = [p.get_text(" ", strip=True) for p in cand.find_all(["p", "div"], recursive=False)]
            raw = "\n\n".join([p for p in parts if p]) if parts else cand.get_text(" ", strip=True)
            abstract_text = raw if raw else None
            if abstract_text:
                break

    return title, abstract_text, link

@st.cache_resource(show_spinner=False)
def get_summarizer(model: str):
    return pipeline("summarization", model=model)

# Basic sentence-ish splitter for chunking long abstracts
_SENTENCE_SPLIT = re.compile(r"(?<=[\.!؟\?])\s+")

def chunk_text(text: str, max_chars: int = 2000, overlap: int = 150) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = _SENTENCE_SPLIT.split(text)
    chunks: List[str] = []
    buf: List[str] = []
    cur_len = 0
    for s in sentences:
        if cur_len + len(s) + 1 > max_chars:
            if buf:
                chunks.append(" ".join(buf))
            # start new buffer with overlap from the end of previous
            if overlap and chunks:
                tail = chunks[-1][-overlap:]
                buf = [tail, s]
                cur_len = len(tail) + len(s)
            else:
                buf = [s]
                cur_len = len(s)
        else:
            buf.append(s)
            cur_len += len(s) + 1
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def summarize_text(text: str, model: str, min_len: int, max_len: int) -> str:
    # Guardrail for misconfigured lengths
    if max_len <= min_len:
        max_len = min_len + 20

    summarizer = get_summarizer(model)
    chunks = chunk_text(text)

    summaries: List[str] = []
    progress = st.progress(0) if len(chunks) > 1 else None
    for i, ch in enumerate(chunks, start=1):
        out = summarizer(ch, max_length=max_len, min_length=min_len, do_sample=False)
        summaries.append(out[0]["summary_text"])  # type: ignore
        if progress:
            progress.progress(i / len(chunks))
    if progress:
        progress.empty()

    combined = "\n\n".join(summaries)
    # Optional second pass to compress combined summary if it's long
    if len(combined) > 1500:
        out = summarizer(combined, max_length=max_len, min_length=min_len, do_sample=False)
        return out[0]["summary_text"]  # type: ignore
    return combined

# ----------------------------- App Body ----------------------------- #
st.markdown(
    """
    <div class="panel" style="margin-bottom:1rem">
      <h1>🛰️ NASA‑style PubMed Summarizer</h1>
      <p>Search PubMed, choose an article, and let the onboard model condense the abstract.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Share link (top of page) ---
SHARE_URL = "https://pubmedsummerizer.streamlit.app/"
st.success(f"Share this app: {SHARE_URL}")
st.text_input("Copy link", value=SHARE_URL, disabled=True, label_visibility="collapsed")

# Inputs
q = st.text_input(T["query"], placeholder="e.g. CRISPR base editing delivery OR Alzheimer's tau PET")
colA, colB, colC = st.columns([0.25, 0.25, 0.5])
with colA:
    do_search = st.button(T["search"], type="primary")
with colB:
    do_clear = st.button(T["clear"])  # clear results
with colC:
    pasted_link = st.text_input(T["copypaste"], placeholder="https://pubmed.ncbi.nlm.nih.gov/...")

if do_clear:
    for k in ("results", "selected", "details", "summary"):
        if k in st.session_state:
            del st.session_state[k]

# Search action
if do_search and q.strip():
    with st.spinner("Searching PubMed…"):
        try:
            st.session_state["results"] = search_pub(q.strip(), max_results)
        except Exception as e:
            st.error(str(e))
            st.session_state["results"] = []

# Results block
results: List[Tuple[str, str]] = st.session_state.get("results", [])
if results:
    st.subheader(T["results"])
    titles = [t for t, _ in results]
    idx = st.radio(T["select"], range(1, len(titles) + 1), format_func=lambda i: titles[i-1], horizontal=False)
    selected = results[idx - 1]
    st.session_state["selected"] = selected

elif "results" in st.session_state and not results:
    st.info(T["nothing"])  # explicit empty result set

# Fetch details either from selected or pasted link
target_link: Optional[str] = None
if pasted_link and pasted_link.startswith("http"):
    target_link = pasted_link.strip()
elif st.session_state.get("selected"):
    target_link = st.session_state["selected"][1]

if target_link:
    with st.spinner("Fetching article details…"):
        try:
            details = get_article_details(target_link)
            st.session_state["details"] = details
        except Exception as e:
            st.error(str(e))
            st.session_state["details"] = (None, None, None)

# Show details
if st.session_state.get("details"):
    title, abstract, link = st.session_state["details"]
    if title and abstract and link:
        st.markdown(f"**{T['title_lbl']}:** {title}")
        st.markdown(f"**{T['link_lbl']}:** [{link}]({link})")
        with st.expander(T["abstract_lbl"], expanded=False):
            st.write(abstract)

        # Summarize button
        if st.button(T["summarize"], type="secondary"):
            with st.spinner("Summarizing…"):
                try:
                    summary = summarize_text(abstract, model_name, min_len, max_len)
                    st.session_state["summary"] = summary
                except Exception as e:
                    st.error(str(e))

# Summary output + downloads
if st.session_state.get("summary"):
    st.subheader(T["summary"])
    st.write(st.session_state["summary"])

    # Downloads
    sum_bytes = io.BytesIO(st.session_state["summary"].encode("utf-8"))
    st.download_button(T["download_sum"], data=sum_bytes, file_name="summary.txt", mime="text/plain")

    if st.session_state.get("details"):
        _t, abs_text, _l = st.session_state["details"]
        if abs_text:
            abs_bytes = io.BytesIO(abs_text.encode("utf-8"))
            st.download_button(T["download_abs"], data=abs_bytes, file_name="abstract.txt", mime="text/plain")

# Footer
st.caption(
    "Built with Streamlit · Requests · BeautifulSoup · Transformers. "
    "Models: DistilBART/BART/PEGASUS (Hugging Face)."
)
