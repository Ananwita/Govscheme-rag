"""Streamlit web app - Multilingual RAG for Indian Government Schemes.

Run from the project folder:
    streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src import config
from src.eligibility import CHECK, LIKELY, NOT, Profile, screen
from src.generator import PROVIDERS
from src.index import build_index
from src.pipeline import RAGPipeline

st.set_page_config(page_title="GovScheme RAG | सरकारी योजना सहायक", page_icon="🏛️", layout="wide")

EXAMPLES = [
    "Who is eligible for PM-KISAN?",
    "पीएम किसान योजना के लिए कौन पात्र है?",
    "Can an income tax payer join Atal Pension Yojana?",
    "आयुष्मान भारत योजना में कितने रुपये का स्वास्थ्य बीमा मिलता है?",
    "What is the premium for PMJJBY life insurance?",
    "मुफ्त गैस कनेक्शन के लिए कौन आवेदन कर सकता है?",
    "How much loan can I get under Shishu category of Mudra?",
    "मनरेगा में साल में कितने दिन काम की गारंटी है?",
]

STATUS_UI = {
    LIKELY: ("✅", "Likely eligible", "संभवतः पात्र"),
    CHECK: ("🟡", "Check details", "विवरण जांचें"),
    NOT: ("❌", "Not eligible", "पात्र नहीं"),
}


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner="Loading multilingual embedding model and index ...")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def _save_llm_env(base_url: str, model: str, api_key: str) -> None:
    """Write the LLM settings into .env so they survive an app restart."""
    import os

    env = config.ROOT / ".env"
    lines = env.read_text(encoding="utf-8").splitlines() if env.exists() else []
    lines = [ln for ln in lines if not ln.strip().startswith(("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"))]
    if base_url:
        lines += [f"LLM_BASE_URL={base_url}", f"LLM_MODEL={model}", f"LLM_API_KEY={api_key}"]
    env.write_text("\n".join(lines) + "\n", encoding="utf-8")
    config.LLM_BASE_URL, config.LLM_MODEL, config.LLM_API_KEY = base_url, model or config.LLM_MODEL, api_key
    os.environ["LLM_BASE_URL"], os.environ["LLM_MODEL"], os.environ["LLM_API_KEY"] = base_url, model, api_key


def scheme_titles(rag: RAGPipeline) -> dict[tuple[str, str], str]:
    return {(c.scheme_id, c.language): c.title for c in rag.retriever.chunks}


# ------------------------------------------------------------------ sidebar
rag = get_pipeline()
titles = scheme_titles(rag)
all_schemes = sorted({c.scheme_id for c in rag.retriever.chunks})

with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Passages to retrieve (top-k)", 2, 10, config.TOP_K)
    ans_lang = st.radio("Answer language", ["auto", "en", "hi"], horizontal=True,
                        help="auto = same language as the question")
    doc_lang = st.radio("Search documents in", ["both", "en", "hi"], horizontal=True,
                        help="Pick the opposite language of your question to demo cross-lingual retrieval")
    pick = st.multiselect("Limit to schemes (optional)", all_schemes,
                          format_func=lambda s: titles.get((s, "en"), s))

    st.divider()
    st.subheader("🤖 LLM (generation step)")
    with st.expander("Connect an LLM", expanded=rag.llm is None):
        provider = st.selectbox("Provider", list(PROVIDERS))
        prov = PROVIDERS[provider]
        st.caption(f"Get a free key: {prov['key_url']}")
        key = st.text_input("API key", type="password",
                            disabled=provider.startswith("Ollama"),
                            help="Not needed for Ollama")
        model = st.text_input("Model", value=prov["model"],
                              help="If this model is not available, a similar one is picked automatically")
        remember = st.checkbox("Remember on this computer (saves to .env)", value=True)
        c1, c2 = st.columns(2)
        if c1.button("Connect", type="primary"):
            with st.spinner("Testing connection ..."):
                ok, msg = rag.configure_llm(prov["base_url"], model.strip(), key.strip() or "ollama")
            if ok:
                st.success(msg)
                if remember:
                    _save_llm_env(prov["base_url"], rag.llm.model, key.strip() or "ollama")
            else:
                st.error(msg)
        if c2.button("Disconnect", disabled=rag.llm is None):
            rag.disable_llm()
            _save_llm_env("", "", "")
            st.rerun()
    use_llm = st.toggle("Use LLM for answer", value=rag.llm is not None, disabled=rag.llm is None,
                        help="Connect an LLM above. Off = offline extractive answers")

    st.divider()
    st.subheader("📊 System")
    info = rag.info
    if info["embedder_active"].startswith("tfidf"):
        st.warning("Running on the TF-IDF fallback. Install `sentence-transformers` for multilingual "
                   "embeddings (see README).")
    st.caption(f"**Embedder:** `{info['embedder_active']}`")
    st.caption(f"**Generator:** {info['llm']}")
    st.caption(f"**Index:** {info['n_documents']} docs · {info['n_chunks']} chunks · "
               f"{len(info['schemes'])} schemes · dim {info['dim']}")

    st.divider()
    st.subheader("📄 Add documents")
    up = st.file_uploader("Upload scheme PDF / TXT / MD", type=["pdf", "txt", "md"],
                          accept_multiple_files=True)
    if st.button("Add & rebuild index", disabled=not up):
        dest = config.DATA_DIR / "uploads"
        dest.mkdir(parents=True, exist_ok=True)
        for f in up:
            (dest / f.name).write_bytes(f.getbuffer())
        with st.spinner("Re-indexing ..."):
            build_index(embedder=rag.embedder)
        st.cache_resource.clear()
        st.rerun()
    if st.button("Rebuild index"):
        with st.spinner("Re-indexing ..."):
            build_index(embedder=rag.embedder)
        st.cache_resource.clear()
        st.rerun()

# ------------------------------------------------------------------ header
st.title("🏛️ Multilingual RAG for Government Schemes")
st.markdown("Ask in **English** or **हिंदी** about Indian government welfare schemes — "
            "answers come with **citations** to the source documents.")

tab_ask, tab_elig, tab_how = st.tabs(["💬 Ask a question", "✅ Eligibility checker", "🔧 How it works"])

# ------------------------------------------------------------------ tab 1: Q&A
with tab_ask:
    if "history" not in st.session_state:
        st.session_state.history = []

    def _set_query(q: str) -> None:
        st.session_state.query_box = q

    st.write("**Try an example:**")
    cols = st.columns(4)
    for i, ex in enumerate(EXAMPLES):
        cols[i % 4].button(ex, key=f"ex{i}", on_click=_set_query, args=(ex,))

    with st.form("ask_form", clear_on_submit=False):
        query = st.text_input("Your question / आपका प्रश्न", key="query_box",
                              placeholder="e.g. Who can apply for Sukanya Samriddhi? / उज्ज्वला योजना के लिए कौन पात्र है?")
        submitted = st.form_submit_button("🔍 Ask", type="primary")

    if submitted and query.strip():
        with st.spinner("Retrieving and answering ..."):
            res = rag.ask(query, top_k=top_k, use_llm=use_llm,
                          languages=None if doc_lang == "both" else [doc_lang],
                          schemes=pick or None, answer_language=ans_lang)
        st.session_state.history.insert(0, res)

    for idx, res in enumerate(st.session_state.history):
        with st.container(border=True):
            st.markdown(f"#### ❓ {res['query']}")
            st.caption(f"Detected language: **{'Hindi' if res['language'] == 'hi' else 'English'}** · "
                       f"intent: {', '.join(res['intents']) or 'general'} · generator: {res['generator']} · "
                       f"retrieval {res['timings_ms']['retrieval']} ms · "
                       f"generation {res['timings_ms']['generation']} ms")
            st.markdown(res["answer"])
            with st.expander(f"📚 Sources ({len(res['citations'])})", expanded=idx == 0):
                for c in res["citations"]:
                    link = c["source"] if c["source"].startswith("http") else None
                    head = f"**[{c['n']}] {c['title']} — {c['section']}** `{c['language']}`"
                    st.markdown(head + (f" · [official source]({link})" if link else f" · {c['source']}"))
                    st.caption(f"fused score {c['score']} · cosine {c['dense']} · BM25 {c['bm25']} · file `{c['file']}`")
                    st.markdown(f"> {c['text'][:600]}{'…' if len(c['text']) > 600 else ''}")
            st.download_button("⬇️ Save this answer (JSON)", json.dumps(res, ensure_ascii=False, indent=2),
                               file_name=f"answer_{idx}.json", key=f"dl{idx}")

    if st.session_state.history and st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

# ------------------------------------------------------------------ tab 2: eligibility
with tab_elig:
    st.markdown("Fill a simple profile to see which schemes you may qualify for. "
                "Each result shows the **cited eligibility text** from the scheme document.")
    with st.form("elig_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age / आयु", 0, 110, 30)
        gender = c1.selectbox("Gender / लिंग", ["female", "male", "other"])
        area = c1.selectbox("Area / क्षेत्र", ["rural", "urban"])
        occupation = c2.selectbox("Occupation / व्यवसाय",
                                  ["farmer", "artisan", "small_business", "labourer", "salaried", "student", "other"])
        owns_land = c2.checkbox("Family owns farmland / खेती की जमीन है")
        bank = c2.checkbox("Has bank account / बैंक खाता है", value=True)
        low_income = c2.checkbox("Low-income / BPL / SECC household")
        tax = c3.checkbox("Pays income tax / आयकरदाता")
        govt = c3.checkbox("Government employee in family")
        kutcha = c3.checkbox("Houseless or kutcha house / कच्चा घर")
        lpg = c3.checkbox("Household already has LPG", value=True)
        girl = c3.checkbox("Has a daughter below 10 / 10 साल से कम बेटी")
        show_lang = st.radio("Show cited text in", ["en", "hi"], horizontal=True)
        go = st.form_submit_button("Check eligibility", type="primary")

    if go:
        prof = Profile(age=int(age), gender=gender, area=area, occupation=occupation,
                       owns_farmland=owns_land, income_tax_payer=tax, government_employee=govt,
                       has_bank_account=bank, low_income_household=low_income,
                       kutcha_or_no_house=kutcha, household_has_lpg=lpg, girl_child_under_10=girl)
        results = screen(prof)
        n_likely = sum(r.status == LIKELY for r in results)
        st.success(f"{n_likely} scheme(s) likely, {sum(r.status == CHECK for r in results)} to check.")
        for r in results:
            icon, en_lbl, hi_lbl = STATUS_UI[r.status]
            title = titles.get((r.scheme_id, show_lang), titles.get((r.scheme_id, "en"), r.scheme_id))
            with st.expander(f"{icon} {title} — {en_lbl} / {hi_lbl}", expanded=r.status == LIKELY):
                st.markdown("**Why:** " + " ".join(r.reasons))
                secs = ["eligibility", "exclusions", "पात्रता", "अपात्रता"]
                for ch in rag.retriever.scheme_chunks(r.scheme_id, show_lang, secs):
                    st.markdown(f"**📄 {ch.section}** — source: {ch.source}")
                    st.markdown(ch.text)
        st.caption("This is a screening aid built on simplified rules; final eligibility is decided by the "
                   "implementing authority.")

# ------------------------------------------------------------------ tab 3: how it works
with tab_how:
    diagram = config.ROOT / "docs" / "pipeline_diagram.png"
    if diagram.exists():
        st.image(str(diagram), caption="Processing pipeline")
    st.markdown("""
1. **Ingestion** – scheme documents (Markdown / PDF, English + Hindi) are parsed with their metadata
   (scheme, language, official source URL).
2. **Section-aware chunking** – documents are split on headings (Eligibility, Benefits, …) into
   ≤200-word chunks so each chunk carries a citable section name.
3. **Multilingual embeddings** – `intfloat/multilingual-e5-small` maps Hindi and English into one vector
   space; vectors are stored in `index/`.
4. **Hybrid retrieval** – cosine similarity (dense) + BM25 (keywords) fused with Reciprocal Rank Fusion,
   plus same-language and section-intent boosts.
5. **Answer generation (LLM)** – an LLM (Groq / Gemini / Ollama) writes the answer ONLY from the
   retrieved passages, in the user's language, with `[n]` citations; an offline extractive answerer is the fallback.
6. **Eligibility checker** – rule-based screening that shows the cited eligibility text.
""")
    res_md = config.EVAL_DIR / "results.md"
    chart = config.EVAL_DIR / "metrics_chart.png"
    if res_md.exists():
        st.subheader("Evaluation results")
        st.markdown(res_md.read_text(encoding="utf-8"))
    if chart.exists():
        st.image(str(chart))
    if not res_md.exists():
        st.info("Run `python evaluate.py` to generate evaluation results.")
