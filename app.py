"""
app.py — Streamlit entry point for the AI-Assisted Cryptanalysis application.

Launch with:
    streamlit run app.py

Navigation structure:
    Sidebar → Home | Encrypt | Decrypt | Cryptanalysis | Frequency Analysis | ML Model | About
"""

import sys
import os

# ── ensure project root is on sys.path so all package imports resolve ──────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

# ── page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="AI Cryptanalysis Lab",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── shared imports ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import string

from ciphers.caesar       import CaesarCipher
from ciphers.vigenere     import VigenereCipher
from ciphers.substitution import SubstitutionCipher
from analysis.frequency   import FrequencyAnalyzer, ENGLISH_LETTER_FREQ
from analysis.language_score import LanguageScorer
from analysis.cipher_detection import CipherDetector
from utils.helpers import (
    validate_input, index_of_coincidence, character_entropy, text_to_upper_alpha
)

# ── lazy-loaded ML predictor (triggers training only once, cached) ─────────────
@st.cache_resource(show_spinner="Loading / training ML model…")
def get_predictor():
    from ml.predictor import CipherPredictor
    return CipherPredictor()

# ── module singletons ──────────────────────────────────────────────────────────
caesar   = CaesarCipher()
vigenere = VigenereCipher()
subst    = SubstitutionCipher()
freq_ana = FrequencyAnalyzer()
scorer   = LanguageScorer()
detector = CipherDetector()

# ══════════════════════════════════════════════════════════════════════════════
# Theme / CSS
# ══════════════════════════════════════════════════════════════════════════════

CYBER_CSS = """
<style>
/* Dark cybersecurity colour palette */
:root {
    --cyber-green:  #00ff9f;
    --cyber-blue:   #00cfff;
    --cyber-red:    #ff4466;
    --cyber-bg:     #0d1117;
    --cyber-card:   #161b22;
    --cyber-border: #30363d;
    --cyber-muted:  #8b949e;
}

/* Metric cards */
.metric-card {
    background: var(--cyber-card);
    border: 1px solid var(--cyber-border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
}
.metric-label { color: var(--cyber-muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-value { color: var(--cyber-green); font-size: 1.6rem; font-weight: 700; font-family: monospace; }

/* Candidate result blocks */
.candidate-block {
    background: var(--cyber-card);
    border-left: 3px solid var(--cyber-green);
    border-radius: 4px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-family: monospace;
}
.candidate-block.best { border-left-color: #ffd700; }

/* Section headers */
.section-header {
    color: var(--cyber-green);
    font-family: monospace;
    border-bottom: 1px solid var(--cyber-border);
    padding-bottom: 0.3rem;
    margin-bottom: 0.8rem;
}

/* Confidence bar */
.conf-bar-outer { background: #21262d; border-radius: 4px; height: 10px; margin: 4px 0; }
.conf-bar-inner { background: linear-gradient(90deg, #00ff9f, #00cfff); height: 10px; border-radius: 4px; }

/* Monospace plaintext display */
.plaintext-display {
    background: #010409;
    border: 1px solid var(--cyber-border);
    border-radius: 6px;
    padding: 1rem;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
    color: #e6edf3;
    word-break: break-all;
    max-height: 200px;
    overflow-y: auto;
}
</style>
"""
st.markdown(CYBER_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Sample ciphertexts for instant demonstration
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_CAESAR = caesar.encrypt(
    "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG AND THE CIPHER MUST BE BROKEN", 3
)
SAMPLE_VIGENERE = vigenere.encrypt(
    "CRYPTOGRAPHY IS THE PRACTICE AND STUDY OF TECHNIQUES FOR SECURE COMMUNICATION", "SECRET"
)
_subst_key = subst.keyword_key("ZEBRA")
SAMPLE_SUBSTITUTION = subst.encrypt(
    "FREQUENCY ANALYSIS IS THE CORNERSTONE OF CLASSICAL CIPHER CRYPTANALYSIS", _subst_key
)

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar navigation
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🔐 AI Cryptanalysis Lab")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔒 Encrypt", "🔓 Decrypt",
         "🔍 Cryptanalysis", "📊 Frequency Analysis",
         "🤖 ML Model", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#8b949e'>College Cybersecurity Project<br>"
        "AI-Assisted Cryptanalysis<br>Python 3.13 · Streamlit · scikit-learn</small>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ══════════════════════════════════════════════════════════════════════════════

def freq_bar_chart(text: str, title: str = "Letter Frequency Distribution"):
    """Matplotlib bar chart comparing ciphertext frequencies to English."""
    letters = text_to_upper_alpha(text)
    if not letters:
        return None

    obs  = freq_ana.letter_frequencies(letters)
    eng  = ENGLISH_LETTER_FREQ
    alph = list(string.ascii_uppercase)
    x    = np.arange(26)
    w    = 0.4

    fig, ax = plt.subplots(figsize=(12, 3.5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    bars_obs = ax.bar(x - w/2, [obs[c] for c in alph], w,
                      color="#00ff9f", alpha=0.85, label="Ciphertext")
    bars_eng = ax.bar(x + w/2, [eng[c] for c in alph], w,
                      color="#00cfff", alpha=0.6,  label="English")

    ax.set_xticks(x)
    ax.set_xticklabels(alph, color="#e6edf3", fontsize=8)
    ax.tick_params(axis="y", colors="#8b949e", labelsize=7)
    ax.set_ylabel("Relative frequency", color="#8b949e", fontsize=8)
    ax.set_title(title, color="#00ff9f", fontsize=10, pad=8)
    ax.legend(framealpha=0.2, labelcolor="white", fontsize=8)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.yaxis.grid(True, color="#21262d", linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def ic_bar(value: float, label: str = ""):
    """Small inline confidence bar rendered with HTML."""
    pct = min(int(value * 100), 100)
    return (
        f"<div style='margin:2px 0'>"
        f"<span style='color:#8b949e;font-size:0.78rem;font-family:monospace'>{label}</span> "
        f"<span style='color:#00ff9f;font-weight:700;font-family:monospace'>{value*100:.1f}%</span><br>"
        f"<div class='conf-bar-outer'><div class='conf-bar-inner' style='width:{pct}%'></div></div>"
        f"</div>"
    )


def metric_card(label: str, value: str):
    return (
        f"<div class='metric-card'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div>"
        f"</div>"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Home
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Home":
    st.markdown("# 🔐 AI-Assisted Cryptanalysis of Classical Ciphers")
    st.markdown(
        "<p style='color:#8b949e;font-size:1.05rem'>"
        "An educational cybersecurity laboratory combining <strong>statistical analysis</strong>, "
        "<strong>classical cryptanalysis</strong>, and <strong>machine learning</strong> "
        "to automatically detect and break classical ciphers."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 🏛️ Supported Ciphers")
        st.markdown(
            "- **Caesar** — shift cipher (key: 0–25)\n"
            "- **Vigenère** — polyalphabetic keyword cipher\n"
            "- **Substitution** — 26-letter permutation cipher"
        )
    with c2:
        st.markdown("### 🔬 Analysis Methods")
        st.markdown(
            "- Letter frequency analysis\n"
            "- Index of Coincidence (IC)\n"
            "- Kasiski examination\n"
            "- Chi-squared scoring\n"
            "- Bigram / trigram log-prob scoring"
        )
    with c3:
        st.markdown("### 🤖 ML Component")
        st.markdown(
            "- 48-dimensional feature vector\n"
            "- IC-profile across key lengths\n"
            "- Random Forest classifier\n"
            "- 2400 synthetic training samples\n"
            "- ~93% test accuracy"
        )

    st.markdown("---")
    st.markdown("### ⚙️ System Architecture")
    st.code(
        """
        Ciphertext input
               │
               ├─► Feature extraction (48-dim vector)
               │         ├── 26 letter frequencies
               │         ├── IC, entropy, chi-squared
               │         ├── bigram/trigram log-probs
               │         └── IC profile at kl=1..10
               │
               ├─► Rule-based heuristics (60% weight)
               │         ├── IC vs English/random thresholds
               │         ├── Estimated key length
               │         └── Frequency distribution shape
               │
               ├─► Random Forest ML (40% weight)
               │         └── Trained on 2400 labelled samples
               │
               └─► Combined cipher detection → Cryptanalysis → Ranked candidates
        """,
        language="text",
    )

    st.markdown("---")
    st.markdown("### 🚀 Quick Start — Sample Ciphertexts")
    tabs = st.tabs(["Caesar (key=3)", "Vigenère (key=SECRET)", "Substitution (key=ZEBRA)"])
    with tabs[0]:
        st.code(SAMPLE_CAESAR, language=None)
        st.caption("Plaintext: THE QUICK BROWN FOX…  |  Key: 3")
    with tabs[1]:
        st.code(SAMPLE_VIGENERE, language=None)
        st.caption("Plaintext: CRYPTOGRAPHY IS THE PRACTICE…  |  Key: SECRET")
    with tabs[2]:
        st.code(SAMPLE_SUBSTITUTION, language=None)
        st.caption("Plaintext: FREQUENCY ANALYSIS IS…  |  Key built from: ZEBRA")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Encrypt
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔒 Encrypt":
    st.markdown("# 🔒 Cipher Encryption")
    st.markdown("<p style='color:#8b949e'>Encrypt plaintext with Caesar, Vigenère, or Substitution.</p>",
                unsafe_allow_html=True)

    cipher_choice = st.selectbox("Cipher", ["Caesar", "Vigenère", "Substitution"])
    plaintext_in  = st.text_area("Plaintext", height=120,
                                 placeholder="Enter text to encrypt…")

    key_input = None
    if cipher_choice == "Caesar":
        key_input = st.slider("Shift key", 0, 25, 3)
    elif cipher_choice == "Vigenère":
        key_input = st.text_input("Keyword", value="SECRET").upper()
    else:
        method = st.radio("Key method", ["Random key", "Keyword-derived key"])
        if method == "Keyword-derived key":
            kw = st.text_input("Keyword for substitution key", value="ZEBRA")
            key_input = subst.keyword_key(kw)
        else:
            if "subst_key" not in st.session_state:
                st.session_state.subst_key = subst.random_key()
            key_input = st.session_state.subst_key
        st.code(f"Key: {key_input}", language=None)

    if st.button("🔒 Encrypt", type="primary") and plaintext_in:
        if cipher_choice == "Caesar":
            ct = caesar.encrypt(plaintext_in, key_input)
            st.success(f"Key used: {key_input} ({chr(ord('A') + key_input)})")
        elif cipher_choice == "Vigenère":
            if not key_input or not text_to_upper_alpha(key_input):
                st.error("Please enter a valid keyword (letters only).")
                st.stop()
            ct = vigenere.encrypt(plaintext_in, key_input)
            st.success(f"Key used: {key_input}")
        else:
            ct = subst.encrypt(plaintext_in, key_input)
            st.success("Substitution key applied.")

        st.markdown("#### 📋 Ciphertext")
        st.markdown(f"<div class='plaintext-display'>{ct}</div>", unsafe_allow_html=True)
        st.download_button("⬇ Download ciphertext", ct, file_name="ciphertext.txt")

        col1, col2, col3 = st.columns(3)
        letters = text_to_upper_alpha(ct)
        col1.metric("Letters", len(letters))
        col2.metric("IC", f"{index_of_coincidence(letters):.4f}")
        col3.metric("Entropy", f"{character_entropy(letters):.3f} bits")

        if len(letters) >= 10:
            fig = freq_bar_chart(ct, "Ciphertext vs English Letter Frequencies")
            if fig:
                st.pyplot(fig)
                plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Decrypt
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔓 Decrypt":
    st.markdown("# 🔓 Cipher Decryption")
    st.markdown("<p style='color:#8b949e'>Decrypt ciphertext when you know the key.</p>",
                unsafe_allow_html=True)

    cipher_choice = st.selectbox("Cipher", ["Caesar", "Vigenère", "Substitution"])
    ct_in = st.text_area("Ciphertext", height=120, placeholder="Enter ciphertext…")

    key_in = None
    if cipher_choice == "Caesar":
        key_in = st.slider("Shift key", 0, 25, 3)
    elif cipher_choice == "Vigenère":
        key_in = st.text_input("Keyword", value="SECRET").upper()
    else:
        key_in = st.text_input("26-letter substitution key",
                               value="ZEBRASCDFGHIJKLMNOPQTUVWXY").upper()

    if st.button("🔓 Decrypt", type="primary") and ct_in:
        valid, err = validate_input(ct_in)
        if not valid:
            st.warning(err)
        else:
            if cipher_choice == "Caesar":
                pt = caesar.decrypt(ct_in, key_in)
                st.success(f"Key: {key_in} ({chr(ord('A') + key_in)})")
            elif cipher_choice == "Vigenère":
                pt = vigenere.decrypt(ct_in, key_in)
                st.success(f"Key: {key_in}")
            else:
                if not subst.validate_key(key_in):
                    st.error("Invalid substitution key — must be all 26 letters exactly once.")
                    st.stop()
                pt = subst.decrypt(ct_in, key_in)
                st.success("Decryption applied.")

            st.markdown("#### 📋 Plaintext")
            st.markdown(f"<div class='plaintext-display'>{pt}</div>", unsafe_allow_html=True)
            score = scorer.score(pt)
            st.metric("English plausibility score", f"{score:.3f}",
                      help="0 = gibberish, 1 = perfect English")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Cryptanalysis
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Cryptanalysis":
    st.markdown("# 🔍 Automatic Cryptanalysis")
    st.markdown(
        "<p style='color:#8b949e'>"
        "Paste ciphertext below. The system will detect the likely cipher, "
        "estimate the key, and present ranked candidate plaintexts.</p>",
        unsafe_allow_html=True,
    )

    # Sample loader
    sample_choice = st.selectbox(
        "Load a sample ciphertext",
        ["— type your own —", "Caesar sample", "Vigenère sample", "Substitution sample"],
    )
    default_ct = ""
    if sample_choice == "Caesar sample":        default_ct = SAMPLE_CAESAR
    elif sample_choice == "Vigenère sample":    default_ct = SAMPLE_VIGENERE
    elif sample_choice == "Substitution sample": default_ct = SAMPLE_SUBSTITUTION

    ct_input = st.text_area("Ciphertext input", value=default_ct, height=150,
                            placeholder="Paste your ciphertext here…")

    if st.button("🔍 Analyse", type="primary"):
        valid, err = validate_input(ct_input, min_length=10)
        if not valid:
            st.error(err)
            st.stop()

        with st.spinner("Running cryptanalysis…"):

            # ── 1. Cipher detection ──────────────────────────────────────────
            predictor = get_predictor()
            detection = detector.detect(ct_input)
            stats     = detector.get_statistics(ct_input)

            # ── 2. Cryptanalysis per cipher ──────────────────────────────────
            caesar_results  = caesar.brute_force(ct_input)
            vigenere_results = vigenere.crack(ct_input, max_key_len=12)
            subst_results   = subst.crack(ct_input)

        st.markdown("---")
        # ── Detection scores ─────────────────────────────────────────────────
        st.markdown("<h3 class='section-header'>🧠 Cipher Detection</h3>",
                    unsafe_allow_html=True)
        dcols = st.columns(len(detection))
        for i, d in enumerate(detection):
            with dcols[i]:
                medal = "🥇" if i == 0 else ("🥈" if i == 1 else "🥉")
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-label'>{medal} {d['cipher']}</div>"
                    f"<div class='metric-value'>{d['percentage']:.1f}%</div>"
                    f"<hr style='border-color:#30363d;margin:6px 0'>"
                    f"<small style='color:#8b949e'>Rule score: {d['rule_score']*100:.1f}%<br>"
                    f"ML score:   {d['ml_score']*100:.1f}%</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # ── Statistics ───────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("<h3 class='section-header'>📈 Statistical Summary</h3>",
                    unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        s1.markdown(metric_card("Letters", str(stats["letter_count"])), unsafe_allow_html=True)
        s2.markdown(metric_card("IC", f"{stats['index_of_coincidence']:.4f}"), unsafe_allow_html=True)
        s3.markdown(metric_card("Entropy", f"{stats['entropy']:.3f} bits"), unsafe_allow_html=True)
        s4.markdown(metric_card("Chi²", f"{stats['chi_squared']:.1f}"), unsafe_allow_html=True)

        st.caption(f"IC interpretation: {stats['ic_interpretation']}")

        # Frequency chart
        fig = freq_bar_chart(ct_input, "Letter Frequency: Ciphertext vs English")
        if fig:
            st.pyplot(fig)
            plt.close(fig)

        # ── Candidate plaintexts ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("<h3 class='section-header'>🏆 Candidate Plaintexts (Ranked by Score)</h3>",
                    unsafe_allow_html=True)

        # Gather all candidates from all three ciphers
        all_candidates = []
        if caesar_results:
            all_candidates.extend(caesar_results[:3])
        if vigenere_results:
            all_candidates.extend(vigenere_results[:2])
        if subst_results:
            all_candidates.extend(subst_results[:1])

        # ── Detection-weighted ranking ──────────────────────────────────────
        # Problem: raw language scores from different cipher attacks are NOT
        # directly comparable. The substitution hill-climbing optimizes the
        # same statistical signals used for scoring, so it can produce a
        # frequency-matched-but-gibberish candidate that outranks the correct
        # Caesar plaintext on raw score alone.
        #
        # Fix: multiply each candidate's language score by a weight derived
        # from the detection system's confidence in that cipher type.
        #
        #   adjusted = lang_score × (BASE + WEIGHT × detection_confidence)
        #
        # BASE = 0.35 ensures no candidate is completely suppressed (in case
        # detection is wrong). WEIGHT = 0.65 gives detection significant influence.
        BASE   = 0.35
        WEIGHT = 0.65

        # Build lookup: cipher name → combined detection confidence
        # Handle Vigenère/Vigenere spelling variant (cipher modules use accent)
        det_conf = {}
        for d in detection:
            name = d["cipher"]
            conf = d["combined_score"]
            det_conf[name] = conf
            # Normalize accent variant so lookup works for both spellings
            det_conf[name.replace("è", "e")] = conf
            det_conf[name.replace("e", "è")] = conf

        for cand in all_candidates:
            cipher_name = cand.get("cipher", "")
            conf = det_conf.get(cipher_name, 1.0 / 3.0)
            cand["_adjusted_score"] = cand["score"] * (BASE + WEIGHT * conf)

        # Sort by detection-weighted score; display original language score
        all_candidates.sort(key=lambda x: x["_adjusted_score"], reverse=True)

        for rank, cand in enumerate(all_candidates[:5], 1):
            key_label = cand.get("key", "—")
            if isinstance(key_label, int):
                key_label = f"{key_label} ({chr(ord('A') + key_label)})"
            score = cand["score"]
            css_class = "candidate-block best" if rank == 1 else "candidate-block"
            st.markdown(
                f"<div class='{css_class}'>"
                f"<span style='color:#ffd700'>#{rank}</span> &nbsp;"
                f"<span style='color:#00cfff'>Cipher: {cand['cipher']}</span> &nbsp;|&nbsp; "
                f"<span style='color:#8b949e'>Key: </span><span style='color:#e6edf3'>{key_label}</span>"
                f"&nbsp;|&nbsp;<span style='color:#8b949e'>Score: </span>"
                f"<span style='color:#00ff9f;font-weight:700'>{score:.4f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            preview = cand["plaintext"][:200] + ("…" if len(cand["plaintext"]) > 200 else "")
            st.markdown(
                f"<div class='plaintext-display' style='margin-bottom:12px'>{preview}</div>",
                unsafe_allow_html=True,
            )

        # Caesar brute-force table
        if caesar_results:
            st.markdown("---")
            with st.expander("📋 Full Caesar Brute-Force Results (all 26 keys)"):
                df_bf = pd.DataFrame([
                    {
                        "Key": r["key"],
                        "Letter": chr(ord("A") + r["key"]),
                        "Score": round(r["score"], 4),
                        "Plaintext (first 60 chars)": r["plaintext"][:60],
                    }
                    for r in caesar_results
                ])
                st.dataframe(df_bf, use_container_width=True, hide_index=True)

        # Vigenère IC profile
        if len(text_to_upper_alpha(ct_input)) >= 30:
            st.markdown("---")
            with st.expander("📐 Vigenère IC Profile (key-length estimation)"):
                ic_results = vigenere.estimate_key_length_ic(ct_input, max_key_len=15)
                df_ic = pd.DataFrame(ic_results)
                fig2, ax2 = plt.subplots(figsize=(10, 2.5))
                fig2.patch.set_facecolor("#0d1117")
                ax2.set_facecolor("#0d1117")
                ax2.bar(df_ic["key_length"], df_ic["avg_ic"], color="#00cfff", alpha=0.8)
                ax2.axhline(y=0.0667, color="#ff4466", linestyle="--", linewidth=1.2,
                            label="English IC (0.0667)")
                ax2.set_xlabel("Key length", color="#8b949e")
                ax2.set_ylabel("Avg IC", color="#8b949e")
                ax2.set_title("IC vs Key Length (peak = likely key length)", color="#00ff9f")
                ax2.tick_params(colors="#8b949e")
                for sp in ax2.spines.values(): sp.set_color("#30363d")
                ax2.legend(fontsize=8, labelcolor="white", framealpha=0.2)
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Frequency Analysis
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Frequency Analysis":
    st.markdown("# 📊 Frequency Analysis")
    st.markdown(
        "<p style='color:#8b949e'>Deep statistical analysis of letter distributions, "
        "bigrams, trigrams, and index of coincidence.</p>",
        unsafe_allow_html=True,
    )

    ct_input = st.text_area("Text to analyse", height=130,
                            placeholder="Paste ciphertext or plaintext here…")

    if st.button("📊 Analyse Frequencies", type="primary") and ct_input:
        valid, err = validate_input(ct_input, min_length=8)
        if not valid:
            st.error(err)
            st.stop()

        letters = text_to_upper_alpha(ct_input)

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total letters",   len(letters))
        m2.metric("IC",              f"{index_of_coincidence(letters):.5f}")
        m3.metric("Entropy",         f"{character_entropy(letters):.3f} bits")
        m4.metric("Distinct letters",f"{sum(1 for c in string.ascii_uppercase if c in letters)}/26")

        st.markdown("---")
        # Letter frequency chart
        fig = freq_bar_chart(ct_input)
        if fig:
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")
        col_a, col_b = st.columns(2)

        # Frequency table
        with col_a:
            st.markdown("#### Letter Frequencies")
            freq_dict = freq_ana.letter_frequencies(letters)
            df_freq = pd.DataFrame([
                {"Letter": ch, "Count": freq_ana.letter_counts(letters)[ch],
                 "Freq %": f"{freq_dict[ch]*100:.2f}",
                 "English %": f"{ENGLISH_LETTER_FREQ[ch]*100:.2f}",
                 "Δ": f"{(freq_dict[ch]-ENGLISH_LETTER_FREQ[ch])*100:+.2f}"}
                for ch in string.ascii_uppercase
            ])
            df_freq = df_freq[df_freq["Count"] > 0].sort_values("Freq %", ascending=False)
            st.dataframe(df_freq, use_container_width=True, hide_index=True, height=320)

        # Bigrams / Trigrams
        with col_b:
            st.markdown("#### Top Bigrams")
            bigrams = freq_ana.get_bigrams(letters, top_n=15)
            if bigrams:
                df_bg = pd.DataFrame(bigrams, columns=["Bigram", "Count"])
                st.dataframe(df_bg, use_container_width=True, hide_index=True, height=160)

            st.markdown("#### Top Trigrams")
            trigrams = freq_ana.get_trigrams(letters, top_n=15)
            if trigrams:
                df_tg = pd.DataFrame(trigrams, columns=["Trigram", "Count"])
                st.dataframe(df_tg, use_container_width=True, hide_index=True, height=160)

        st.markdown("---")
        st.markdown("#### Detailed Language Score Breakdown")
        detail = scorer.score_detailed(ct_input)
        bar_cols = st.columns(5)
        labels   = ["Freq Match", "Chi² Score", "Bigram", "Word Match", "Combined"]
        values   = [detail["freq_score"], detail["chi_score"],
                    detail["bigram_score"], detail["word_score"], detail["combined"]]
        for col, lab, val in zip(bar_cols, labels, values):
            col.metric(lab, f"{val:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ML Model
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 ML Model":
    st.markdown("# 🤖 Machine Learning Model")
    st.markdown(
        "<p style='color:#8b949e'>"
        "A Random Forest classifier trained on 2400 synthetic ciphertext samples "
        "(800 per class). All metrics below are measured on a held-out 20% test set — "
        "no numbers are hardcoded.</p>",
        unsafe_allow_html=True,
    )

    predictor = get_predictor()

    if not predictor.is_ready():
        st.error("Model not ready. Run `python -m ml.train` first.")
        st.stop()

    metrics = predictor.metrics

    # Summary metrics
    st.markdown("---")
    st.markdown("<h3 class='section-header'>📊 Training & Evaluation Metrics</h3>",
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Test Accuracy",  f"{metrics['accuracy']*100:.1f}%"),
                unsafe_allow_html=True)
    c2.markdown(metric_card("Precision",      f"{metrics['precision']*100:.1f}%"),
                unsafe_allow_html=True)
    c3.markdown(metric_card("Recall",         f"{metrics['recall']*100:.1f}%"),
                unsafe_allow_html=True)
    c4.markdown(metric_card("Train samples",  str(metrics.get("n_train", "—"))),
                unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # Confusion matrix
    with col_left:
        st.markdown("#### Confusion Matrix")
        cm = np.array(metrics["confusion_matrix"])
        class_names = metrics["class_names"]

        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        fig_cm.patch.set_facecolor("#0d1117")
        ax_cm.set_facecolor("#0d1117")
        im = ax_cm.imshow(cm, cmap="YlGn")
        ax_cm.set_xticks(range(len(class_names)))
        ax_cm.set_yticks(range(len(class_names)))
        ax_cm.set_xticklabels(class_names, color="#e6edf3", fontsize=9, rotation=15)
        ax_cm.set_yticklabels(class_names, color="#e6edf3", fontsize=9)
        ax_cm.set_xlabel("Predicted", color="#8b949e")
        ax_cm.set_ylabel("True",      color="#8b949e")
        ax_cm.set_title("Confusion Matrix (test set)", color="#00ff9f", fontsize=10)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax_cm.text(j, i, str(cm[i, j]),
                           ha="center", va="center",
                           color="black" if cm[i, j] > cm.max() * 0.5 else "white",
                           fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax_cm)
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    # Feature importances
    with col_right:
        st.markdown("#### Top Feature Importances")
        fi = metrics.get("feature_importance", {})
        top_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:12]
        names_fi  = [x[0] for x in top_fi]
        values_fi = [x[1] for x in top_fi]

        fig_fi, ax_fi = plt.subplots(figsize=(5, 4))
        fig_fi.patch.set_facecolor("#0d1117")
        ax_fi.set_facecolor("#0d1117")
        bars = ax_fi.barh(names_fi[::-1], values_fi[::-1],
                          color=["#ffd700" if "ic_" in n else "#00cfff" for n in names_fi[::-1]],
                          alpha=0.85)
        ax_fi.tick_params(colors="#e6edf3", labelsize=7)
        ax_fi.set_xlabel("Importance", color="#8b949e", fontsize=8)
        ax_fi.set_title("Random Forest Feature Importances", color="#00ff9f", fontsize=9)
        for sp in ax_fi.spines.values(): sp.set_color("#30363d")
        ax_fi.xaxis.grid(True, color="#21262d", linewidth=0.5)
        ax_fi.set_axisbelow(True)
        gold_patch = mpatches.Patch(color="#ffd700", label="IC-profile features (NEW)")
        blue_patch = mpatches.Patch(color="#00cfff", label="Original features")
        ax_fi.legend(handles=[gold_patch, blue_patch], fontsize=7, framealpha=0.2, labelcolor="white")
        plt.tight_layout()
        st.pyplot(fig_fi)
        plt.close(fig_fi)

    st.markdown("---")
    st.markdown("#### Full Classification Report")
    st.code(metrics.get("report", "Not available"), language="text")

    # Live prediction demo
    st.markdown("---")
    st.markdown("<h3 class='section-header'>🔮 Live ML Prediction</h3>",
                unsafe_allow_html=True)
    demo_text = st.text_area(
        "Enter any ciphertext to see the raw ML prediction probabilities",
        value=SAMPLE_CAESAR, height=100,
    )
    if demo_text:
        result = predictor.predict(demo_text)
        ranked = predictor.predict_ranked(demo_text)
        if result["predicted"] == "Unknown":
            st.warning("Text too short for ML prediction (< 10 letters).")
        else:
            st.markdown(f"**Predicted:** `{result['predicted']}` "
                        f"(confidence: `{result['confidence']*100:.1f}%`)")
            for r in ranked:
                st.markdown(
                    ic_bar(r["probability"], f"{r['cipher']:<15}"),
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown("#### Dataset Summary")
    data_path = os.path.join(PROJECT_ROOT, "data", "sample_dataset.csv")
    if os.path.exists(data_path):
        df_data = pd.read_csv(data_path)
        d1, d2, d3 = st.columns(3)
        d1.metric("Total samples",      len(df_data))
        d2.metric("Features per sample", str(len(df_data.columns) - 1))
        d3.metric("Classes",             df_data["label"].nunique())
        st.markdown("**Class distribution:**")
        st.dataframe(df_data["label"].value_counts().reset_index()
                     .rename(columns={"index":"Cipher","label":"Count"}),
                     use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: About
# ══════════════════════════════════════════════════════════════════════════════

elif page == "ℹ️ About":
    st.markdown("# ℹ️ About This Project")
    st.markdown(
        """
        ## AI-Assisted Cryptanalysis of Classical Ciphers
        **College Cybersecurity Project**

        ---

        ### Problem Statement
        Classical ciphers — Caesar, Vigenère, and Substitution — were once considered
        unbreakable. Today, they are vulnerable to statistical analysis. This project
        builds an automated cryptanalysis system that combines classical frequency
        analysis with modern machine learning to identify and break these ciphers.

        ---

        ### Educational Scope
        This is a **purely educational** laboratory focused on historical ciphers.
        It does **not** implement:
        - Password cracking for real systems
        - Attacks on modern cryptographic algorithms (AES, RSA, etc.)
        - Any credential theft, malware, or exploitation

        ---

        ### Technology Stack
        | Component | Technology |
        |-----------|-----------|
        | Language | Python 3.13 |
        | Web UI | Streamlit |
        | ML | scikit-learn (Random Forest) |
        | Numerics | NumPy, Pandas |
        | Visualization | Matplotlib |

        ---

        ### Key Scientific Decisions

        **Why IC-profile features?**
        A Caesar cipher produces the same IC value at every column-stride.
        A Vigenère cipher has a low IC at stride=1 but recovers to English IC
        at the true key length. The 8 new IC-profile features (feat[40–47]) capture
        this structural difference and are the primary reason the model achieves
        93.5% accuracy.

        **Why prose, not pangrams, for training?**
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG" contains all 26 letters
        nearly equally (IC ≈ 0.044). That falls inside the Vigenère IC range.
        Training on pangram-based text taught the model to classify Caesar
        as Vigenère. The fix was to use representative English prose (IC ≈ 0.065).

        **Why Random Forest?**
        - Handles non-linear decision boundaries (Caesar vs Substitution differ in
          frequency ordering, not just IC — a non-linear distinction)
        - Provides interpretable feature importances
        - Fast training on 2400 samples
        - No feature scaling required

        ---

        ### ML Methodology
        1. Generate 2400 synthetic ciphertext samples (800 per class)
        2. Extract 48-dimensional feature vector from each sample
        3. Train/test split: 80/20, stratified
        4. Train Random Forest (200 trees, balanced class weights)
        5. Evaluate on held-out test set — all reported numbers are measured
        """
    )

