"""
DAiSEE Engagement Detection — Streamlit Demo
Loads exported .keras models + scaler/SVD from exported_models/
"""

import os, json, pickle, warnings, tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from huggingface_hub import hf_hub_download
from video_utils import extract_features_from_video

warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DAiSEE Engagement Detector",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES  = ['Very Low', 'Low', 'High', 'Very High']
CLASS_EMOJI  = ['😴', '😐', '🙂', '🤩']
CLASS_COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
MODELS_DIR   = Path(__file__).parent / 'exported_models'
TTA_RUNS     = 5

# --- Hugging Face Configuration ---
# REPLACE with your actual Repo ID (e.g., 'username/daisee-models')
HF_REPO_ID = "sujals1238/DAISEE"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #1565C0, #E91E63);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.subtitle {
    color: #888;
    font-size: 0.95rem;
    margin-bottom: 2rem;
    font-weight: 300;
}
.metric-card {
    background: #0f1117;
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #E91E63;
}
.metric-label {
    font-size: 0.78rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.prediction-box {
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.pred-label {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
}
.pred-conf {
    font-size: 1rem;
    color: #aaa;
    margin-top: 0.3rem;
}
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #555;
    border-bottom: 1px solid #1e2130;
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem 0;
}
.stButton > button {
    background: linear-gradient(135deg, #1565C0, #E91E63);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    padding: 0.6rem 2rem;
    font-size: 0.9rem;
    letter-spacing: 0.5px;
    width: 100%;
}
.stButton > button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)


# ── Load assets (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models from cloud…")
def load_all_models():
    """Load keras models + sklearn preprocessing objects."""
    try:
        import keras
        from keras.models import load_model
    except ImportError:
        st.error("Keras 3 not installed. Run: pip install keras")
        st.stop()

    def focal_loss(gamma=2.0, label_smoothing=0.05):
        import keras.ops as ops
        def _loss(y_true, y_pred):
            y_pred = ops.clip(y_pred, 1e-7, 1.0)
            n_cls  = ops.cast(ops.shape(y_true)[-1], "float32")
            y_sm   = y_true * (1 - label_smoothing) + label_smoothing / n_cls
            ce     = -ops.sum(y_sm * ops.log(y_pred), axis=-1)
            pt     = ops.sum(y_sm * y_pred, axis=-1)
            return ops.mean(ops.power(1 - pt, gamma) * ce)
        return _loss

    custom_obj = {'_loss': focal_loss()}
    models     = {}

    model_files = {
        '1D CNN':          '1D_CNN_individual.keras',
        '1D ResNet':       '1D_ResNet_individual.keras',
        'DepthwiseCNN':    'DepthwiseCNN_individual.keras',
        '1D DenseNet':     '1D_DenseNet_individual.keras',
        '1D InceptionNet': '1D_InceptionNet_individual.keras',
        'AttentionResNet': 'AttentionResNet_individual.keras',
        'AEClassifier':    'AEClassifier_individual.keras',
        'UNetClassifier':  'UNetClassifier_individual.keras',
    }

    ae_models = {'AEClassifier', 'UNetClassifier'}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, fname in model_files.items():
        fpath = MODELS_DIR / fname
        if not fpath.exists():
            try:
                print(f"--- [DEPLOYMENT] Downloading {fname} from Hugging Face... ---")
                downloaded_path = hf_hub_download(repo_id=HF_REPO_ID, filename=fname)
                import shutil
                shutil.copy(downloaded_path, fpath)
                print(f"--- [DEPLOYMENT] Successfully saved {fname} ---")
            except Exception as e:
                print(f"--- [DEPLOYMENT] Failed to download {fname}: {e} ---")
                models[name] = {'model': None, 'is_ae': False, 'loaded': False, 
                                'error': f"HF Download failed: {e}"}
                continue

        if fpath.exists():
            try:
                m = load_model(str(fpath), custom_objects=custom_obj)
                models[name] = {'model': m, 'is_ae': name in ae_models, 'loaded': True}
            except Exception as e:
                models[name] = {'model': None, 'is_ae': False,
                                'loaded': False, 'error': str(e)}
        else:
            models[name] = {'model': None, 'is_ae': False,
                            'loaded': False, 'error': f'{fname} not found'}

    # Preprocessing
    scaler, svd = None, None
    pre_path    = MODELS_DIR / 'preprocessing.pkl'
    scaler_path = MODELS_DIR / 'scaler.pkl'
    svd_path    = MODELS_DIR / 'svd.pkl'
    meta_path   = MODELS_DIR / 'metadata.json'

    # Try loading from combined preprocessing.pkl first
    if pre_path.exists():
        try:
            with open(pre_path, 'rb') as f:
                prep_data = pickle.load(f)
            if isinstance(prep_data, dict):
                scaler = prep_data.get('scaler')
                svd    = prep_data.get('svd')
            else:
                scaler = prep_data # Fallback if it's just the scaler
        except Exception as e:
            st.error(f"Error loading preprocessing.pkl: {e}")

    # Fallback to individual files
    if scaler is None and scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    if svd is None and svd_path.exists():
        with open(svd_path, 'rb') as f:
            svd = pickle.load(f)

    # Metadata
    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    return models, scaler, svd, metadata


def preprocess(raw_features: np.ndarray, scaler, svd) -> np.ndarray:
    """Scale → SVD → reshape for Conv1D."""
    x = scaler.transform(raw_features.reshape(1, -1))
    x = svd.transform(x)
    return x.reshape(1, -1, 1).astype(np.float32)


def predict_single(model_info: dict, x_proc: np.ndarray,
                   tta_runs: int = TTA_RUNS) -> np.ndarray:
    """TTA prediction; handles AE dual-output models."""
    m      = model_info['model']
    is_ae  = model_info['is_ae']
    probs  = []
    for _ in range(tta_runs):
        x_aug = x_proc + np.random.normal(0, 0.01, x_proc.shape).astype(np.float32)
        out   = m.predict(x_aug, verbose=0)
        p     = out[0] if is_ae else out   # AE returns (clf, recon)
        probs.append(p[0])
    return np.stack(probs).mean(0)         # (4,)


def ensemble_predict(models: dict, x_proc: np.ndarray) -> np.ndarray:
    """Average softmax across all loaded models."""
    all_probs = []
    for info in models.values():
        if info['loaded']:
            all_probs.append(predict_single(info, x_proc))
    return np.stack(all_probs).mean(0)     # (4,)


# ── Plotting helpers ──────────────────────────────────────────────────────────
def plot_prob_bar(probs: np.ndarray, title: str = '') -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#0f1117')
    bars = ax.barh(CLASS_NAMES, probs * 100,
                   color=CLASS_COLORS, edgecolor='none', height=0.55)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{p*100:.1f}%', va='center', ha='left',
                color='white', fontsize=10, fontweight='bold')
    ax.set_xlim(0, 115)
    ax.set_xlabel('Probability (%)', color='#888', fontsize=9)
    ax.tick_params(colors='#aaa', labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.label.set_color('#888')
    ax.tick_params(axis='x', colors='#555')
    ax.tick_params(axis='y', colors='#ccc')
    if title:
        ax.set_title(title, color='#ccc', fontsize=10,
                     fontfamily='monospace', pad=8)
    plt.tight_layout()
    return fig


def plot_model_comparison(all_probs: dict) -> plt.Figure:
    names = list(all_probs.keys())
    preds = [np.argmax(p) for p in all_probs.values()]
    confs = [np.max(p) * 100 for p in all_probs.values()]

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.55)))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#0f1117')

    colors = [CLASS_COLORS[p] for p in preds]
    bars   = ax.barh(names, confs, color=colors, edgecolor='none', height=0.6)
    for bar, pred, conf in zip(bars, preds, confs):
        ax.text(bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f'{CLASS_NAMES[pred]}  {conf:.1f}%',
                va='center', ha='left', color='white',
                fontsize=9, fontweight='bold')

    ax.set_xlim(0, 130)
    ax.set_xlabel('Confidence (%)', color='#888', fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors='#aaa', labelsize=9)
    ax.set_title('Per-model predictions', color='#ccc',
                 fontsize=10, fontfamily='monospace', pad=8)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  APP LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="main-title">DAiSEE Engagement Detector</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deep Learning ensemble — '
            'OpenFace features → 4-class engagement prediction</div>',
            unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
models, scaler, svd, metadata = load_all_models()
loaded_count = sum(1 for v in models.values() if v['loaded'])

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    selected_model = st.selectbox(
        "Primary model",
        options=['Ensemble (all models)'] + [k for k, v in models.items() if v['loaded']],
    )
    use_tta = st.toggle("Test-Time Augmentation", value=True)
    tta_n   = st.slider("TTA runs", 1, 10, TTA_RUNS,
                        disabled=not use_tta)

    st.divider()
    st.markdown("### 📦 Model Status")
    for name, info in models.items():
        icon = "✅" if info['loaded'] else "❌"
        st.markdown(f"`{icon}` {name}")

    st.divider()
    if metadata:
        st.markdown("### 📊 Training Info")
        if 'best_model' in metadata:
            st.markdown(f"**Best model:** {metadata['best_model']}")
        if 'export_date' in metadata:
            st.markdown(f"**Exported:** {metadata['export_date'][:10]}")
        if 'n_bags' in metadata:
            st.markdown(f"**Bags:** {metadata['n_bags']}")

# ── Status bar ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Models loaded", f"{loaded_count} / {len(models)}")
with c2:
    pp = scaler is not None and svd is not None
    st.metric("Preprocessing", "✅ Ready" if pp else "❌ Missing")
with c3:
    st.metric("Engagement classes", "4")
with c4:
    st.metric("TTA runs", tta_n if use_tta else "Off")

st.divider()

# ── Input tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🎥 Video Upload", "📁 Feature File", "✏️ Manual Input", "🎲 Random Sample"])

# ── VIDEO UPLOAD TAB ─────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Process Raw Video")
    st.info("Upload a video file (MP4/AVI) to extract facial features and classify engagement.")
    
    video_file = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov'])
    
    if video_file:
        st.video(video_file)
        
        if st.button("🚀 Process & Predict", key="btn_video"):
            if scaler is None or svd is None:
                st.error("Preprocessing objects missing. Models must be loaded first.")
            else:
                with st.spinner("Extracting features (this may take a minute)..."):
                    # Save to temp file for MediaPipe
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                        tmp.write(video_file.read())
                        tmp_path = tmp.name
                    
                    try:
                        raw_features = extract_features_from_video(tmp_path)
                        os.unlink(tmp_path) # Cleanup
                        
                        if raw_features is not None:
                            st.success("Features extracted successfully!")
                            
                            # Inference logic (same as upload tab)
                            x = preprocess(raw_features, scaler, svd)
                            n = tta_n if use_tta else 1
                            
                            if selected_model == 'Ensemble (all models)':
                                all_probs = {}
                                for name, info in models.items():
                                    if info['loaded']:
                                        all_probs[name] = predict_single(info, x, n)
                                probs = np.stack(list(all_probs.values())).mean(0)
                            else:
                                info = models[selected_model]
                                probs = predict_single(info, x, n)
                                all_probs = {selected_model: probs}

                            pred_idx  = int(np.argmax(probs))
                            pred_conf = float(probs[pred_idx]) * 100

                            col_v1, col_v2 = st.columns([1, 1])
                            with col_v1:
                                st.markdown(
                                    f'<div class="prediction-box" style="background:{CLASS_COLORS[pred_idx]}22;'
                                    f'border:2px solid {CLASS_COLORS[pred_idx]}">'
                                    f'<div style="font-size:3rem">{CLASS_EMOJI[pred_idx]}</div>'
                                    f'<div class="pred-label" style="color:{CLASS_COLORS[pred_idx]}">'
                                    f'{CLASS_NAMES[pred_idx]}</div>'
                                    f'<div class="pred-conf">Prediction — {pred_conf:.1f}% confidence</div>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                            with col_v2:
                                st.pyplot(plot_prob_bar(probs, 'Class probabilities'))
                                
                            if len(all_probs) > 1:
                                st.markdown('<div class="section-header">Per-model breakdown</div>',
                                            unsafe_allow_html=True)
                                st.pyplot(plot_model_comparison(all_probs))
                        else:
                            st.error("No faces detected in the video.")
                    except Exception as e:
                        st.error(f"Error processing video: {e}")
                        if os.path.exists(tmp_path): os.unlink(tmp_path)
with tab2:
    st.markdown("Upload a `.npy` or `.csv` file containing a single "
                "**pre-extracted OpenFace feature vector** (2836-dim or raw).")
    uploaded = st.file_uploader("Feature file", type=['npy', 'csv', 'txt'])
    run_upload = False
    raw_features_upload = None

    if uploaded:
        try:
            if uploaded.name.endswith('.npy'):
                arr = np.load(uploaded, allow_pickle=True)
            else:
                arr = np.loadtxt(uploaded, delimiter=',')

            arr = arr.flatten()
            st.success(f"Loaded — shape: {arr.shape}")
            raw_features_upload = arr
            run_upload = st.button("🔍 Predict", key='btn_upload')
        except Exception as e:
            st.error(f"Could not load file: {e}")

    if run_upload and raw_features_upload is not None:
        if scaler is None or svd is None:
            st.error("Preprocessing objects (scaler.pkl / svd.pkl) not found "
                     "in exported_models/. See README.")
        else:
            with st.spinner("Running inference…"):
                x = preprocess(raw_features_upload, scaler, svd)
                n = tta_n if use_tta else 1

                if selected_model == 'Ensemble (all models)':
                    all_probs = {}
                    for name, info in models.items():
                        if info['loaded']:
                            all_probs[name] = predict_single(info, x, n)
                    probs = np.stack(list(all_probs.values())).mean(0)
                else:
                    info = models[selected_model]
                    probs = predict_single(info, x, n)
                    all_probs = {selected_model: probs}

            pred_idx  = int(np.argmax(probs))
            pred_conf = float(probs[pred_idx]) * 100

            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.markdown(
                    f'<div class="prediction-box" style="background:{CLASS_COLORS[pred_idx]}22;'
                    f'border:2px solid {CLASS_COLORS[pred_idx]}">'
                    f'<div style="font-size:3rem">{CLASS_EMOJI[pred_idx]}</div>'
                    f'<div class="pred-label" style="color:{CLASS_COLORS[pred_idx]}">'
                    f'{CLASS_NAMES[pred_idx]}</div>'
                    f'<div class="pred-conf">Engagement — {pred_conf:.1f}% confidence</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with col_b:
                st.pyplot(plot_prob_bar(probs, 'Class probabilities'))

            if len(all_probs) > 1:
                st.markdown('<div class="section-header">Per-model breakdown</div>',
                            unsafe_allow_html=True)
                st.pyplot(plot_model_comparison(all_probs))

# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("Paste comma-separated feature values (2836 dimensions — "
                "mean + std + min + max of 709 OpenFace features).")
    text_input = st.text_area("Feature vector", height=120,
                              placeholder="0.123, -0.456, 0.789, …")
    run_manual = st.button("🔍 Predict", key='btn_manual')

    if run_manual and text_input.strip():
        try:
            arr = np.array([float(x) for x in text_input.split(',')])
            if scaler is None or svd is None:
                st.error("Preprocessing objects missing.")
            else:
                with st.spinner("Running inference…"):
                    x = preprocess(arr, scaler, svd)
                    n = tta_n if use_tta else 1

                    if selected_model == 'Ensemble (all models)':
                        all_probs = {}
                        for name, info in models.items():
                            if info['loaded']:
                                all_probs[name] = predict_single(info, x, n)
                        probs = np.stack(list(all_probs.values())).mean(0)
                    else:
                        info  = models[selected_model]
                        probs = predict_single(info, x, n)
                        all_probs = {selected_model: probs}

                pred_idx  = int(np.argmax(probs))
                pred_conf = float(probs[pred_idx]) * 100
                st.markdown(
                    f'<div class="prediction-box" style="background:{CLASS_COLORS[pred_idx]}22;'
                    f'border:2px solid {CLASS_COLORS[pred_idx]}">'
                    f'<div style="font-size:3rem">{CLASS_EMOJI[pred_idx]}</div>'
                    f'<div class="pred-label" style="color:{CLASS_COLORS[pred_idx]}">'
                    f'{CLASS_NAMES[pred_idx]}</div>'
                    f'<div class="pred-conf">Engagement — {pred_conf:.1f}% confidence</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.pyplot(plot_prob_bar(probs, 'Class probabilities'))
        except Exception as e:
            st.error(f"Parse error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("Generate a **random synthetic feature vector** to test "
                "the pipeline end-to-end without real data.")
    seed = st.number_input("Random seed", value=42, min_value=0)
    run_random = st.button("🎲 Generate & Predict", key='btn_random')

    if run_random:
        if scaler is None or svd is None:
            st.error("Preprocessing objects missing.")
        else:
            rng = np.random.default_rng(int(seed))
            # Simulate realistic OpenFace 4-stat feature distribution
            raw = rng.normal(0, 1, 2836).astype(np.float32)
            st.info(f"Generated random vector — shape: {raw.shape}  seed: {seed}")

            with st.spinner("Running inference…"):
                x = preprocess(raw, scaler, svd)
                n = tta_n if use_tta else 1

                all_probs = {}
                for name, info in models.items():
                    if info['loaded']:
                        all_probs[name] = predict_single(info, x, n)
                ens_probs = np.stack(list(all_probs.values())).mean(0)

            pred_idx  = int(np.argmax(ens_probs))
            pred_conf = float(ens_probs[pred_idx]) * 100

            col_r1, col_r2 = st.columns([1, 1])
            with col_r1:
                st.markdown(
                    f'<div class="prediction-box" style="background:{CLASS_COLORS[pred_idx]}22;'
                    f'border:2px solid {CLASS_COLORS[pred_idx]}">'
                    f'<div style="font-size:3rem">{CLASS_EMOJI[pred_idx]}</div>'
                    f'<div class="pred-label" style="color:{CLASS_COLORS[pred_idx]}">'
                    f'{CLASS_NAMES[pred_idx]}</div>'
                    f'<div class="pred-conf">Ensemble — {pred_conf:.1f}% confidence</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with col_r2:
                st.pyplot(plot_prob_bar(ens_probs, 'Ensemble probabilities'))

            st.markdown('<div class="section-header">All models</div>',
                        unsafe_allow_html=True)
            st.pyplot(plot_model_comparison(all_probs))

            # Show raw prob table
            with st.expander("Raw probability table"):
                rows = []
                for name, p in all_probs.items():
                    rows.append({'Model': name,
                                 **{c: f'{v*100:.2f}%'
                                    for c, v in zip(CLASS_NAMES, p)},
                                 'Prediction': CLASS_NAMES[np.argmax(p)]})
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#444;font-size:0.8rem;'>"
    "DAiSEE Engagement Detection · MIT Academy of Engineering · "
    "Bagging Ensemble of 8 Deep Learning Architectures"
    "</div>",
    unsafe_allow_html=True
)
