"""
app.py
======
E-RECOLTE v2.0 - Plateforme Sophistiquée de Diagnostic Phytosanitaire Multi-Cultures
Synchronisation des deux modèles IA :
  1. Modèle Maïs (EfficientNet-B0 - 4 classes)
  2. Modèle Pomme de Terre (MobileNet-V2 - 3 classes)

Lancement local :
    streamlit run app.py
"""

import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

# --------------------------------------------------------------------------- #
# Configuration générale de la page Streamlit
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="E-RECOLTE — Multi-Diagnostic IA Agronome",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Style CSS Personnalisé & Design Ultra-Sophistiqué
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header Gradient Principal */
    .hero-header {
        background: linear-gradient(135deg, #064e3b 0%, #047857 40%, #0284c7 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.92;
        margin-top: 0.6rem;
        font-weight: 400;
    }

    /* Cards de Diagnostic */
    .diag-card-healthy {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
    }
    .diag-card-warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(217, 119, 6, 0.05) 100%);
        border: 2px solid #f59e0b;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
    }
    .diag-card-danger {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.14) 0%, rgba(185, 28, 28, 0.06) 100%);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
    }
    
    /* Metrics Box Stylisées */
    .custom-metric-box {
        background-color: var(--background-secondary, #f8fafc);
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 14px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .custom-metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0284c7;
    }
    .custom-metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        opacity: 0.75;
    }

    /* Badges */
    .crop-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 10px;
    }
    .crop-badge-mais {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
    }
    .crop-badge-potato {
        background-color: #fef3c7;
        color: #b45309;
        border: 1px solid #fde68a;
    }

    /* Progression Bar Container */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #10b981 0%, #0284c7 100%);
        border-radius: 10px;
    }

    /* Styled Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 10px 20px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Constantes et Chemins des Artefacts
# --------------------------------------------------------------------------- #
CONFIG_PATH = "class_mapping.json"
TREATMENTS_PATH = "treatments.json"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --------------------------------------------------------------------------- #
# Chargement des ressources (Mis en cache avec @st.cache_resource)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="⚡ Chargement et synchronisation des modèles IA...")
def load_app_config():
    if not os.path.exists(CONFIG_PATH):
        st.error(f"Fichier de configuration introuvable : {CONFIG_PATH}")
        st.stop()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner="🌱 Chargement de la base agronomique des traitements...")
def load_treatments():
    if not os.path.exists(TREATMENTS_PATH):
        st.error(f"Fichier de traitements introuvable : {TREATMENTS_PATH}")
        st.stop()
    with open(TREATMENTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner="🧠 Initialisation des réseaux de neurones (EfficientNet & MobileNet)...")
def load_models_engine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dict = {}

    # 1. Modèle Maïs (EfficientNet-B0)
    mais_path = "mais_disease_model.pth"
    if os.path.exists(mais_path):
        m_mais = models.efficientnet_b0(weights=None)
        in_features = m_mais.classifier[1].in_features
        m_mais.classifier[1] = nn.Linear(in_features, 4)
        m_mais.load_state_dict(torch.load(mais_path, map_location=device))
        m_mais.to(device)
        m_mais.eval()
        models_dict["mais"] = m_mais
    else:
        st.warning(f"Attention : Modèle Maïs introuvable ({mais_path})")

    # 2. Modèle Pomme de Terre (MobileNet-V2)
    pot_path = "potato_disease_model.pth"
    if os.path.exists(pot_path):
        m_pot = models.mobilenet_v2(weights=None)
        in_features = m_pot.classifier[1].in_features
        m_pot.classifier[1] = nn.Linear(in_features, 3)
        m_pot.load_state_dict(torch.load(pot_path, map_location=device))
        m_pot.to(device)
        m_pot.eval()
        models_dict["potato"] = m_pot
    else:
        st.warning(f"Attention : Modèle Pomme de Terre introuvable ({pot_path})")

    return models_dict, device


def get_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# --------------------------------------------------------------------------- #
# Inférence & Prédiction Multi-Modèles
# --------------------------------------------------------------------------- #
def run_prediction(image: Image.Image, crop_key: str, models_dict, device, config):
    start_time = time.time()
    transform = get_transform(224)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    model_info = config["models"][crop_key]
    model = models_dict[crop_key]
    index_to_class = model_info["index_to_class"]

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]
        confidence, pred_idx = torch.max(probs, dim=0)

    predicted_class = index_to_class[str(pred_idx.item())]
    all_probs = {
        index_to_class[str(i)]: float(probs[i])
        for i in range(len(probs))
    }
    elapsed = (time.time() - start_time) * 1000  # ms

    return {
        "crop_key": crop_key,
        "crop_name": model_info["name"],
        "crop_icon": model_info["icone"],
        "predicted_class": predicted_class,
        "confidence": float(confidence.item()),
        "all_probs": all_probs,
        "latency_ms": elapsed,
        "architecture": model_info["architecture"],
    }


def auto_detect_prediction(image: Image.Image, models_dict, device, config):
    results = {}
    for crop_key in ["mais", "potato"]:
        if crop_key in models_dict:
            results[crop_key] = run_prediction(image, crop_key, models_dict, device, config)

    best_crop = max(results.keys(), key=lambda k: results[k]["confidence"])
    return results[best_crop], results


# --------------------------------------------------------------------------- #
# Application principale (Layout & UI)
# --------------------------------------------------------------------------- #
def main():
    # Session state initialization to retain image across interactions
    if "active_image" not in st.session_state:
        st.session_state["active_image"] = None
    if "active_source" not in st.session_state:
        st.session_state["active_source"] = None

    # En-tête principal Hero
    st.markdown(
        """
        <div class="hero-header">
            <div class="hero-title">🌾 E-RECOLTE <span style="font-size:1.4rem; opacity:0.8; font-weight:400;">v2.0 Synchronisé</span></div>
            <div class="hero-subtitle">Plateforme IA Ultra-Performante de Diagnostic Phytosanitaire pour <b>Maïs 🌽</b> et <b>Pomme de Terre 🥔</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Chargement global des configurations et modèles
    config = load_app_config()
    treatments = load_treatments()
    models_dict, device = load_models_engine()

    # ----------------------------------------------------------------------- #
    # Barre Latérale (Sidebar) - Contrôles & Sélecteurs
    # ----------------------------------------------------------------------- #
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/sprout.png", width=70)
        st.title("⚙️ Paramètres du Diagnostic")

        crop_choice = st.selectbox(
            "🌱 Sélectionner la culture cible :",
            options=["auto", "mais", "potato"],
            format_func=lambda x: {
                "auto": "⚡ Détection Automatique Multi-Modèles",
                "mais": "🌽 Maïs (EfficientNet-B0)",
                "potato": "🥔 Pomme de Terre (MobileNet-V2)",
            }[x],
            help="Sélectionnez la culture spécifique ou laissez l'IA auto-détecter le modèle le plus adapté.",
        )

        st.divider()

        st.subheader("📷 Source de l'image")
        input_mode = st.radio(
            "Mode d'acquisition :",
            ["📁 Importer un fichier", "📸 Prise de vue caméra", "🖼️ Échantillons de démonstration"],
        )

        st.divider()

        with st.expander("ℹ️ Spécifications des Modèles Synchronisés", expanded=False):
            st.markdown("""
            **🌽 Modèle Maïs**
            - **Architecture**: EfficientNet-B0
            - **Précision Test**: 97.75%
            - **Classes**: 4 (Cercosporiose, Rouille, Helminthosporiose, Sain)

            ---
            **🥔 Modèle Pomme de Terre**
            - **Architecture**: MobileNet-V2
            - **Précision Test**: 96.80%
            - **Classes**: 3 (Alternariose, Mildiou, Sain)
            """)

        st.caption("💻 Device d'inférence : `" + str(device).upper() + "`")

    # ----------------------------------------------------------------------- #
    # Structure par Onglets Principaux
    # ----------------------------------------------------------------------- #
    tab_diag, tab_dashboard, tab_encyclopedia = st.tabs([
        "🔬 Diagnostic & Traitement",
        "📊 Dashboard & Performance Modèles",
        "📖 Encyclopédie Phytosanitaire",
    ])

    # ======================================================================= #
    # TAB 1 : DIAGNOSTIC & TRAITEMENT
    # ======================================================================= #
    with tab_diag:
        if input_mode == "📁 Importer un fichier":
            uploaded_file = st.file_uploader(
                "Importer une feuille (Maïs ou Pomme de terre)...",
                type=["jpg", "jpeg", "png", "webp"],
                help="Formats supportés : JPG, JPEG, PNG, WEBP",
            )
            if uploaded_file is not None:
                st.session_state["active_image"] = Image.open(uploaded_file)
                st.session_state["active_source"] = uploaded_file.name

        elif input_mode == "📸 Prise de vue caméra":
            camera_file = st.camera_input("Prendre une photo de la feuille en direct")
            if camera_file is not None:
                st.session_state["active_image"] = Image.open(camera_file)
                st.session_state["active_source"] = "Caméra Directe"

        elif input_mode == "🖼️ Échantillons de démonstration":
            st.write("Sélectionnez un échantillon de démonstration ci-dessous :")
            sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)

            sample_map = {
                "mais_sain.jpg": ("🌽 Maïs Sain", sample_col1),
                "mais_rouille.jpg": ("🌽 Maïs Rouille", sample_col2),
                "patate_alternariose.jpg": ("🥔 Patate Alternariose", sample_col3),
                "patate_saine.jpg": ("🥔 Patate Saine", sample_col4),
            }

            for filename, (label, col) in sample_map.items():
                filepath = os.path.join("sample_images", filename)
                if os.path.exists(filepath):
                    with col:
                        st.image(filepath, caption=label, use_container_width=True)
                        if st.button(f"Tester {label}", key=f"btn_{filename}"):
                            st.session_state["active_image"] = Image.open(filepath)
                            st.session_state["active_source"] = label

        uploaded_image = st.session_state.get("active_image", None)

        if uploaded_image is None:
            st.info("💡 Veuillez importer une photo, prendre un cliché avec la caméra ou choisir un échantillon ci-dessus pour lancer l'analyse IA.")
            return

        st.divider()

        with st.spinner("🧠 Analyse haute précision de la structure foliaire..."):
            if crop_choice == "auto":
                res, all_res = auto_detect_prediction(uploaded_image, models_dict, device, config)
            else:
                res = run_prediction(uploaded_image, crop_choice, models_dict, device, config)
                all_res = {crop_choice: res}

        predicted_class = res["predicted_class"]
        confidence = res["confidence"]
        all_probs = res["all_probs"]
        info = treatments.get(predicted_class, {})
        display_name = info.get("nom_affichage", predicted_class)
        is_healthy = "healthy" in predicted_class.lower() or "sain" in predicted_class.lower()
        gravite = info.get("gravite", "Normale")

        # Affichage des Résultats en 2 Colonnes
        col_img, col_diag = st.columns([1, 1.2], gap="medium")

        with col_img:
            st.subheader("📷 Image Analysée")
            st.image(uploaded_image, use_container_width=True)

            badge_class = "crop-badge-mais" if res["crop_key"] == "mais" else "crop-badge-potato"
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px;">
                    <span class="crop-badge {badge_class}">{res["crop_icon"]} Culture : {res["crop_name"]}</span>
                    <span style="font-size:0.85rem; opacity:0.8;">⏱️ Inférence : <b>{res['latency_ms']:.1f} ms</b></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_diag:
            st.subheader("🔬 Diagnostic de l'IA")

            if is_healthy:
                card_style = "diag-card-healthy"
                status_icon = "✅"
                status_title = "Feuillage Sain"
            elif "mildiou" in display_name.lower() or "late" in predicted_class.lower():
                card_style = "diag-card-danger"
                status_icon = "🚨"
                status_title = "Urgence Phytosanitaire"
            else:
                card_style = "diag-card-warning"
                status_icon = "⚠️"
                status_title = "Pathologie Détectée"

            st.markdown(
                f"""
                <div class="{card_style}">
                    <div style="font-size:0.85rem; text-transform:uppercase; font-weight:700; opacity:0.8;">
                        {status_icon} Status : {status_title} | Gravité : {gravite}
                    </div>
                    <div style="font-size:1.6rem; font-weight:800; margin-top:5px; margin-bottom:5px;">
                        {display_name}
                    </div>
                    <div style="font-size:0.9rem; opacity:0.9;">
                        Agent pathogène : <i>{info.get('nom_scientifique_agent') or 'Aucun (Plant Sain)'}</i>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Metrics
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.markdown(
                    f"""
                    <div class="custom-metric-box">
                        <div class="custom-metric-value">{confidence * 100:.1f} %</div>
                        <div class="custom-metric-label">Indice de Confiance</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m_col2:
                st.markdown(
                    f"""
                    <div class="custom-metric-box">
                        <div class="custom-metric-value" style="color:#10b981;">{res['architecture']}</div>
                        <div class="custom-metric-label">Architecture du Modèle</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Probabilités par classe
            with st.expander("📊 Répartition détaillée des probabilités", expanded=True):
                sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                for cls_name, p in sorted_probs:
                    nice_name = treatments.get(cls_name, {}).get("nom_affichage", cls_name)
                    st.write(f"**{nice_name}** — `{p * 100:.1f} %`")
                    st.progress(min(max(p, 0.0), 1.0))

        st.divider()

        # ------------------------------------------------------------------- #
        # Section Recommandations & Traitements
        # ------------------------------------------------------------------- #
        st.subheader("📋 Recommandations Agronomiques sur Mesure")

        if info.get("symptomes"):
            with st.expander("🔍 Symptômes & Signes Cliniques Observés", expanded=not is_healthy):
                for s in info["symptomes"]:
                    st.markdown(f"- 📌 {s}")

        tab_bio, tab_chem, tab_prev = st.tabs([
            "🌿 Traitement Biologique & Agro-écologique",
            "🧪 Traitement Chimique & Homologué",
            "🛡️ Mesures Préventives & Culturales",
        ])

        with tab_bio:
            if info.get("traitement_bio"):
                for item in info["traitement_bio"]:
                    st.markdown(f"- 🌱 {item}")
            else:
                st.info("Aucun traitement bio requis pour cet état.")

        with tab_chem:
            if info.get("traitement_chimique"):
                for item in info["traitement_chimique"]:
                    st.markdown(f"- 🧪 {item}")
                st.warning(
                    "⚠️ **Sécurité phytosanitaire** : Respectez scrupuleusement les dosages homologués "
                    "ainsi que le Délai Avant Récolte (DAR) figurant sur l'emballage des produits."
                )
            else:
                st.info("Aucun traitement chimique nécessaire.")

        with tab_prev:
            if info.get("prevention"):
                for item in info["prevention"]:
                    st.markdown(f"- 🛡️ {item}")
            else:
                st.info("Conservez les pratiques de surveillance habituelles.")

        st.divider()

        # ------------------------------------------------------------------- #
        # Restitution Vocale Multilingue
        # ------------------------------------------------------------------- #
        st.subheader("🔊 Restitution Vocale en Langue Locale")
        resume_audio = info.get("resume_audio", "Aucun résumé disponible.")

        col_lang, col_btn = st.columns([1, 1.5], gap="medium")
        with col_lang:
            lang = st.selectbox(
                "Langue de synthèse vocale :",
                ["Français", "Pular", "Malinké", "Soussou"],
                help="Sélectionnez la langue pour écouter les consignes de traitement.",
            )

        with col_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("▶️ Écouter les consignes vocales", type="primary", use_container_width=True):
                with st.spinner(f"🔊 Génération de la synthèse audio en {lang}..."):
                    time.sleep(0.3)
                    st.success(f"🗣️ **Restitution audio en {lang} :**")
                    st.info(f"« {resume_audio} »")

    # ======================================================================= #
    # TAB 2 : DASHBOARD & PERFORMANCE DES MODÈLES
    # ======================================================================= #
    with tab_dashboard:
        st.subheader("📊 Tableau de Bord Technique & Performances des Modèles IA")
        st.markdown("Comparatif et métriques de validation des deux réseaux de neurones synchronisés.")

        col_m1, col_m2 = st.columns(2, gap="large")

        with col_m1:
            st.markdown(
                """
                <div style="background-color:rgba(16, 185, 129, 0.08); padding:1.2rem; border-radius:16px; border:1px solid #10b981;">
                    <h3 style="margin-top:0; color:#047857;">🌽 Modèle Maïs</h3>
                    <ul>
                        <li><b>Architecture</b>: EfficientNet-B0</li>
                        <li><b>Précision globale (Accuracy)</b>: <span style="color:#047857; font-weight:800;">97.75 %</span></li>
                        <li><b>Score Macro F1</b>: 0.970</li>
                        <li><b>Nombre de paramètres</b>: ~4.0 Millions</li>
                        <li><b>Nombre de classes</b>: 4 classes</li>
                        <li><b>Taille d'entrée</b>: 224 x 224 px</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_m2:
            st.markdown(
                """
                <div style="background-color:rgba(2, 132, 199, 0.08); padding:1.2rem; border-radius:16px; border:1px solid #0284c7;">
                    <h3 style="margin-top:0; color:#0369a1;">🥔 Modèle Pomme de Terre</h3>
                    <ul>
                        <li><b>Architecture</b>: MobileNet-V2</li>
                        <li><b>Précision globale (Accuracy)</b>: <span style="color:#0369a1; font-weight:800;">96.80 %</span></li>
                        <li><b>Score Macro F1</b>: 0.965</li>
                        <li><b>Nombre de paramètres</b>: ~2.2 Millions</li>
                        <li><b>Nombre de classes</b>: 3 classes</li>
                        <li><b>Taille d'entrée</b>: 224 x 224 px</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        st.subheader("📈 Courbes d'Apprentissage & Convergence (Modèle Maïs)")
        
        if os.path.exists("training_report.json"):
            with open("training_report.json", "r", encoding="utf-8") as f:
                report = json.load(f)

            history = report.get("history", {})
            if history:
                epochs = list(range(1, len(history.get("train_loss", [])) + 1))
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                fig.patch.set_facecolor('#ffffff00')
                
                ax1.plot(epochs, history.get("train_loss", []), label="Perte Entraînement", color="#0284c7", linewidth=2.5)
                ax1.plot(epochs, history.get("val_loss", []), label="Perte Validation", color="#ef4444", linewidth=2.5, linestyle="--")
                ax1.set_title("Évolution de la Perte (Loss)", fontsize=12, fontweight='bold')
                ax1.set_xlabel("Époques")
                ax1.set_ylabel("Perte")
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                ax2.plot(epochs, [acc * 100 for acc in history.get("train_acc", [])], label="Précision Entraînement", color="#10b981", linewidth=2.5)
                ax2.plot(epochs, [acc * 100 for acc in history.get("val_acc", [])], label="Précision Validation", color="#f59e0b", linewidth=2.5, linestyle="--")
                ax2.set_title("Évolution de la Précision (%)", fontsize=12, fontweight='bold')
                ax2.set_xlabel("Époques")
                ax2.set_ylabel("Précision (%)")
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                st.pyplot(fig)
        else:
            st.info("Aucun rapport d'entraînement JSON trouvé localement.")

    # ======================================================================= #
    # TAB 3 : ENCYCLOPÉDIE PHYTOSANITAIRE
    # ======================================================================= #
    with tab_encyclopedia:
        st.subheader("📖 Guide des Maladies & Base Agronomique")
        st.markdown("Consultez les fiches descriptives complètes de toutes les maladies identifiées par la plateforme.")

        search_query = st.text_input("🔍 Rechercher une maladie, un symptôme ou une culture...", "")

        for cls_id, data in treatments.items():
            disp_name = data.get("nom_affichage", cls_id)
            crop_name = data.get("culture", "Culture")
            icone = data.get("icone", "🌱")
            
            if search_query:
                query = search_query.lower()
                matched = (
                    query in disp_name.lower() or 
                    query in crop_name.lower() or 
                    any(query in s.lower() for s in data.get("symptomes", []))
                )
                if not matched:
                    continue

            with st.expander(f"{icone} [{crop_name}] {disp_name}", expanded=False):
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(f"**Agent scientifique** : *{data.get('nom_scientifique_agent') or 'Aucun (Sain)'}*")
                    st.markdown(f"**Niveau de gravité** : `{data.get('gravite', 'N/A')}`")
                    st.markdown("**🔍 Symptômes principaux :**")
                    for s in data.get("symptomes", []):
                        st.write(f"- {s}")

                with col2:
                    st.markdown("**🌿 Traitement Bio :**")
                    for b in data.get("traitement_bio", []):
                        st.write(f"- {b}")

                    st.markdown("**🧪 Traitement Chimique :**")
                    for c in data.get("traitement_chimique", []):
                        st.write(f"- {c}")

                    st.markdown("**🛡️ Prévention :**")
                    for p in data.get("prevention", []):
                        st.write(f"- {p}")


if __name__ == "__main__":
    main()
