# 🌾 E-RECOLTE v2.0 — Plateforme IA de Diagnostic Phytosanitaire Multi-Cultures

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**E-RECOLTE v2.0** est une plateforme web interactive basée sur la vision par ordinateur et le Deep Learning, conçue pour assister les agriculteurs, techniciens agricoles et agronomes dans le diagnostic rapide des maladies du **Maïs 🌽** et de la **Pomme de Terre 🥔**, avec préconisations de traitements adaptés et restitution vocale en langues locales.

---

## 🌟 Fonctionnalités Principales

- 🧠 **Moteur Multi-Modèles Synchronisé** :
  - **🌽 Modèle Maïs** : Basé sur **EfficientNet-B0** (*Précision globale : 97.75%*).
  - **🥔 Modèle Pomme de Terre** : Basé sur **MobileNet-V2** (*Précision globale : 96.80%*).
  - **⚡ Auto-détection** : Sélection automatique du modèle pertinent selon l'image fournie.
- 🔬 **Diagnostic Visuel Instantané** :
  - Import de fichiers (JPG, PNG), capture caméra en direct ou sélection parmi des échantillons de démonstration.
  - Cartes de statut colorées avec indices de confiance et répartition des probabilités par classe.
- 📋 **Recommandations Agronomiques sur Mesure** :
  - **🌿 Traitement Biologique** (agro-écologie, extraits naturels, rotation).
  - **🧪 Traitement Chimique** (molécules homologuées et respect des Délais Avant Récolte - DAR).
  - **🛡️ Prévention & Pratiques Culturales**.
- 🔊 **Restitution Vocale en Langues Locales** :
  - Synthèse et accompagnement vocal disponible en **Français**, **Pular**, **Malinké** et **Soussou**.
- 📊 **Tableau de Bord & Encyclopédie** :
  - Comparatif technique des réseaux de neurones et visualisation des courbes d'apprentissage (Loss/Accuracy).
  - Guide complet et moteur de recherche intégré pour toutes les pathologies référencées.

---

## 📊 Performance des Modèles Synchronisés

| Culture | Architecture | Nb. Paramètres | Précision (Val Acc) | Score Macro F1 | Classes Référencées |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **🌽 Maïs** | `EfficientNet-B0` | ~4.0M | **97.75 %** | **0.970** | Cercosporiose, Rouille, Helminthosporiose, Sain |
| **🥔 Pomme de Terre** | `MobileNet-V2` | ~2.2M | **96.80 %** | **0.965** | Alternariose, Mildiou, Sain |

---

## 🚀 Installation & Lancement Local

### 1. Cloner le dépôt GitHub
```bash
git clone https://github.com/Slymanidiallo/erecolte-plant-disease-detection.git
cd erecolte-plant-disease-detection
```

### 2. Créer et activer un environnement virtuel (optionnel mais recommandé)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / MacOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Lancer l'application Streamlit
```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`.

---

## 📁 Structure du Projet

```text
erecolte-plant-disease-detection/
├── app.py                     # Code principal de l'application Streamlit v2.0
├── class_mapping.json         # Mappings et configurations des modèles (Maïs & Patate)
├── treatments.json            # Base de connaissances agronomiques et recommandations
├── mais_disease_model.pth     # Poids entraînés du modèle Maïs (EfficientNet-B0)
├── potato_disease_model.pth   # Poids entraînés du modèle Pomme de terre (MobileNet-V2)
├── training_report.json       # Rapport de validation et historique d'entraînement
├── training_curves.png        # Courbes graphiques de convergence
├── requirements.txt           # Dépendances Python nécessaires
├── sample_images/             # Échantillons d'images de test pour la démo
└── README.md                  # Documentation du projet
```

---

## 👤 Auteur & Contact

**Thierno S. Diallo**  
📧 Email : [thierno.s.diallo@aims-senegal.org](mailto:thierno.s.diallo@aims-senegal.org)  
🐙 GitHub : [@Slymanidiallo](https://github.com/Slymanidiallo)

---
*Projet développé dans le cadre de l'initiative E-RECOLTE — Intelligence Artificielle appliquée à l'Agriculture Durable.*
