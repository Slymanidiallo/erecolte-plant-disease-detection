# E-RECOLTE — Plant Disease Detection | Détection des maladies des plantes

> **EN:** Streamlit and PyTorch application for image-based plant disease classification and agronomic information.  
> **FR :** Application Streamlit et PyTorch de classification des maladies des plantes à partir d'images et de consultation d'informations agronomiques.

## 🇬🇧 English

### Overview
E-RECOLTE is an image-classification prototype designed to assist with the visual identification of plant diseases. The repository contains a Streamlit application, PyTorch model weights, class mappings, a treatment-information file and a training report.

### Current project components
- `app.py`: Streamlit user interface and model inference.
- `class_mapping.json`: class labels and model configuration.
- `treatments.json`: disease descriptions and agronomic guidance.
- `mais_disease_model.pth`: corn model weights.
- `potato_disease_model.pth`: potato model weights.
- `training_report.json`: recorded training and validation metrics.
- `training_curves.png`: training-curve visualization.
- `sample_images/`: sample images, where provided.
- `requirements.txt`: Python dependencies.

### Application workflow
1. Launch the Streamlit application.
2. Upload a plant-leaf image (JPG, JPEG or PNG).
3. The application preprocesses the image and runs the configured PyTorch model.
4. Review the predicted class, confidence and class-probability breakdown.
5. Consult the disease information and the biological, chemical and prevention guidance available in the local knowledge file.

### Recorded evaluation
The committed `training_report.json` records the following results for a four-class corn classification run:

| Metric | Recorded value |
|---|---:|
| Validation accuracy | 97.75% |
| Macro F1-score | 0.9703 |
| Validation samples | 577 |
| Epochs recorded | 10 |

These values describe the committed experiment only. They are not a guarantee of performance on new farms, cameras, lighting conditions or field images. The README does not claim equivalent performance for the potato model because a corresponding evaluation report is not present in the repository root.

### Installation
```bash
git clone https://github.com/Slymanidiallo/erecolte-plant-disease-detection.git
cd erecolte-plant-disease-detection
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The model weights and JSON configuration files must remain at the paths expected by the application.

### Limitations and responsible use
- Predictions are experimental and may be wrong; consult an agronomist before acting on a diagnosis.
- Treatment advice must be checked against local regulations, product labels, crop conditions and pre-harvest intervals.
- The code describes the voice interface as a simulation; real text-to-speech in local languages is not implemented in this version.
- Do not interpret the model's confidence as a calibrated probability of disease.

## 🇫🇷 Français

### Présentation
E-RECOLTE est un prototype de classification d'images destiné à aider à l'identification visuelle des maladies des plantes. Le dépôt contient une application Streamlit, des poids de modèles PyTorch, des correspondances de classes, un fichier d'informations agronomiques et un rapport d'entraînement.

### Composants actuels
- `app.py` : interface Streamlit et inférence du modèle.
- `class_mapping.json` : libellés des classes et configuration du modèle.
- `treatments.json` : descriptions des maladies et conseils agronomiques.
- `mais_disease_model.pth` : poids du modèle pour le maïs.
- `potato_disease_model.pth` : poids du modèle pour la pomme de terre.
- `training_report.json` : métriques enregistrées d'entraînement et de validation.
- `training_curves.png` : visualisation des courbes d'apprentissage.
- `sample_images/` : images d'exemple, lorsqu'elles sont présentes.
- `requirements.txt` : dépendances Python.

### Fonctionnement
1. Lancer l'application Streamlit.
2. Importer une image de feuille (JPG, JPEG ou PNG).
3. L'application prétraite l'image et exécute le modèle PyTorch configuré.
4. Examiner la classe prédite, le score de confiance et les probabilités par classe.
5. Consulter les informations sur la maladie et les conseils biologiques, chimiques et préventifs disponibles dans le fichier local.

### Résultats enregistrés
Le fichier `training_report.json` contient les résultats suivants pour une expérience de classification du maïs en quatre classes :

| Métrique | Valeur enregistrée |
|---|---:|
| Précision de validation (accuracy) | 97,75 % |
| F1-score macro | 0,9703 |
| Échantillons de validation | 577 |
| Époques enregistrées | 10 |

Ces valeurs décrivent uniquement l'expérience enregistrée dans le dépôt. Elles ne garantissent pas les mêmes performances sur de nouvelles exploitations, caméras, conditions d'éclairage ou images de terrain. Le README ne revendique pas de performance équivalente pour le modèle pomme de terre, car aucun rapport d'évaluation correspondant n'est présent à la racine du dépôt.

### Installation
```bash
git clone https://github.com/Slymanidiallo/erecolte-plant-disease-detection.git
cd erecolte-plant-disease-detection
python -m venv .venv
# Windows : .venv\Scripts\activate
# macOS/Linux : source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Les poids des modèles et les fichiers JSON doivent rester aux emplacements attendus par l'application.

### Limites et utilisation responsable
- Les prédictions sont expérimentales et peuvent être erronées ; demandez l'avis d'un agronome avant toute intervention.
- Les conseils de traitement doivent être vérifiés selon la réglementation locale, les étiquettes des produits, les conditions de culture et les délais avant récolte.
- Le code présente l'interface vocale comme une simulation ; la synthèse vocale réelle en langues locales n'est pas implémentée dans cette version.
- Le score de confiance du modèle ne doit pas être interprété comme une probabilité de maladie parfaitement calibrée.
