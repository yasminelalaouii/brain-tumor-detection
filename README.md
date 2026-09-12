# Brain Tumor Detection — Application Web

Application web (Flask) qui permet d'uploader une image IRM du cerveau et de prédire, à l'aide d'un modèle CNN basé sur VGG16, s'il s'agit d'un gliome, d'un méningiome, d'une tumeur pituitaire, ou d'une absence de tumeur.

## Aperçu

- Upload d'une image IRM via une interface web simple
- Prédiction avec un modèle VGG16 entraîné (`mon_model_vgg16.h5`)
- Affichage du résultat avec le taux de confiance, une définition du type de tumeur détecté, et les traitements recommandés

## Données d'entraînement

Le modèle a été entraîné sur le dataset **Brain Tumor MRI Dataset** disponible sur Kaggle :
 https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

## Modèle pré-entraîné (`mon_model_vgg16.h5`)

Le fichier du modèle est trop volumineux pour être hébergé directement sur GitHub. Télécharge-le ici :

 https://drive.google.com/file/d/1fnDVF8OVfA4B0cLzF9nKqv2NrqX9-6vO/view?usp=drive_link

Une fois téléchargé, place `mon_model_vgg16.h5` à la racine du projet (au même niveau que `app.py`).

> Le modèle attend des images en **240x240 pixels** (RGB), a été entraîné avec **Keras 3.9.0** sur un backend TensorFlow.

## Installation

1. Cloner le repo :
   ```bash
   git clone <URL_DE_TON_REPO>
   cd brain-tumor-detection
   ```

2. Créer un environnement virtuel :
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

4. Télécharger `mon_model_vgg16.h5` (voir section ci-dessus) et le placer à la racine du projet.

5. Vérifier que le dossier `static/uploads/` existe (il doit contenir un fichier `.gitkeep`).

## Lancer l'application

```bash
python app.py
```

Puis ouvrir un navigateur sur `http://127.0.0.1:5000`.

Sous Windows, tu peux aussi double-cliquer sur `lancer_app.bat` (après avoir créé et activé ton propre `venv` comme indiqué ci-dessus).

## Technologies

Python, Flask, TensorFlow / Keras (VGG16), Pillow, HTML/CSS (Jinja2)

## Avertissement

Ce projet est un exercice académique / portfolio. Il ne s'agit pas d'un outil de diagnostic médical validé et ne doit en aucun cas remplacer l'avis d'un professionnel de santé.
