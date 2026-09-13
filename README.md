# Brain Tumor Detection — Web Application

A Flask web application that lets users upload a brain MRI image and predicts, using a VGG16-based CNN model, whether it shows a glioma, meningioma, pituitary tumor, or no tumor.

## Overview

- Upload an MRI image through a simple web interface
- Prediction using a trained VGG16 model (`mon_model_vgg16.h5`)
- Displays the result with confidence score, a definition of the detected tumor type, and recommended treatments

## Training Data

The model was trained on the **Brain Tumor MRI Dataset** available on Kaggle:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

## Pretrained Model (`mon_model_vgg16.h5`)

The model file is too large to host directly on GitHub. Download it here:

https://drive.google.com/file/d/1fnDVF8OVfA4B0cLzF9nKqv2NrqX9-6vO/view?usp=drive_link

Once downloaded, place `mon_model_vgg16.h5` at the root of the project (same level as `app.py`).

> The model expects **240x240 pixel** images (RGB), and was trained with **Keras 3.9.0** on a TensorFlow backend.

## Installation

1. Clone the repo:
   ```bash
   git clone <YOUR_REPO_URL>
   cd brain-tumor-detection
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download `mon_model_vgg16.h5` (see section above) and place it at the root of the project.

5. Make sure the `static/uploads/` folder exists (it should contain a `.gitkeep` file).

## Running the Application

```bash
python app.py
```

Then open a browser at `http://127.0.0.1:5000`.

On Windows, you can also double-click `lancer_app.bat` (after creating and activating your own `venv` as described above).

## Technologies

Python, Flask, TensorFlow / Keras (VGG16), Pillow, HTML/CSS (Jinja2)

## Disclaimer

This project is an academic/portfolio exercise. It is not a validated medical diagnostic tool and should never replace the advice of a healthcare professional.
