from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

from PIL import Image

app = Flask(__name__)
model = load_model("mon_model_vgg16.h5")

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class_info = {
    0: {
        "label": "Glioma Tumor",
        "definition": """
        Gliomas are a group of tumors that arise from glial cells, which support and protect the neurons in the brain or spinal cord. 
        Gliomas can be either benign or malignant, with malignant forms being more aggressive and requiring immediate treatment. 
        Gliomas often cause symptoms like headaches, seizures, or motor dysfunction due to their growth within the brain tissue.
        """,
        "treatment": """
        Treatment for gliomas typically involves a combination of surgery, radiation therapy, and chemotherapy. 
        Surgery is the primary method for removing the tumor, though it can be complex depending on the tumor's location. 
        Radiation therapy helps to target any remaining tumor cells, while chemotherapy may be used to prevent further tumor growth. 
        A multidisciplinary team, including neurosurgeons, oncologists, and radiation therapists, usually coordinates treatment.
        """,
        "prognosis": """
        The prognosis for glioma patients varies depending on the tumor grade and location. Low-grade gliomas have a better prognosis 
        with long-term survival rates, while high-grade gliomas, such as glioblastomas, have a more guarded prognosis. Early diagnosis 
        and treatment significantly improve outcomes.
        """
    },
    1: {
        "label": "Meningioma Tumor",
        "definition": """
        Meningiomas are tumors that form in the meninges, the protective membranes covering the brain and spinal cord. 
        They are typically slow-growing and often benign, but they can still cause significant neurological symptoms due to their size and location. 
        Symptoms may include seizures, vision problems, or neurological deficits.
        """,
        "treatment": """
        Treatment for meningiomas typically includes surgery to remove the tumor. If surgery is not possible or if the tumor is difficult to access, 
        radiation therapy may be used to shrink the tumor. In some cases, observation is recommended for small, asymptomatic meningiomas. 
        A neurosurgeon will lead the surgical process, while radiation oncologists may be involved in cases where radiation is necessary.
        """,
        "prognosis": """
        Meningiomas generally have a favorable prognosis, especially if they are benign and fully resected. However, some meningiomas may 
        recur, requiring further treatment. The location of the tumor also affects the prognosis—tumors near critical brain structures may 
        result in more complications even after treatment.
        """
    },
    2: {
        "label": "No Tumor",
        "definition": """
        No tumor is detected in the brain. This indicates that the MRI scan or diagnostic test shows no signs of abnormal growth or masses in the brain tissue. 
        It’s a positive result that suggests the absence of a malignancy or any other abnormality typically associated with brain tumors.
        """,
        "treatment": """
        No treatment is necessary when no tumor is found. This result often provides peace of mind to the patient, though regular follow-up appointments may 
        still be recommended, especially for individuals with other risk factors or who experience symptoms related to neurological health.
        """,
        "prognosis": """
        The prognosis is excellent for patients who do not have a brain tumor, as no medical intervention is required. Maintaining a healthy lifestyle 
        and staying aware of any future neurological changes is important for long-term health.
        """
    },
    3: {
        "label": "Pituitary Tumor",
        "definition": """
        Pituitary tumors develop in the pituitary gland, a small but crucial gland located at the base of the brain. These tumors can affect the production 
        of hormones, leading to a range of symptoms such as vision problems, hormonal imbalances, and changes in menstrual cycles. 
        Most pituitary tumors are benign (non-cancerous) but can still cause significant health issues.
        """,
        "treatment": """
        Treatment for pituitary tumors often involves medication to control hormone production, and in some cases, surgery to remove the tumor. 
        If surgery is not feasible, radiation therapy may be used to shrink the tumor. Endocrinologists and neurosurgeons usually collaborate on the management 
        of pituitary tumors to ensure hormonal balance is restored and any tumor growth is managed.
        """,
        "prognosis": """
        The prognosis for pituitary tumors is generally good, especially when the tumor is detected early. Many patients can lead normal lives after 
        successful treatment, though hormone replacement therapy may be necessary if the tumor interferes with pituitary function. 
        Recurrence of the tumor is rare after complete removal.
        """
    }
}


def preprocess_image(image_path):
    # Le modele (mon_model_vgg16.h5) attend des images en 240x240 (verifie via model.input_shape)
    image = load_img(image_path, target_size=(240, 240))
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image_url = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', result="No file uploaded.")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', result="No selected file.")

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        image_url = '/' + file_path.replace('\\', '/')

        processed = preprocess_image(file_path)
        prediction = model.predict(processed)
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100

        label = class_info[predicted_class]['label']
        definition = class_info[predicted_class]['definition']
        treatment = class_info[predicted_class]['treatment']

        result = {
            'label': label,
            'confidence': confidence,
            'definition': definition,
            'treatment': treatment
        }

    return render_template('index.html', result=result, image=image_url)

if __name__ == '__main__':
    app.run(debug=True)
