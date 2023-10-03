import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import io
import class_details as cd
# from sklearn.preprocessing import LabelEncoder

# Load your classification model
model = load_model("./skin_cancer_model_2.h5")

SIZE = 32

# Define skin disease classes (modify as needed)
class_names = ["Actinic keratoses",
               "Basal cell carcinoma",
               "Benign keratosis-like",
               "Dermatofibroma",
               "Melanoma",
               "Melanocytic nevi",
               "Vascular lesions"
               ]


# Define details for each class (modify as needed)



def classify_skin_disease(image):
    new_size = (32, 32)
    image = cv2.resize(image, new_size)
    img = np.asarray(image)
    img = img / 255
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    result = model.predict(img)
    predicted_class_index = np.argmax(result)
    predicted_class_label = class_names[predicted_class_index]
    confidence_scores = result[0][predicted_class_index]
    return predicted_class_label, confidence_scores


def main():
    st.set_page_config(
        page_title="SkinDiseaseNet",
        page_icon=":microscope:",
        layout="wide"
    )

    st.title("Skin Disease Detection and Classification")
    st.markdown("## 🌼 Welcome to SkinDiseaseNet 🌼")
    st.write(
        "An AI-powered tool to detect and classify skin diseases. Upload an image and get insights about the detected disease."
    )

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose an option", ["Home", "Image Upload"])

    if app_mode == "Home":
        st.image("banner_image.jpg", use_column_width=True)
        st.markdown(
            "This is the home page. Learn more about skin diseases and prevention.")
        st.markdown(
            "![Skin Health](skin_health.jpg)"
            "\n_Image source: [Unsplash](https://unsplash.com)_"
        )

    elif app_mode == "Image Upload":
        st.subheader("Image Upload and Classification")
        uploaded_file = st.file_uploader("Upload an image", type=[
                                         "jpg", "png"], key="image_upload")

        if uploaded_file is not None:

            image_bytes = io.BytesIO(uploaded_file.read())
            image = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), 1)

            # image = cv2.imread(image_path)
        
            st.image(image, caption="Uploaded Image", use_column_width=True)
            predicted_class, confidence = classify_skin_disease(image)
            # confidence = round(confidence * 100, 1)
            st.markdown(f"**Detected Skin Disease:** {predicted_class} with confidence score of **{confidence:.2f} %**")

            if predicted_class in class_names:
                details = cd.class_details[predicted_class]

                st.markdown(f"**Cause:** {details['cause']}")
                st.markdown(f"**Symptoms:** {details['symptoms']}")
                st.markdown(f"**Prevention:** {details['prevention']}")


if __name__ == "__main__":
    main()
