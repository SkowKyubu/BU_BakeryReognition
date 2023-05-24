import base64
import re
import uuid
from io import BytesIO
from pathlib import Path
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from keras.models import load_model
import cv2
from PIL import Image
import numpy as np

# Page: Figure_recognition
def Figure_recognition_page():
    # Add content for the home page here
    st.title("Figure recognition app")

    ### UPLOAD PART :
    type_image = ['Draw','Upload image']
    choix_du_telechargement = st.radio("You want to :",type_image)
    if choix_du_telechargement== 'Upload image':
        im = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        if im is not None:
            im = Image.open(im)

            # Print the uploaded image :
            width, height = im.size
            img = im.resize((100, int(100 * height / width)))
            st.image(img)

            im = np.array(im)
            im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)
    ### DRAWING PART :
    if choix_du_telechargement== 'Draw':
        st.markdown("### Drawing section")

        st.markdown(
            """
        Press the 'Download' button at the bottom of canvas to update exported image.
        """
        )
        try:
            Path("tmp/").mkdir()
        except FileExistsError:
            pass

        if "button_id" not in st.session_state:
            st.session_state["button_id"] = ""

        # Regular deletion of tmp files
        if st.session_state["button_id"] == "":
            st.session_state["button_id"] = re.sub(
                "\d+", "", str(uuid.uuid4()).replace("-", "")
            )

        button_id = st.session_state["button_id"]
        file_path = f"tmp/{button_id}.png"

        custom_css = f""" 
            <style>
                #{button_id} {{
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    background-color: rgb(255, 255, 255);
                    color: rgb(38, 39, 48);
                    padding: .25rem .75rem;
                    position: relative;
                    text-decoration: none;
                    border-radius: 4px;
                    border-width: 1px;
                    border-style: solid;
                    border-color: rgb(230, 234, 241);
                    border-image: initial;
                }} 
                #{button_id}:hover {{
                    border-color: rgb(246, 51, 102);
                    color: rgb(246, 51, 102);
                }}
                #{button_id}:active {{
                    box-shadow: none;
                    background-color: rgb(246, 51, 102);
                    color: white;
                    }}
            </style> """

        bg_color = "#000000"
        stroke_color = "#ffffff"
        data = st_canvas(update_streamlit=False, key="png_export",background_color=bg_color,
                         stroke_color=stroke_color)
        if data is not None and data.image_data is not None:
            img_data = data.image_data
            im = Image.fromarray(img_data.astype("uint8"), mode="RGBA")
            im.save(file_path, "PNG")

            buffered = BytesIO()
            im.save(buffered, format="PNG")
            img_data = buffered.getvalue()
            try:
                # some strings <-> bytes conversions necessary here
                b64 = base64.b64encode(img_data.encode()).decode()
            except AttributeError:
                b64 = base64.b64encode(img_data).decode()

    ### LOADING MODEL PART :
    st.markdown("## Import the model for the prediction")
    # Définir les options de sélection de modèle
    model_options = {
        'Neural Network': 'NN_model.h5',
        'Deep Network': 'CNN_model.h5',
    }

    # Afficher le menu déroulant pour la sélection de modèle
    selected_model = st.selectbox('Select a model', list(model_options.keys()))

    # Charger le modèle sélectionné
    model_path = model_options[selected_model]
    model = load_model(model_path)

    if im is not None:
        im_array = np.array(im)
        im_rgb = cv2.cvtColor(im_array, cv2.COLOR_RGBA2RGB)  # RGBA to RGB
        im_grey = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2GRAY)  # RGB to grey
        # Resize the image to 28x28 pixels
        im_resized = cv2.resize(im_grey, (28, 28))
        # Reshape the image to (1,28,28,1) to match the input shape expected by the CNN model
        im_input = im_resized.reshape((1, 28, 28, 1))

        im_input = np.asarray(im_input)
        prediction = model.predict(im_input)
        #predicted_class = np.argmax(prediction)
        #st.markdown("# Prediction : " + str(predicted_class))

        predicted_classes = np.argsort(-prediction)

        #for i in range(len(predicted_classes)):
         #   st.markdown(f"#{i + 1} Prediction : {predicted_classes[i]} - Confidence : {prediction[predicted_classes[i]]}")

        st.markdown("## Prediction : " + str(predicted_classes[0][0]))

# Page: Bakery Recognition
def Bakery_recognition_page():
    st.title("Bakery Recognition Page")
    ### UPLOAD PART :
    st.markdown("Utilize this model to accurately identify five types of pastries: croissant, pain au chocolat, cookie, donut, and cannele.")
    im = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    if im is not None:
        im = Image.open(im)

        # Print the uploaded image :
        width, height = im.size
        img = im.resize((100, int(100*height/width)))
        st.image(img)

        im = im.resize((224, 224))  # Resize to the expected input shape of the model
        im = np.array(im)
        im = np.expand_dims(im, axis=0)
        im = im / 255.0  # Normalize pixel values to [0, 1]
        # LOAD THE MODEL :
        model = load_model('mymodelpower.h5', compile=False)


        # Define the class labels
        class_labels = ['cannele', 'cookie','croissant','donuts','pain au chocolat']  # Replace with your own class labels

        # Make predictions using the model
        predictions = model.predict(im)

        # Get the predicted class label
        predicted_class_index = np.argmax(predictions, axis=1)
        predicted_class_label = class_labels[predicted_class_index[0]]

        # Print the predicted class label
        st.markdown("### Predicted class : " + str(predicted_class_label))

# Sidebar navigation
pages = {
    "Figure recognition": Figure_recognition_page,
    "Bakery recognition": Bakery_recognition_page,
}

st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", tuple(pages.keys()))

# Display the selected page
pages[selected_page]()




