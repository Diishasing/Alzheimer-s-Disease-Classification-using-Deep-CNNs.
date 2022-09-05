import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf
import cv2


st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True) #model is stored in the cache memory so it wont take time to load again

def load_model():
    model = tf.keras.models.load_model('C:/Users/91701/Desktop/Alz_Classification/predict.hdf5') #model saved during training for prediction
    
    return model

model = load_model()

html_temp = '''
            <div style = "background-color: white;padding:10px">
            <h2 style = "color: black; text-align:center;">ALZHEIMER'S CLASSIFICATION STREAMLIT APP
            </h2>
            </div>'''
st.markdown(html_temp, unsafe_allow_html = True)
if st.checkbox("Show classifying classes"):
    st.write("The Alzheimer's Classification App can predict the following classes:")
    st.write('1. Mild Demented')
    st.write('2. Moderetly Demented')
    st.write('3. Non Demented')
    st.write('4. VeryMild Demented')


files = st.file_uploader("Uplaod an brain MRI image: ", type = ['jpg', 'png', 'jpeg'])

def prediction_(data, model):
    size = (200,200)    
    image = ImageOps.fit(data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction

if files is None:
    st.text('PLease uplaod an image file to predict ')
else:
    img = Image.open(files)
    st.image(img, use_column_width=True)
    pred_button = st.button("Predict")
    if pred_button:
        predictions = prediction_(img, model)
        classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
        string = 'this is : ' + classes[np.argmax(predictions)]
        st.success(string)    





