from streamlit_drawable_canvas import st_canvas
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from modules import predict_image, model_loader, gemini_api

model = model_loader.load_model()

st.set_page_config(
    page_title="Kuzushiji Character Recognition",
    page_icon="🇯🇵",
)

st.title("Japanese Kuzushiji Character Recognition")

st.sidebar.header("Canvas Settings")
stroke_width = st.sidebar.slider("Stroke width: ", 1, 30, 15)
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
    
tab1, tab2 = st.tabs(["🖼️ Draw", "📁 Upload Image"])

with tab1:
    st.write("Draw a character on the canvas below and click 'Predict' to see the model's prediction.")
    
    canvas_result = st_canvas(
        stroke_width=stroke_width,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )
    
    if canvas_result.image_data is not None:
        if np.min(canvas_result.image_data) < 255:
            if st.button("Predict", key="btn_drawn"):
                char, confidence, img_inverted, predict = predict_image.predict_image(model,canvas_result.image_data,"canvas")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(img_inverted, caption='Processed Image', width=200)
                with col2:
                    st.metric("CNN Prediction", f"Hiragana : {char}", f"Confidence : {confidence:.1f}%")
                with col3:
                    if gemini_api_key:
                        with st.spinner("Recognizing curved japanese by gemini..."):
                            char,_ = gemini_api.gemini_req(Image.fromarray(img_inverted), gemini_api_key)
                            st.metric("Gemini Prediction", f"Hiragana : {char}")
                    else:
                        st.write("Enter your Gemini API Key in the sidebar to enable Gemini predictions.")
                
with tab2:
    st.write("Upload an image of a Japanese character and click 'Predict' to see the model's prediction.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', width=200)
        
        img_array = np.array(image)
        
        if st.button("Predict", key="btn_uploaded"):
            char, confidence, img_inverted, predict = predict_image.predict_image(model,img_array,"upload")
                
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(img_inverted, caption='Processed Image', width=200)
            with col2:
                st.metric("CNN Prediction", f"Hiragana : {char}", f"Confidence : {confidence:.1f}%")
            with col3:
                if gemini_api_key:
                        with st.spinner("Recognizing curved japanese by gemini..."):
                            char,_ = gemini_api.gemini_req(Image.fromarray(img_inverted), gemini_api_key)
                            st.metric("Gemini Prediction", f"Hiragana : {char}")
                else:
                        st.write("Enter your Gemini API Key in the sidebar to enable Gemini predictions.")


        
