import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_model(path="model_kmnist49.h5"):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None