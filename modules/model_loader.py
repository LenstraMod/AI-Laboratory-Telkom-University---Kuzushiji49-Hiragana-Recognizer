import streamlit as st
import tensorflow as tf
import numpy as np
import h5py

@st.cache_resource
def load_model(path="model_kmnist49.h5"):
    try:
       model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv2d_8'),
            tf.keras.layers.MaxPooling2D((2, 2), name='max_pooling2d_8'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_9'),
            tf.keras.layers.MaxPooling2D((2, 2), name='max_pooling2d_9'),
            tf.keras.layers.Flatten(name='flatten_4'),
            tf.keras.layers.Dense(256, activation='relu', name='dense_8'),
            tf.keras.layers.Dropout(0.3, name='dropout_4'),
            tf.keras.layers.Dense(49, activation='softmax', name='dense_9')
       ])
       
       model.build((None, 28, 28, 1))
       
       with h5py.File(path, 'r') as f:
            weight_groups = f['model_weights']
            for layer in model.layers:
                if layer.name not in weight_groups:
                    continue

                try:
                    g1 = weight_groups[layer.name]

                    # step into "sequential_4"
                    g2 = g1[list(g1.keys())[0]]

                    # step into the actual layer folder again
                    g3 = g2[layer.name]

                    kernel = np.array(g3['kernel'])
                    bias   = np.array(g3['bias'])

                    layer.set_weights([kernel, bias])
                except Exception as e:
                    print(f"skip {layer.name}: {e}")
                
       return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None