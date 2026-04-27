import streamlit as st
import pickle
import numpy as np
import pandas as pd

pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Laptop Price Predictor")

# Inputs
company = st.selectbox('Brand', df['Company'].unique())
laptop_type = st.selectbox('Type', df['TypeName'].unique())

ram = st.selectbox('RAM(in GB)', df['Ram'].unique())

os = st.selectbox('OS', df['OpSys'].unique())

weight = st.number_input('Weight of Laptop')

touchscreen = st.selectbox('Touchscreen', ['No','Yes'])
ips = st.selectbox('IPS', ['No','Yes'])

screen_size = st.slider('Screen Size (inches)', 10.0, 18.0, 13.0)

resolution = st.selectbox(
    'Screen Resolution',
    ['1920x1080','1366x768','1600x900','3840x2160',
     '3200x1800','2880x1800','2560x1600','2560x1440','2304x1440']
)

cpu = st.selectbox('CPU', df['Cpu_brand'].unique())
gpu = st.selectbox('GPU', df['Gpu_brand'].unique())

hdd = st.selectbox('HDD(in GB)', [0,128,256,512,1024,2048])
ssd = st.selectbox('SSD(in GB)', [0,8,128,256,512,1024])

if st.button('Predict Price'):

    # binary encoding
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # PPI calculation
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2 + Y_res**2) ** 0.5) / screen_size

    # IMPORTANT: correct column order
    query = pd.DataFrame({
        'Company': [company],
        'TypeName': [laptop_type],
        'Ram': [ram],
        'OpSys': [os],
        'Weight': [weight],
        'TouchScreen': [touchscreen],
        'IPS': [ips],
        'ppi': [ppi],
        'Cpu_brand': [cpu],
        'Gpu_brand': [gpu],
        'HDD': [hdd],
        'SSD': [ssd]
    })

    prediction = np.exp(pipe.predict(query)[0])

    st.success(f"Predicted Price: Rs {int(prediction):,}")