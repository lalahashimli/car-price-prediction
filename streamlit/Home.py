import streamlit as st
import pandas as pd
import warnings
import pickle
import time
import PIL
from PIL import Image


warnings.filterwarnings(action='ignore')

# df = pd.read_csv('turbo.csv')

moto4_image = Image.open('streamlit/moto4.png')

interface = st.container()


with interface:

    st.title('Moto4 - Car Price Prediction System')

    st.image(moto4_image)
    
#     st.markdown('***')
    st.write('<hr style="height: px; background-color: gray; border: none; margin: px 0;" />', unsafe_allow_html=True)



    st.header('Project Description')

    st.markdown("<p style='font-size: 20px;'>Our company is working on a system for everyone who is using car or busy with car selling. The model helps you predicting the price of your car, if you intend to sell your car or are considering purchasing a new one.This system calculate average market price for you  and subsequently provides output on which car suits your needs best.It will be created like a part of a website, available for integration into any car-selling platform.When car owners visit the website to sell their vehicles, they can enter key car features that significantly influence the pricing.The system will then use average statistical indicators to estimate an appropriate price.If the seller chooses to proceed, the predicted value will be displayed as an announcement on the website instantly.</p>",unsafe_allow_html = True)

#     st.markdown('***')

