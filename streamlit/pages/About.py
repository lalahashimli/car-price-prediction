import streamlit as st
import pandas as pd
import warnings
import pickle
import time
import streamlit as st
import PIL
from PIL import Image


df = pd.read_csv('../../turbo.csv')


interface = st.container()

with interface:


    st.title('About Dataset')

    st.subheader('The dataset includes information about:')
    
    st.markdown("<p style='font-size: 20px;'>1. Brand - The brand or manufacturer of the car</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>2. Model - The specific model of the car</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>3. City - The city or location where the car is being sold</p>",unsafe_allow_html = True)

    st.markdown("<p style='font-size: 20px;'>4. Fuel type - The type of fuel the car uses</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>5. Gear - The gear of car as string</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>6. Ban type - The ban type of car as string</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>7. Gear box - The type of transmission in the car (e.g., automatic, manual)</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>8. Mileage - The total distance the car has been driven</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>9. Year - The year the car was manufactured</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>10. Color - The color of the car</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>11. For which market - for which market a car is assembled</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>12. Engine volume - The size of the car's engine in cubic centimeters</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>13. Engine power - The power output of the car's engine in horsepower (HP)</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>14. Avtosalon - whether a car is a avtosalon car or not</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>15. New - whether a car is new or not</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>16. Is it colored? - whether a car is colored or not</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>17. Has a stroke? -whether a car has a stroke or not</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>18. Solder disc - whether a car has soldered discs or not</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>19. ABS - whether a car has ABS or not</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>20. Lyuk - whether a car has lyuk or not</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>21. Rain sensor - whether a car has rain sensor or not</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>22. Skin salon - whether a car has skin salon or not</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>23. Central locking - whether a car has central locking or not</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>24. Parking radar - whether a car has parking radar or not</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>25. Air conditioning - whether a car has air conditioning or not</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>26. Heated seats - whether a car has heated seats or not</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>27. Xenon lamps - whether a car has xenon lamps or not</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>28. Rear view camera - whether a car has rear view camera or not</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>29. Side curtains radar - whether a car has side curtains or not</p>",unsafe_allow_html = True)
    st.markdown("<p style='font-size: 20px;'>29. Seat ventilation - whether a car has seat ventilation or not</p>",unsafe_allow_html = True)


    st.info(f'The dataset consist of {df.shape[0]} rows by {df.shape[1]} columns')
    
    show_data = st.checkbox("Show Data")
 
    if show_data:
        st.write(df)
