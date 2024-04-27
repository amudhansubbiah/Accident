import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from prediction import ordinal_encoder, get_prediction

model = joblib.load(r'C:\amudhan\tmlc\traffic-1\models\exratree.joblib')
st.set_page_config(page_title="Accident Severity Prediction",page_icon="*", layout="wide")

#creating options menu
options_lane=[1,2,3]
options_day=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
options_age=['under 18', '18-30','31-50','Unknown','over 51']
options_driver_exp=['1-2yr','Above 10yr' ,'5-10yr', '2-5yr','No Licence','Below 1yr','unknown'] 
options_light_condition=['Daylight', 'Darkness - lights lit','Darkness - no lighting', 'Darkness - lights unlit'] 
options_junction_type=['No junction','Y Shape','Crossing','O Shape','Other','Unknown','T Shape','X Shape']
options_road_surface=['Asphalt roads','Earth roads','Asphalt roads with some distress','Gravel roads','Other']



st.markdown("<h1 style='text align:center;'> Accident Severity Prediction  </h1>",unsafe_allow_html=True)
def main():
    with st.form('Prediction Form'):
        st.subheader("Enter the input for the following features")

        hour=st.slider("Pickup Hour:", 0,23, value=0, format='%d')
        day_of_week= st.selectbox("Select the Day of the Week", options=options_day)
        driver_age= st.selectbox("Select Driver Age", options=options_age) 
        driving_experience=st.selectbox("Select Driver Experience", options=options_driver_exp)
        junction_type= st.selectbox("Select Junction Type", options=options_junction_type)             
        road_surface_conditions= st.selectbox("Select Road Surface Condition", options=options_road_surface)   
        light_condition= st.selectbox("Select Light Condition", options=options_light_condition)           
        vehicles_involved= st.slider("No of Vehicles involved:", 0,60, value=0, format='%d')          
        casualties=st.slider("No of pepole injured:", 0,60, value=0, format='%d')       
        minute= st.slider("Pickup minute:", 0,60, value=0, format='%d')                    
        
        submit=st.form_submit_button("Predict")

        if(submit):
            day_of_week=ordinal_encoder(day_of_week,options_day)
            driver_age=ordinal_encoder(driver_age,options_age)
            driving_experience=ordinal_encoder(driving_experience,options_driver_exp)
            junction_type=ordinal_encoder(junction_type,options_junction_type)
            road_surface_conditions=ordinal_encoder(road_surface_conditions,options_road_surface)
            light_condition=ordinal_encoder(light_condition,options_light_condition)


            data=np.array([hour, day_of_week,driver_age,driving_experience,junction_type,road_surface_conditions,light_condition,vehicles_involved,casualties,minute]).reshape(1,-1)

            pred=get_prediction(data=data,model=model)

            st.write(f"Prediction Severity is: {pred[0]}")
if __name__ == '__main__':
   main()


    