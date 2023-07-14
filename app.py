import pickle
import pandas as pd
import streamlit as st
import numpy as np
from streamlit_modal import Modal
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from dateutil import relativedelta
from datetime import date
join_date=""

def calculate_months(start_date, end_date):
    start = date(start_date.year, start_date.month, start_date.day)
    end = date(end_date.year, end_date.month, end_date.day)
    delta = relativedelta.relativedelta(end, start)
    return delta.years * 12 + delta.months

st.set_option('deprecation.showPyplotGlobalUse', False)
with open("label_categorical.pkl", "rb") as file:
    label_categorical = pickle.load(file)

MODEL_PATH = "final_model.pkl"
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

with open("normalizer.pkl", "rb") as file:
    normalizer = pickle.load(file)

if __name__ == "__main__":
    st.title("Customer Churn Prediction")
    st.info('This app is created to predict Customer Churn of a Telecom Company')
    add_selectbox = st.sidebar.selectbox("Select Prediction Type",("Online", "Batch"))
    
    if add_selectbox=='Online':
        st.markdown("<p style='font-size: 24px;font-weight: bold;'>Customer Tenure</p>", unsafe_allow_html=True)
    # Create a radio button to select input type
        input_type = st.radio("Select Input Type", ("Tenure (in months)", "Joining Date"))

        if input_type == "Tenure (in months)":
            account_length = st.number_input("Enter Tenure (in months)", min_value=0)
            st.write("Tenure (in months):", str(account_length))
        else:
            join_date = st.date_input("Select Joining Date",max_value=date.today())
        if join_date:
            today = date.today()
            start_date = join_date
            end_date = today
            account_length = calculate_months(start_date, end_date)
            st.write("Tenure (in months):", str(account_length))
        #account_length=col1.number_input("Customer Active Since(Months)", 0, 1200)
        #user_date = col2.date_input("Enter a date")

        st.markdown("<p style='font-size: 24px;font-weight: bold;'>Customer Location</p>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        area_code=col1.selectbox(
                "Area Code",
                ['area_code_408', 
                'area_code_415', 
                'area_code_510'],
                index=0,
            )

        state=col2.selectbox(
                "State",
                ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 
                'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 
                'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 
                'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 
                'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 
                'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 
                'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY'],
                index=0,
            )
        st.markdown("<p style='font-size: 24px;font-weight: bold;'>Voice Plan</p>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        voice_plan=col1.selectbox("Has Voice plan?", ["No", "Yes"], index=0)
        voice_messages=col2.number_input("Total Voice messages", 0.0, 1200.0)
        #intl_plan=st.selectbox("Has International plan?", ["No", "Yes"], index=0)
        churn='no'
        
        st.markdown("<p style='font-size: 24px;font-weight: bold;'>International Calls Details</p>", unsafe_allow_html=True)
        col1, col2,col3,col4 = st.columns(4)
        intl_plan=col1.selectbox("Has International plan?", ["No", "Yes"], index=0)
        intl_mins=col2.number_input("Total International Minutes", 0.0, 1200.0)
        intl_calls=col3.number_input("Total International Calls", 0, 1200)
        intl_charge=col4.number_input("International Charges", 0.0, 1200.0)
        st.markdown("<p style='font-size: 24px;font-weight: bold;'>Day Calls Details</p>", unsafe_allow_html=True)
        col1, col2,col3 = st.columns(3)
        day_mins=col1.number_input("Total Day Minutes", 0.0, 1200.0)
        day_calls=col2.number_input("Total Day Calls", 0, 1200)
        day_charge=col3.number_input("Day Charges", 0.0, 1200.0)
        st.markdown("<p style='font-size: 24px;font-weight: bold;'>Evening Calls Details</p>", unsafe_allow_html=True)
        col1, col2,col3 = st.columns(3)
        eve_mins=col1.number_input("Total Evening Minutes", 0.0, 1200.0)
        eve_calls=col2.number_input("Total Evening Calls", 0, 1200)
        eve_charge=col3.number_input("Evening Charges", 0.0, 1200.0)
        st.markdown("<p style='font-size: 24px;font-weight: bold;'>Night Calls Details</p>", unsafe_allow_html=True)
        col1, col2,col3 = st.columns(3)
        night_mins=col1.number_input("Total Night Minutes", 0.0, 1200.0)
        night_calls=col2.number_input("Total Night Calls", 0, 1200)
        night_charge=col3.number_input("Night Charges", 0.0, 1200.0)
        st.markdown("<p style='font-size: 24px;font-weight: bold;'>Customer-Company Interaction</p>", unsafe_allow_html=True)
        customer_calls=st.number_input("Called Customer Care for?", 0, 12)


        input_df=pd.DataFrame(columns=['state', 'area_code', 'voice_plan', 'intl_plan', 'churn',
           'account_length', 'voice_messages', 'intl_mins', 'intl_calls',
           'intl_charge', 'day_mins', 'day_calls', 'day_charge', 'eve_mins',
           'eve_calls', 'eve_charge', 'night_mins', 'night_calls', 'night_charge',
           'customer_calls'])
        data1=[state, area_code, voice_plan, intl_plan, churn,
           account_length, voice_messages, intl_mins, intl_calls,
           intl_charge, day_mins, day_calls, day_charge, eve_mins,
           eve_calls, eve_charge, night_mins, night_calls, night_charge,
           customer_calls]
        data={}
        i=""
        j=0
        for i in input_df:
            data[i]=data1[j]
            j=j+1
        label_categorical['No']=0
        label_categorical['Yes']=1
        new_row = pd.DataFrame([data])
        input_df = pd.concat([input_df, new_row], ignore_index=True)
        input_df=input_df.replace(label_categorical)
        normalised=np.round(normalizer.transform(input_df),2)
        input_df=pd.DataFrame(normalised,columns=input_df.columns)
        input_df.drop(['churn'],axis=1,inplace=True)
        #st.write(input_df)
        if st.button("Predict"):
            if model.predict(input_df)==1:
                st.title("Likely to Churn")
                prob=np.round(model.predict_proba(input_df)[:,1],3)[0]
                st.write('The probability percentage is',str(np.round(prob*100,3)),'%')
            else:
                st.title("Not Likely to Churn")
                prob=np.round(model.predict_proba(input_df)[:,0],3)[0]
                st.write('The probability percentage is',str(np.round(prob*100,3)),'%')
    if add_selectbox=="Batch":
        st.markdown("<p style='font-size: 24px;font-weight: bold;'>File Should Contain Following Coloumns</p>", unsafe_allow_html=True)
        dummy_df=pd.DataFrame(columns=['state', 'area_code', 'voice_plan', 'intl_plan',
           'account_length', 'voice_messages', 'intl_mins', 'intl_calls',
           'intl_charge', 'day_mins', 'day_calls', 'day_charge', 'eve_mins',
           'eve_calls', 'eve_charge', 'night_mins', 'night_calls', 'night_charge',
           'customer_calls'])
        #html_table = dummy_df.to_html(index=False)
        #html_table = html_table.replace('<table', '<table style="border-collapse: collapse;"')
        #st.markdown(html_table, unsafe_allow_html=True)
        st.write(dummy_df)
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            data.columns=['state', 'area_code',
           'account_length','voice_plan', 'voice_messages', 'intl_plan','intl_mins', 'intl_calls',
           'intl_charge', 'day_mins', 'day_calls', 'day_charge', 'eve_mins',
           'eve_calls', 'eve_charge', 'night_mins', 'night_calls', 'night_charge',
           'customer_calls']
            input_df=data.copy()
            data['churn']=np.zeros(len(data))
            data=data[['state', 'area_code', 'voice_plan', 'intl_plan', 'churn',
           'account_length', 'voice_messages', 'intl_mins', 'intl_calls',
           'intl_charge', 'day_mins', 'day_calls', 'day_charge', 'eve_mins',
           'eve_calls', 'eve_charge', 'night_mins', 'night_calls', 'night_charge',
           'customer_calls']]
            st.write(data)
            data=data.replace(label_categorical)
            normalised=np.round(normalizer.transform(data),2)
            data=pd.DataFrame(normalised,columns=data.columns)
            data.drop(['churn'],axis=1,inplace=True)
            data1=pd.DataFrame()
            if st.button("Predict"):
                input_df["Prediction"]=model.predict(data)
                j=0
                Prediction_Proba=[]
                Prediction=[]
                for i in input_df["Prediction"]:
                    if i==1:
                        Prediction_Proba.append(str(np.round(model.predict_proba(data)[j,1]*100,3))+"%")
                        Prediction.append("Likely to Churn")
                    else:
                        Prediction_Proba.append(str(np.round(model.predict_proba(data)[j,0]*100,3))+"%")
                        Prediction.append("Not Likely to Churn")
                    j=j+1

                input_df["Confidence"]=Prediction_Proba
                input_df["Prediction"]=Prediction
                st.write(input_df)


button=st.button("Stats For Nerd")  
modal = Modal("Stats For Nerd","black")

if button:
    modal.open()
import streamlit.components.v1 as components
if modal.is_open():
    
    with modal.container():
        col1, col2,col3 = st.columns(3)
        if col1.button('Classification Report','Classification Report'):
            st.write("Classification Report of Trained Model")
            book1=pd.read_csv("Book1.csv")
            book1.replace(np.NaN,"",inplace=True)
            #book1 = book1.style.set_properties(**{'text-align': 'center'})
            html_table = book1.to_html(index=False)
            html_table = html_table.replace('<table', '<table style="border-collapse: collapse;"')
            st.markdown(html_table, unsafe_allow_html=True)
        if col2.button('Confusion Matrix','Confusion Matrix'):
            st.write("Confusion Matrix of Trained Model")
            categories = ['No', 'Yes']
            #plt.figure(figsize=(4, 3))
            df_cm=pd.read_csv('Confusion Matrix.csv')
            df_cm.index=['No', 'Yes']
            sns.heatmap(df_cm, annot=True,cmap = 'Blues',fmt = '.1f',xticklabels = categories, yticklabels = categories)
            plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
            plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
            plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
            st.pyplot()
        if col3.button('Visualization'):
            #visualize()
            col4, col5 = st.columns(2)
            # Display the first image in the first column
            image1 = 'Original_data.png'
            col4.image(image1, caption="Original Data",use_column_width=True)
            # Display the second image in the second column
            image2 = 'Model_Prediction.png'
            col5.image(image2, caption="Predicted Data",use_column_width=True)
        st.write("[Click Here to view complete GitHub Repository](https://github.com/VinayNagaraj07/Telecom_Churn_Prediction)")
        
st.markdown('''
    ## Disclaimer
    
    This Predictions are made from training on a specific Dataset only and for it is to be used solely learning purposes only. Please consult with a qualified professional before making any decisions.
    
    ---
    ''')
