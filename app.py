import pickle
import pandas as pd
import streamlit as st
import numpy as np
from streamlit_modal import Modal

with open("label_categorical.pkl", "rb") as file:
    label_categorical = pickle.load(file)

MODEL_PATH = "final_model.pkl"
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

with open("normalizer.pkl", "rb") as file:
    normalizer = pickle.load(file)

if __name__ == "__main__":
    st.title("Customer Churn Prediction")

    account_length=st.number_input("Customer Active Since(Months)", 0, 1200)

    area_code=st.selectbox(
            "Area Code",
            ['area_code_408', 
            'area_code_415', 
            'area_code_510'],
            index=0,
        )

    state=st.selectbox(
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
    

    voice_plan=st.selectbox("Has Voice plan?", ["No", "Yes"], index=0)
    voice_messages=st.number_input("Total Voice messages", 0.0, 1200.0)
    intl_plan=st.selectbox("Has International plan?", ["No", "Yes"], index=0)
    churn='no'
    intl_mins=st.number_input("Total International Minutes", 0.0, 1200.0)
    intl_calls=st.number_input("Total International Calls", 0, 1200)
    intl_charge=st.number_input("International Charges", 0.0, 1200.0)

    day_mins=st.number_input("Total Day Minutes", 0.0, 1200.0)
    day_calls=st.number_input("Total Day Calls", 0, 1200)
    day_charge=st.number_input("Day Charges", 0.0, 1200.0)

    eve_mins=st.number_input("Total Evening Minutes", 0.0, 1200.0)
    eve_calls=st.number_input("Total Evening Calls", 0, 1200)
    eve_charge=st.number_input("Evening Charges", 0.0, 1200.0)

    night_mins=st.number_input("Total Night Minutes", 0.0, 1200.0)
    night_calls=st.number_input("Total Night Calls", 0, 1200)
    night_charge=st.number_input("Night Charges", 0.0, 1200.0)

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
            st.write('The probability percentage is',str(np.round(prob*100,2)),'%')
        else:
            st.title("Not Likely to Churn")
            prob=np.round(model.predict_proba(input_df)[:,0],3)[0]
            st.write('The probability percentage is',str(np.round(prob*100,2)),'%')

button=st.button("Stats For Nerd")	
modal = Modal("Stats For Nerd","black")

if button:
    modal.open()
import streamlit.components.v1 as components
if modal.is_open():
	
	with modal.container():
		#st.write("Prediction Model Used - Support Vector Classification (SVC)")
		#st.write("Word Embedding done Using - Pre-Trained Glove")
		#button1=st.button("Classification Report")
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
			categories = ['figurative', 'irony', 'regular', 'sarcasm']
			plt.figure(figsize=(7, 5))
			df_cm=pd.read_csv('Confusion Matrix.csv')
			df_cm.index=['figurative', 'irony', 'regular', 'sarcasm']
			sns.heatmap(df_cm, annot=True,cmap = 'Blues',fmt = '.1f',xticklabels = categories, yticklabels = categories)
			plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
			plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
			plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
			st.pyplot()
		if col3.button('Word Cloud','Word Cloud'):
			if (text2!=""):
				wordcloud = WordCloud(width=800, height=400).generate(text)
				st.title('Word Cloud')
				plt.figure(figsize=(7, 5))
				plt.imshow(wordcloud, interpolation='bilinear')
				plt.axis('off')
				st.pyplot(plt)
			else:
				st.write("No Words to Plot")
		
		st.write("[Click Here to view complete GitHub Repository](https://github.com/VinayNagaraj07/Twitter-Sentiment-Analysis)")
		
st.markdown('''
    ## Disclaimer
    
    This Predictions are made from training on a specific Dataset only and for it is to be used solely learning purposes only. Please consult with a qualified professional before making any decisions.
    
    ---
    ''')
