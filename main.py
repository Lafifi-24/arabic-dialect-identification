import streamlit as st
import pandas as pd
import pickle 
import operator
import numpy as np
import folium
from streamlit_folium import st_folium
from arabert.preprocess import ArabertPreprocessor
from spellchecker import SpellChecker
from aaransia import transliterate
check_frensh= SpellChecker(language='fr')
check_English=SpellChecker()
import time

import requests
import re
import pyarabic.araby as araby

API_URL_arabert = "https://api-inference.huggingface.co/models/lafifi-24/arabert_arabic_dialect_identification"
API_URL_arabicbert="https://api-inference.huggingface.co/models/lafifi-24/arabicBert_arabic_dialect_identification"
API_URL_arbert="https://api-inference.huggingface.co/models/lafifi-24/arbert_arabic_dialect_identification"
headers = {"Authorization": "Bearer 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"}



header = st.container()
dataset = st.container()
modeling = st.container()

test = st.container()

@st.cache
def get_data():
    df = pd.read_csv('data/data_v0.1.0.csv')
    return df
@st.cache(allow_output_mutation=True)
def prepro(model_name):
    return ArabertPreprocessor(model_name=model_name)
arabert_prep = prepro("bert-base-arabert")
    
# @st.cache(allow_output_mutation=True)
# def load_model():
#     model = pickle.load(open('models/NB.pkl', 'rb'))
#     return model
# NB_model = load_model()

######################################
def query(payload,model):
    if model=='arbert':
      response = requests.post(API_URL_arbert, headers=headers, json=payload)
    if model=='arabert':
      response = requests.post(API_URL_arabert, headers=headers, json=payload)
    if model=='arabicbert':
      response = requests.post(API_URL_arabicbert, headers=headers, json=payload)
    return response.json()

def pred(output):
    dic={}
    for i in output[0]:
        dic[i['label']]=i['score']
    return dic
def check(word):
    if(re.search(r'[a-zA-Z]',word)!=None):
        if word == check_English.correction(word) or word == check_frensh.correction(word):
            return False     
    return True
  
def preprocessing(text):
    text=text.lower()
    #remove links
    text = re.sub(r'http\S+', '',  text)
    #remove users nam
    text=' '.join(w for w in re.split(r"@\w*",text) if w)
    #remove English word and frensh word 
    if re.search(r'[a-zA-Z]',text)!=None:
        text=' '.join(w for w in text.split() if check(w))
        #use aranisia
        text=transliterate(text, source='ma', target='ar' , universal=True)
    #get just arabic text
    text=re.sub(r'[u0600-u06FF]+', '', text).strip()
    text=re.sub(r'[a-z]+', '', text).strip()
    #remove duplicate letter
    text=re.sub(r'(.)\1+', r'\1', text).strip()
    #removing delimiters from strings
    text=' '.join(w for w in re.split(r"\W", text) if w)
    #remove letters
    text=' '.join(w for w in araby.tokenize(text) if len(w)>1)
    return text
def araBert_model(text,model):
  text=preprocessing(text)
  output=query({'inputs':arabert_prep.preprocess(text)},model)
  return pred(output)
###############################


def display_map(df):
    #Setting up the world countries data URL
    url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'
    country_shapes = f'{url}/world-countries.json'
    m = folium.Map(location=[26.877981, 90.483711],zoom_start=3,min_zoom=3)
        #Adding the Choropleth layer onto our base map
    folium.Choropleth(
            #The GeoJSON data to represent the world country
            geo_data=country_shapes,
            name='choropleth COVID-19',
            data=df,
            #The column aceppting list with 2 value; The country name and  the numerical value
            columns=['Country', 'Value'],
            key_on='feature.properties.name',
            nan_fill_color='white',
            highlight=True
        ).add_to(m)
    return m
    





with header:
    st.title('Welcome to our NLP projet')
    

with dataset:
    st.header('Dataset description:')
    st.text('The dataset consists of 365K tweets of 18 dialects in addition to the MSA. Almost \n20K tweet for each dialect.')
    st.text('Here is a visualization of data distribution:')

    df = get_data()
    st.subheader('Data distribution')
    dialect_dist = pd.DataFrame(df['dialect'].value_counts())
    st.bar_chart(dialect_dist)

with modeling:
    st.header('Modeling:')
    st.text('In this project we tried to test multiple approches and compare between them')
    st.markdown(
        """
        The tested approches are:
        - Machine learning

        - DL

        """
        )


with test:
    
    st.header('Models Testing')

    col1, col2 = st.columns([1,2])

    model = col1.selectbox('Select a model', options=('AraBert','ArabicBert','ArBert'), index=0)#'Multinomial NB','Random Forest',
    input = col2.text_input('Enter an input text:', '')
    
     #pickle.load(open('models/NB.pkl', 'rb'))
    col2.button('Predict')
    

    
    #if col2.button('Predict'):

    
    i=0
    while i==0:
        try:
            if model == 'AraBert':
                dc = araBert_model(input,"arabert")
                print("XXXXXX"+str(dc))
                st.subheader('AraBert')
                a=dict(sorted(dc.items(), key = operator.itemgetter(1), reverse = True)[:4])
                pred = pd.DataFrame.from_dict(a, orient='index').rename(columns={0:'Country'})
                st.bar_chart(pred)
                opt = max(dc.items(), key=operator.itemgetter(1))[0]
                if opt=='MSA':
                    dc={'SA':1,'MA':1,'DZ':1,'EG':1,'SY':1,'QA':1,'LB':1,'YE':1,'AE':1,'KW':1,'SD':1,'BH':1,'JO':1,'IQ':1,'PL':1,'OM':1,'LY':1,'TN':1}
            elif model == 'ArabicBert':
                dc = araBert_model(input,"arabicbert")
                print("XXXXXX"+str(dc))
                st.subheader('ArabicBert')
                a=dict(sorted(dc.items(), key = operator.itemgetter(1), reverse = True)[:4])
                pred = pd.DataFrame.from_dict(a, orient='index').rename(columns={0:'Country'})
                st.bar_chart(pred)
                opt = max(dc.items(), key=operator.itemgetter(1))[0]
                if opt=='MSA':
                    dc={'SA':1,'MA':1,'DZ':1,'EG':1,'SY':1,'QA':1,'LB':1,'YE':1,'AE':1,'KW':1,'SD':1,'BH':1,'JO':1,'IQ':1,'PL':1,'OM':1,'LY':1,'TN':1}
            elif model == 'ArBert':
                dc = araBert_model(input,"arbert")
                print("XXXXXX"+str(dc))
                st.subheader('ArBert')
                a=dict(sorted(dc.items(), key = operator.itemgetter(1), reverse = True)[:4])
                pred = pd.DataFrame.from_dict(a, orient='index').rename(columns={0:'Country'})
                st.bar_chart(pred)
                opt = max(dc.items(), key=operator.itemgetter(1))[0]
                if opt=='MSA':
                    dc={'SA':1,'MA':1,'DZ':1,'EG':1,'SY':1,'QA':1,'LB':1,'YE':1,'AE':1,'KW':1,'SD':1,'BH':1,'JO':1,'IQ':1,'PL':1,'OM':1,'LY':1,'TN':1}
            i=1
        except:
            i=0
            with st.spinner('Wait for it...'):
                time.sleep(10)
            st.success('Done!')

    df = pd.DataFrame(list(dc.items()),columns=['Country', 'Value'])
    df['Country'] = df['Country'].map({'EG':'Egypt','SA':'Saudi Arabia','MA':'Morocco','DZ':'Algeria','SY':'Syria','QA':'Qatar','LB':'Lebanon','YE':'Yemen',
    'AE':'United Arab Emirates','KW':'Kuwait','SD':'Sudan','BH':'Bahrain','JO':'Jordan','IQ':'Iraq','PL':'Palestine','OM':'Oman','LY':'Libya','TN':'Tunisia'})
    st.write('Predicted dialect: ', opt)

    m = display_map(df)
    st_map = st_folium(m, width=1500, height=450)
