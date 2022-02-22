import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime import lime_tabular
import urllib
from urllib.request import urlopen
import requests
import configparser
import json
import plotly.express as px
from sklearn.cluster import KMeans
from joblib import load
from PIL import Image
import plotly.graph_objects as go
plt.style.use('fivethirtyeight')
sns.set_style('darkgrid')
st.set_page_config(layout="wide",page_title="credit_scoring"),

def main() :

    @st.cache
    def load_data():
        #path = "C:/Users/birro/Documents/projet_openclassrooms/"
        data = pd.read_csv('data_selectMille.csv', index_col='SK_ID_CURR', encoding ='utf-8')
        sample = pd.read_csv('X_test_sample700.csv', index_col='SK_ID_CURR', encoding ='utf-8')
        description = pd.read_csv("features_description.csv", usecols=['Row', 'Description'],  index_col=0, encoding= 'unicode_escape')#, encoding= 'unicode_escape'

        target = data.iloc[:, 0:]

        return data, sample, target, description


    def load_model():
        '''loading the trained model'''
        clf = load("best_model_lgbm.joblib")
        return clf


    @st.cache(allow_output_mutation=True)
    def load_knn(sample):
        knn = knn_training(sample)
        return knn


    @st.cache
    def load_infos_gen(data):
        
        lst_infos = [data.shape[0],
                     round((data["ANNUITY_INCOME_PERC"].mean())*100, 2),
                     round((data["PAYMENT_RATE"].mean())*100, 2)]
    
        nb_credits = lst_infos[0]
        income_annuity_moy_rate = lst_infos[1]
        payment_moy_rate = lst_infos[2]

        targets = data.TARGET.value_counts()

        return nb_credits, income_annuity_moy_rate, payment_moy_rate, targets


    def identite_client(data, id):
        data_client = data.loc[[id]]
        data_client['CODE_GENDER'] = data_client['CODE_GENDER'].map({0: 'Male', 1: 'Female'})
        data_client['FLAG_OWN_CAR'] = data_client['FLAG_OWN_CAR'].map({0: 'Yes', 1: 'No'})
        data_client['FLAG_PHONE'] = data_client['FLAG_PHONE'].map({0: 'Yes', 1: 'No'})
        data_client['FLAG_WORK_PHONE'] = data_client['FLAG_WORK_PHONE'].map({0: 'Yes', 1: 'No'})
        return data_client

    @st.cache
    def load_age_population(data):
        data_age = round((data["DAYS_BIRTH"]/365), 2)
        return data_age

    @st.cache
    def load_income_population(sample):
        df_income = pd.DataFrame(sample["PAYMENT_RATE"])
        df_income = df_income.loc[df_income['PAYMENT_RATE'] < 10000, :]
        return df_income
    
    

    @st.cache
    def load_prediction(sample, id, clf):
        score = clf.predict_proba(sample.loc[[id]])[:,1]
        return score


    @st.cache
    def load_kmeans(sample, id, mdl):
        index = sample.loc[id].index.values
        index = index[0]
        data_client = pd.DataFrame(sample.loc[sample.index, :])
        df_neighbors = pd.DataFrame(knn.fit_predict(data_client), index=data_client.index)
        df_neighbors = pd.concat([df_neighbors, data], axis=1)
        return df_neighbors.iloc[:,1:].sample(15)

    @st.cache
    def knn_training(sample):
        knn = KMeans(n_clusters=2).fit(sample)
        return knn 
    
    
    #Loading data……
    data, sample, target, description = load_data()
    id_client = sample.index.values
    #clf = load_model()


    #######################################
    # SIDEBAR
    #######################################

    #Title display
    
        
    
    html_temp = """
    <div style="background-color: silver; padding:10px; border-radius:12px">
    <h1 style="color: black; text-align:center">Prêt à dépenser : Credit Scoring</h1>
    </div>
    """
    # <p style="font-size: 20px; font-weight: bold; text-align:center">Credit decision support…</p>
    st.markdown(html_temp, unsafe_allow_html=True)
    
    image_logo = Image.open('logo.PNG')
    st.sidebar.image(image_logo)

    #Customer ID selection
    st.sidebar.header("**Customer General Info**")
    #st.sidebar.text("**General Info**")
    
    #st.info('This is a purely informational message')
    
    
    #Loading selectbox
    chk_id = st.sidebar.selectbox("Client ID", id_client)

    #Loading general info
    #nb_credits, rev_moy, credits_moy, targets = load_infos_gen(data)
    
    nb_credits, income_annuity_moy_rate, payment_moy_rate, targets = load_infos_gen(data)


    ### Display of information in the sidebar ###
    #Number of loans in the sample
    st.sidebar.markdown("<u>Number of loans in the sample :</u>", unsafe_allow_html=True)
    st.sidebar.text(nb_credits)

    #Average income
    st.sidebar.markdown("<u>Average Annuity/Income % :</u>", unsafe_allow_html=True)
    st.sidebar.text(income_annuity_moy_rate)

    #AMT CREDIT
    st.sidebar.markdown("<u>Average Payment Rate % :</u>", unsafe_allow_html=True)
    st.sidebar.text(payment_moy_rate)
    
    #PieChart
    #st.sidebar.markdown("<u>......</u>", unsafe_allow_html=True)
    
    
    
    fig, ax = plt.subplots(figsize=(5,5))
    #plt.pie(targets, explode=[0, 0.1], labels=['No default', 'Default'], autopct='%1.1f%%', startangle=90)
    sns.countplot(x=data['TARGET'])#, order=['No default', 'Default']
    st.sidebar.pyplot(fig)
        

    #######################################
    # HOME PAGE - MAIN CONTENT
    #######################################
    
    #Display Customer ID from Sidebar
    st.write("Customer ID selection :", chk_id)
        
    st.header("**Credit Decision**")
    
    # Deployement prediction  :
    #Appel de l'API : 
    #API_url = "http://127.0.0.1:5000/credit/" + str(chk_id)
    API_url = "https://app-birro.herokuapp.com/credit/" + str(chk_id)
   
    

    with st.spinner('Chargement du score du client...'):
        json_url = urlopen(API_url)
        API_data = json.loads(json_url.read())
        prediction = API_data['client risk in %']
    
    ## credit decision limit
    if prediction >= 50:
        color = "red"
        message = "Credit rejected"
    else:
        color = "green"
        message = "Credit granted"
        
    # gauge
    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = prediction,
        mode = "gauge+number",
        
        title = {'text': message},
        delta = {'reference': 100},
        gauge = {'axis': {'range': [None, 100]},
             'steps' : [
                 {'range': [0, 50], 'color': "green"},
                 {'range': [50, 100], 'color': "red"}],
             'bar': {'color': "gray"},
             'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 1, 'value': 50}}))

    fig.update_layout(font = {'color': color, 'family': "Arial"})
    st.plotly_chart(fig, use_container_width=True, sharing="streamlit")


    #Customer information display : Customer Gender, Age …
    st.header("**Customer information**")

    if st.checkbox("Show customer information ?"):
                       
        infos_client = identite_client(data, chk_id)
        st.write("**Customer Gender : **", infos_client["CODE_GENDER"].values[0])
        st.write("**Customer Age : **{:.0f} ans".format(int((infos_client["DAYS_BIRTH"]/365)*(-1))))
        st.write("**Customer Own Car ? : **", infos_client["FLAG_OWN_CAR"].values[0])
        st.write("**Customer Work Phone ? : **", infos_client["FLAG_WORK_PHONE"].values[0])
        

        #Age distribution plot
        data_age = load_age_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_age, edgecolor = 'k', color="gray", bins=10)
        ax.axvline(int((infos_client["DAYS_BIRTH"].values / 365)*(-1)), color="green", linestyle='--')
        ax.set(title='Customer age', xlabel='Age(Year)', ylabel='')
        st.pyplot(fig)
    
        
        st.subheader("*Annuity rate*")
        st.write("**Payment rate: **", infos_client["PAYMENT_RATE"].values[0])
        st.write("**Credit annuities / Income: **", infos_client["ANNUITY_INCOME_PERC"].values[0])

        
        #Income distribution plot
        data_income = load_income_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_income["PAYMENT_RATE"], edgecolor = 'k', color="gray", bins=10)
        ax.axvline(infos_client["PAYMENT_RATE"].values[0], color="green", linestyle='--')
        ax.set(title='Customer payment rate', xlabel='Payment Rate', ylabel='')
        st.pyplot(fig)
        
        #Relationship Age / Payment Rate interactive plot 
        data_sk = data.reset_index(drop=False)
        data_sk.DAYS_BIRTH = ((data_sk['DAYS_BIRTH']/365)*(-1)).round(1)
        fig, ax = plt.subplots(figsize=(10, 10))
        fig = px.scatter(data_sk, x='DAYS_BIRTH', y="PAYMENT_RATE", 
                         size="PAYMENT_RATE", color='CODE_GENDER') # ,

        fig.update_layout({'plot_bgcolor':'#f0f0f0'}, 
                          title={'text':"Relationship Age / Payment Rate", 'x':0.5, 'xanchor': 'center'}, 
                          title_font=dict(size=18, family='Verdana'), legend=dict(y=1.1, orientation='h'))


        fig.update_traces(marker=dict(line=dict(width=0.5, color='#3a352a')), selector=dict(mode='markers'))
        fig.update_xaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Age", title_font=dict(size=18, family='Verdana'))
        fig.update_yaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Payment Rate", title_font=dict(size=18, family='Verdana'))

        st.plotly_chart(fig)
    
    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)
        
    

    #Customer solvability display

    st.header("**Customer analysis**")

    
    if st.checkbox("Features importance global"):
        
        imageLocation = st.empty()
        img_color = Image.open("shap_value.png")
        imageLocation.image(img_color)
             
    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)

    #Feature importance / description
    if st.checkbox("Features importance customer {:.0f} ".format(chk_id)):
        shap.initjs()
        X = sample.iloc[:, :]
        #X = X[X.index == chk_id]
        #number = st.slider("Pick a number of features…", 0, 25, 5)

        #fig, ax = plt.subplots(figsize=(10, 10))
        #explainer = shap.TreeExplainer(load_model())
        #shap_values = explainer.shap_values(X)
        #shap.summary_plot(shap_values[0], X, plot_type ="bar", max_display=number, color_bar=False, plot_size=(5, 5))
        #st.pyplot(fig)
        
        # compute SHAP values
        st.text("*Les unités sur l'axe des x sont des unités de log-odds, donc des valeurs négatives impliquent des probabilités inférieures à 0,5 que le client soit en defaut de paiement")
        st.text("*Le texte gris avant les noms des caractéristiques indique la valeur de chaque caractéristique pour cet échantillon.")
        st.text("*La couleur bleue pour une variable donnée indique que la valeur de celle-ci diminue la probabilité que le client soit en defaut de paiement")
        explainer = shap.Explainer(load_model(), X)
        X = X[X.index == chk_id]
        shap_values = explainer(X)
        fig, ax = plt.subplots(figsize=(10, 10))
        shap.plots.waterfall(shap_values[0])
        st.pyplot(fig)
        
        if st.checkbox("lime value ?"):
            X.reset_index().drop('SK_ID_CURR', axis=1)
                            
            lime_explainer = lime_tabular.LimeTabularExplainer(X.to_numpy(), mode="classification", feature_names=X.columns,verbose=True)
            exp = lime_explainer.explain_instance(data_row=X.iloc[0], predict_fn=load_model().predict_proba, num_features=7)
            html = exp.as_html()
            components.html(html, height=800, width=800)#, height=1000, width=1000
        
        if st.checkbox("Need help about feature description ?") :
            list_features = description.index.to_list()
            feature = st.selectbox('Feature checklist…', list_features)
            st.table(description.loc[description.index == feature][:1])
        
    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)
        
        
    ###########################################################################"""
    st.markdown("<u>Customer Data :</u>", unsafe_allow_html=True)
    st.write(identite_client(data, chk_id))        
    

    #Similar customer files display
    chk_voisins = st.checkbox("Show similar customer ?")

    if chk_voisins:
        knn = load_knn(sample)
        st.markdown("<u>List of the 15 files closest to this Customer :</u>", unsafe_allow_html=True)
        st.dataframe(load_kmeans(sample, chk_id, knn))
        st.markdown("<i>Target 1 = Customer with default</i>", unsafe_allow_html=True)
    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)
        
        
    st.markdown('***')
    #st.markdown("Thanks for going through this Web App with me! I'd love feedback on this, so if you want to reach out you can find Code from [Github](https://github.com/DeepScienceData/Projet-OpenClassRoms)* ❤️")


if __name__ == '__main__':
    main()
