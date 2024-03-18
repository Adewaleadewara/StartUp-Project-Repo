import streamlit as st
import pandas as pd 
import warnings 
warnings.filterwarnings('ignore')
import joblib

# st.title('START UP PROJECT')
# st.subheader('Built By Gomycode Daintree')

st.markdown("<h1 style = 'color: #FFC700; text-align: center; font-family: trebuchet ms'>STARTUP PROFIT PREDICTOR</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: tangerine'> Built By Gomycode Data Science Daintree</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)

st.image('pngwing.com.png', use_column_width=True)
st.header('Project Background Information', divider = True)
st.write("The overarching objective of this ambitious project is to meticulously engineer a highly sophisticated predictive model meticulously designed to meticulously assess the intricacies of startup profitability. By harnessing the unparalleled power and precision of cutting-edge machine learning methodologies, our ultimate aim is to furnish stakeholders with an unparalleled depth of insights meticulously delving into the myriad factors intricately interwoven with a startup's financial success. Through the comprehensive analysis of extensive and multifaceted datasets, our mission is to equip decision-makers with a comprehensive understanding of the multifarious dynamics shaping the trajectory of burgeoning enterprises. Our unwavering commitment lies in empowering stakeholders with the indispensable tools and knowledge requisite for making meticulously informed decisions amidst the ever-evolving landscape of entrepreneurial endeavors.")

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

data = pd.read_csv('startUp.csv')
st.dataframe(data)

st.sidebar.image('rag-doll-with-blue-cap-checklist.jpg', caption='Welcome User')

st.sidebar.markdown("<br>", unsafe_allow_html= True)
st.sidebar.markdown("<br>", unsafe_allow_html= True)

# Decleare user input variables
st.sidebar.subheader('Input Variable', divider=True)
rd_spend = st.sidebar.number_input('Research and Development Expense')
admin = st.sidebar.number_input('Administrative Expense')
mkt = st.sidebar.number_input('Marketing Expense')

# Display the users-input
input_var = pd.DataFrame()
input_var['R&D Spend'] = [rd_spend]
input_var['Administration'] = [admin]
input_var['Marketing Spend'] = [mkt]

st.markdown("<br>", unsafe_allow_html= True)
## Display the users input variable
st.subheader('Users Input Variables', divider=True)
st.dataframe(input_var)

# Importing the model Scalers
rd_spend = joblib.load('R&D Spend_scaler.pkl')
admin = joblib.load('Administration_scaler.pkl')
mkt = joblib.load('Marketing Spend_scaler.pkl')
# Transform the users input with the imported scalers
input_var['R&D Spend'] = rd_spend.transform(input_var[['R&D Spend']])
input_var['Administration'] = admin.transform(input_var[['Administration']])
input_var['Marketing Spend'] = mkt.transform(input_var[['Marketing Spend']])

# st.dataframe(input_var)
model = joblib.load('StartUpModel.pkl')
predicted = model.predict(input_var)

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)


# Creating prediction and interpretation tab
prediction, interprete = st.tabs(['Model Prediction', 'Model Interpretation'])
with prediction:
    if prediction.button('Predicted Profit'):
        prediction.balloons()
        prediction.success(f'The Predicted profit for your organisation is: {predicted[0].round(2)}')
with interprete:
    intercept = model.intercept_
    coef  = model.coef_
    interprete.write(f'A percentage increase in Research and Development Expense make Profit to increase by {coef[0].round(2)} Naira')
    interprete.write(f'A percentage increase in Administration Expense make Profit to increase by {coef[0].round(2)} Naira')
    interprete.write(f'A percentage increase in Marketing Spend Expense make Profit to increase by {coef[0].round(2)} Naira')
    interprete.write(f'The value of Profit when none of these expenses were made is {intercept.round(2)} Naira')