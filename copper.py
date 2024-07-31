import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer

# -------------------------------This is the configuration page for our Streamlit Application---------------------------
st.set_page_config(
    page_title="Industrial Copper Modeling",
    layout="wide"
)

# -------------------------------This is the sidebar in a Streamlit application, helps in navigation--------------------
with st.sidebar:
    selected = option_menu("Main Menu", ["About Project", "Predictions"],
                           icons=["house", "gear"],
                           styles={"nav-link": {"font": "sans serif", "font-size": "20px", "text-align": "centre"},
                                   "nav-link-selected": {"font": "sans serif", "background-color": "#0072b1"},
                                   "icon": {"font-size": "20px"}
                                   }
                           )


# -----------------------------------------------About Project Section--------------------------------------------------
if selected == "About Project":
    st.markdown("# :blue[Industrial Copper Modelling]")
    st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
    st.markdown("### :blue[Technologies :] Python, Pandas, Numpy, Scikit-Learn, Streamlit, Python scripting, "
                "Machine Learning, Data Preprocessing, Visualization, EDA")
    st.markdown("### :blue[Overview :] The objective of this project is to create two machine learning models "
                "specifically designed for the copper business. These models will be developed to tackle the "
                "difficulties associated with accurately anticipating selling prices and effectively classifying lead. "
                "The process of making forecasts manually may be a time-intensive task and may not provide appropriate "
                "price choices or effectively collect leads. The models will employ sophisticated methodologies, "
                "including data normalization, outlier detection and treatment, handling of improperly formatted data, "
                "identification of feature distributions, and utilization of tree-based models, particularly the "
                "decision tree algorithm, to accurately forecast the selling price and leads.")
    st.markdown("### :blue[Domain :] Manufacturing")
    st.markdown("### :blue[BY :] Mohana Anjan A V ")

# ------------------------------------------------Predictions Section---------------------------------------------------
if selected == "Predictions":
    st.markdown("# :blue[Predicting Results based on Trained Models]")

    tab1, tab2 = st.tabs(["Predict Selling Price", " Predict Status"])

    country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]

    status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered',
                      'Offerable']

    item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']

    application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67.,
                           79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]

    product = ['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665',
               '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407',
               '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662',
               '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738',
               '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']

    # ------------------------------Predict Selling Price------------------------------
    try:
        with tab1:
            st.markdown("### :orange[Predicting Selling Price (Regression Task) (Accuracy: 91%)]")
            with st.form("form1"):
                col1, col2, col3 = st.columns([5, 2, 5])

                # -----New Data inputs from the user for predicting the selling price-----
                with col1:
                    status = st.selectbox("Status", status_options, key=1)
                    item_type = st.selectbox("Item Type", item_type_options, key=2)
                    country = st.selectbox("Country", sorted(country_options), key=3)
                    application = st.selectbox("Application", sorted(application_options), key=4)
                    product_ref = st.selectbox("Product Reference", product, key=5)
                with col3:
                    quantity_tons = st.number_input('Enter Quantity in Tons (Min: 0.00001 and Max: 1000000000.0)',
                                                    min_value=0.00001, max_value=1000000000.0)
                    thickness = st.number_input('Enter Thickness (Min: 0.18 and Max: 400.0)', min_value=0.18,
                                                max_value=400.0)
                    width = st.number_input('Enter Width (Min: 1.0 and Max: 2990.0)', min_value=1.0, max_value=2990.0)
                    customer = st.number_input('Enter Customer ID')

                    # -----Submit Button for PREDICT SELLING PRICE-----
                    submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")

                    if submit_button is not None:
                        with open(r"model.pkl", 'rb') as file:
                            loaded_model = pickle.load(file)
                        with open(r'scaler.pkl', 'rb') as f:
                            scaler_loaded = pickle.load(f)
                        with open(r"rohe.pkl", 'rb') as f:
                            ohe_loaded = pickle.load(f)
                        with open(r"rohe2.pkl", 'rb') as f:
                            ohe2_loaded = pickle.load(f)

                        # -----Sending that data to the trained models for selling price prediction-----
                        new_sample = np.array([[np.log(float(quantity_tons)), application, np.log(float(thickness)),
                                                float(width), country, float(customer), int(product_ref), item_type,
                                                status]])
                        new_sample_ohe = ohe_loaded.transform(new_sample[:, [7]]).toarray()
                        new_sample_ohe2 = ohe2_loaded.transform(new_sample[:, [8]]).toarray()
                        new_sample = np.concatenate(
                            (new_sample[:, [0, 1, 2, 3, 4, 5, 6]], new_sample_ohe, new_sample_ohe2),
                            axis=1)
                        new_sample1 = scaler_loaded.transform(new_sample)
                        new_pred = loaded_model.predict(new_sample1)[0]

                        # Used np.log earlier to handle data discrepancies, so to get the real output using np.exp
                        st.write('## :green[Predicted selling price:] ', np.exp(new_pred))

    except Exception as e:
        st.write("Error")

    # ------------------------------Predict Status------------------------------
    try:
        with tab2:
            st.markdown("### :orange[Predicting Status (Classification Task) (Accuracy: 72%)]")
            with st.form("form2"):
                col1, col2, col3 = st.columns([5, 1, 5])

                # -----New Data inputs from the user for predicting the status-----
                with col1:
                    citem_type = st.selectbox("Item Type", item_type_options, key=21)
                    ccountry = st.selectbox("Country", sorted(country_options), key=31)
                    capplication = st.selectbox("Application", sorted(application_options), key=41)
                    cproduct_ref = st.selectbox("Product Reference", product, key=51)

                with col3:
                    cquantity_tons = st.number_input('Enter Quantity in Tons (Min: 0.00001 and Max: 1000000000.0)',
                                                    min_value=0.00001, max_value=1000000000.0)
                    cthickness = st.number_input('Enter Thickness (Min: 0.18 and Max: 400.0)', min_value=0.18,
                                                max_value=400.0)
                    cwidth = st.number_input('Enter Width (Min: 1.0 and Max: 2990.0)', min_value=1.0, max_value=2990.0)
                    ccustomer = st.number_input('Enter Customer ID')
                    cselling = st.number_input('Selling Price', min_value=0.1, max_value=100001015.0)

                    # -----Submit Button for PREDICT STATUS-----
                    csubmit_button = st.form_submit_button(label="PREDICT STATUS")

                    if csubmit_button is not None:
                        if cquantity_tons and cselling:
                            with open(r"cmodel.pkl", 'rb') as file:
                                cloaded_model = pickle.load(file)
                            with open(r'cscaler.pkl', 'rb') as f:
                                cscaler_loaded = pickle.load(f)
                            with open(r"cohe.pkl", 'rb') as f:
                                ct_loaded = pickle.load(f)

                            # -----Sending that data to the trained models for status prediction-----
                            new_sample = np.array(
                                [[np.log(float(cquantity_tons)), np.log(float(cselling)), capplication,
                                  np.log(float(cthickness)), float(cwidth), ccountry, int(ccustomer),
                                  int(product_ref), citem_type]])
                            new_sample_ohe = ct_loaded.transform(new_sample[:, [8]]).toarray()
                            new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, 6, 7]], new_sample_ohe),
                                                        axis=1)
                            new_sample = cscaler_loaded.transform(new_sample[:, :12])
                            new_pred = cloaded_model.predict(new_sample)
                            if new_pred == 1:
                                st.write('## :green[The Status is Won] ')
                            else:
                                st.write('## :red[The status is Lost] ')

    except Exception as e:
        st.write()