"""
An orchestration script for one file analysis
"""
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from tabular_eda.te import *
from helper_functions import display_app_header, generate_zip_structured, sub_text, generate_zip_pp, open_html
from tabular_eda.report_generation import create_pdf_html, create_pdf
import sweetviz as sv

import os, glob
from os import path
import tempfile

from sklearn import set_config
from sklearn.utils import estimator_html_repr

def one_file_section(step_3, temp_folder, data): 

     # Pandas profiling subsection --------------------------------------------
    if step_3 == "EDA":

        st.session_state.pr = analyse_file(temp_folder, st.session_state.data)

        st_profile_report(st.session_state.pr)

        st.session_state.pdf_report = create_pdf_html(temp_folder+"/pandas_prof.html",
                                    "Step 4",
                                    "pandas_profiling_dqw.pdf")
        
        # option to download in app
        display_app_header(main_txt = "Step 4",
                            sub_txt= "Download report",
                            is_sidebar=True)

        st.sidebar.download_button(
                "⬇️",
            data=st.session_state.pdf_report,
            file_name="pandas_profiling_dqw.pdf"
        )
    # PyCaret subsection ------------------------------------------------------     
    elif step_3 == "Preprocess and compare":
        
        pyc_info()

        st.write(st.session_state.download)
        st.write(st.session_state.transformed_data)

        dataset_columns = data.columns
        options_columns = dataset_columns.insert(0, 'None')
        
        # control flow - catch if the download has been triggered
        if st.session_state.download is not None:
           
            # allow the user to clear cache just for the dl button
            form = st.form("checkboxes", clear_on_submit = True)
            with form:
                st.checkbox('I want to preprocess my file again.')
                    
            submit = form.form_submit_button("Go! 🏃")
            if submit:
                st.session_state.download = None
                model = None
                label_col = None

        # class column is the label - ask the user to select - not necessary for unsupervised
        model = st.selectbox('Select the type of model you are preparing data for:',
        ('None', 'Unsupervised', 'Supervised'))

        if model != 'None':
            if model == 'Supervised':
                label_col = st.selectbox('Select the label column:', 
                                        options_columns, 
                                        index=0)
            else:
                label_col = None

            # run this part only if the download button is clear
            if st.session_state.download == None:
                    
                    preprocess(temp_folder, st.session_state.data, model, 
                    label_col, options_columns)

                    download_zip(temp_folder) 
        
    else:

        st.warning("Please select next step in sidebar.")

def download_zip(temp_folder):

    with open(temp_folder+"/preprocessed_data.zip", "rb") as fp:
        st.session_state.download = st.sidebar.download_button(
                "⬇️",
            data=fp,
            file_name="preprocessed_data_dqw.zip",
            mime="application/zip"
        )

        
@st.cache(allow_output_mutation=True)
def analyse_file(temp_folder, data):

    """
    The portion of structured data app dedicated to 1 file analysis with pandas-profiling
    """
    
    # generate a report and save it 
    pr = data.profile_report()
    
    pr.to_file(temp_folder+"/pandas_prof.html")
    
    return(pr)

def pyc_info():
    # show pycaret info
    pycaret_info = st.expander("Click here for more info on PyCaret methods used")
    with pycaret_info:
        text = """
        PyCaret is an exteremly useful low-code ML library. It helps automate ML workflows.
        <br>In this part of the app, you can pass a tabular dataset to the PyCaret setup function
        which runs the following preprocessing steps on your data:
        <br><li> - Missing value removal
        <li> - One-hot encoding of categorical features
        <li> - Outlier mitigation
        <li> - Target imbalance mitigation </li>
        <br> Why do we select the model we are preparing the data for? This is
        important as PyCaret's setup method works differently for supervised and unsupervised models.
        For the former, we need to specify a target column (label). For the latter, this is not neccessary.
        <br> The output of the setup function are:
        <li> - The preprocessed dataset, which you can compare with the original one using Sweetviz.
        <li> - The train and test datasets, which you can compare with each other using Sweetviz.
        """
        sub_text(text)

def preprocess(temp_folder, data, model, label_col, options_columns):
    """
    Automated preprocessing of the structured dataset w/ pycaret
    """

    # unsupervised
    if model == 'Unsupervised':

        from pycaret.clustering import setup, get_config, save_config

        pyc_user_methods = methods_pyc(options_columns, model)

        clf_unsup = setup(data = data, 
                          silent = True, 
                          numeric_imputation = pyc_user_methods[0],
                          categorical_imputation = pyc_user_methods[1],
                          ignore_features = pyc_user_methods[2],
                          high_cardinality_features = pyc_user_methods[3],
                          high_cardinality_method = pyc_user_methods[4],
                          #remove_outliers = pyc_user_methods[6],
                          #outliers_threshold = pyc_user_methods[7],
                          normalize = pyc_user_methods[8],
                          normalize_method = pyc_user_methods[9],
                          transformation = pyc_user_methods[10],
                          transformation_method = pyc_user_methods[11]
                          )

        # save pipeline
        save_config(temp_folder+"/preprocessed_data/pycaret_pipeline.pkl")

        # save html of the sklearn data pipeline
        set_config(display = 'diagram')

        pipeline = get_config('prep_pipe')

        with open(temp_folder+'/prep_pipe.html', 'w') as f:  
            f.write(estimator_html_repr(pipeline))

        show_pp_file(temp_folder, data, get_config('X'))

    # superivised
    elif model != 'Unsupervised':

        from pycaret.classification import setup,  get_config, save_config
  
        if label_col != 'None':
    
            pyc_user_methods = methods_pyc(options_columns, model)

            clf_sup = setup(data = data, 
                          silent = True, 
                          target = label_col, 
                          numeric_imputation = pyc_user_methods[0],
                          categorical_imputation = pyc_user_methods[1],
                          ignore_features = pyc_user_methods[2],
                          high_cardinality_features = pyc_user_methods[3],
                          high_cardinality_method = pyc_user_methods[4],
                          fix_imbalance = pyc_user_methods[5],
                          remove_outliers = pyc_user_methods[6],
                          outliers_threshold = pyc_user_methods[7],
                          normalize = pyc_user_methods[8],
                          normalize_method = pyc_user_methods[9],
                          transformation = pyc_user_methods[10],
                          transformation_method = pyc_user_methods[11]
                          )

            # save pipeline
            save_config(temp_folder+"/preprocessed_data/pycaret_pipeline.pkl")

            # save html of the sklearn data pipeline
            set_config(display = 'diagram')

            pipeline = get_config('prep_pipe')

            with open(temp_folder+'/prep_pipe.html', 'w') as f:  
                f.write(estimator_html_repr(pipeline))

            show_pp_file(temp_folder, data, get_config('X'), get_config('X_train'), get_config('X_test'),
            get_config('y'), get_config('y_train'), get_config('y_test'))



def show_pp_file(temp_folder, data, X, X_train = None, X_test = None, y = None, y_train = None, y_test = None):

    """
    Pass PyC files, show sklearn pipeline and zip everything so it cn be downloaded
    """
    
    st.subheader("Preprocessing done! 🧼")
    st.write("A preview of data and the preprocessing pipeline is below.")
    st.write(X.head())
    open_html(temp_folder+'/prep_pipe.html', height = 400, width = 300)

    st.subheader("Compare files 👀")
    st.write("Please use the comparison app section to compare files with Sweetviz.")

    # download files
    zip = generate_zip_pp(temp_folder, data, X, X_train, X_test, y, y_train, y_test)

    display_app_header(main_txt = "Step 4",
                    sub_txt= "Download preprocessed files",
                    is_sidebar=True)

def methods_pyc(columns, model):
    """
    Define which imputation method to run on missing values
    Define which features to ignore
    Define miscellaneous methods
    """

    st.subheader("Missing values")
    sub_text("Select imputation methods for both numerical and categorical columns.")
    imputation_num = st.selectbox("Select missing values imputation method for numerical features:",
    ("mean", "median", "zero"))

    imputation_cat = st.selectbox("Select missing values imputation method for categorical features:",
    ("constant", "mode"))

    sub_text("Select which columns to skip preprocessing for.")
    ignore = st.multiselect("Select which columns to ignore:",
    (columns), default = None)

    #if ignore != 'None':
        # if only 1 column is selected, we need to pass a list
        #if type(ignore) is str:
            #cardinal = [ignore]

    #sub_text("Select which columns you want to run ordinal one-hot encoding for. Ordinal features need to be in a specific order.")
    #sub_text("An example would be low < medium < high.")
    #ordinal = st.selectbox("Select ordinal columns:",
    #(columns))

    #ordinal_values = st.text_input("Write ordered ordinal values separated by a semi-colon (;)",
    #help="Example input: low; medium; high")
    st.subheader("Cardinal one-hot encoding")
    sub_text("Select the columns that have high cardinality, i.e., that contain variables with many levels.")
    cardinal = st.multiselect("Select cardinal columns:",
    (columns), default = None)

    cardinal_method = None

    if cardinal != 'None':
        # if only 1 column is selected, we need to pass a list
        #if type(cardinal) is str:
            #cardinal = [cardinal]

        text = """
        Below are the avaluable cardinality methods. If frequency is selected, the original value is replaced 
        with the frequency distribution. If clustering is selected, statistical attributes of data are clustered 
        and replaces the original value of the feature is replaced with the cluster label. 
        The number of clusters is determined using a combination of Calinski-Harabasz and Silhouette criteria.
        """
        sub_text(text)
        cardinal_method = st.selectbox("Select cardinal encoding method:",
        ("frequency", "clustering"))

    if model != 'Unsupervised':
        st.subheader("Resampling and bias mitigation")
        sub_text("When the training dataset has an unequal distribution of target class it can be fixed using the fix_imbalance parameter in the setup. When set to True, SMOTE (Synthetic Minority Over-sampling Technique) is used as a default method for resampling.")
        resampling = st.checkbox("Activate resampling")

        st.subheader("Outlier mitigation")
        sub_text("The Remove Outliers function in PyCaret allows you to identify and remove outliers from the dataset before training the model. Outliers are identified through PCA linear dimensionality reduction using the Singular Value Decomposition technique.")
        mitigation = st.checkbox("Activate outlier mitigation")

        mitigation_method = 0.05

        if mitigation:

            mitigation_method = st.slider("Pick mitigation threshold:",
            min_value = 0.01, max_value = 0.1, value = 0.05)
        
    else:
        resampling = None
        mitigation = None
        mitigation_method = None

    st.subheader("Normalize")
    sub_text("Normalization is a technique often applied as part of data preparation for machine learning. The goal of normalization is to rescale the values of numeric columns in the dataset without distorting differences in the ranges of values or losing information.")
    normalization = st.checkbox("Activate normalization")

    normalization_method = "zscore"

    if normalization:
       text="""
       <li> <b>Z-score</b> is a numerical measurement that describes a value's relationship to the mean of a group of values"
       <li> <b>minmax</b> scales and translates each feature individually such that it is in the range of 0 – 1
       <li> <b>maxabs</b> scales and translates each feature individually such that the maximal absolute value of each feature will be 1.0. Sparsity is left intact.
       <li> <b>robust</b> scales and translates each feature according to the Interquartile range. Use in case there's outliers.
       """
       sub_text(text)
       normalization_method = st.selectbox("Pick normalization method:",
        ("zscore", "minmax", "maxabs", "robust"))

    st.subheader("Feature Transform")
    sub_text(" Transformation changes the shape of the distribution such that the transformed data can be represented by a normal or approximate normal distribution.")
    feat_trans = st.checkbox("Activate feature transformation")

    feat_trans_method = "yeo-johnson"

    if feat_trans:
       text="""
        There are two methods available for transformation yeo-johnson and quantile.
       """
       sub_text(text)
       feat_trans_method = st.selectbox("Pick feature transformation method:",
        ("yeo-johnson", "quantile"))   

  
    return([imputation_num, imputation_cat, ignore, 
    cardinal, cardinal_method, resampling, 
    mitigation, mitigation_method, normalization, 
    normalization_method, 
    feat_trans, feat_trans_method])