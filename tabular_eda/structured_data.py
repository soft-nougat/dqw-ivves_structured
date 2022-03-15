"""
A script with the strucutured data analysis logic
Additional scripts: report_generation
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
import pycaret as pyc


from sklearn import set_config
from sklearn.utils import estimator_html_repr

from tabular_eda.one_file_analysis_section import one_file_section

def structured_data_app(temp_folder):

    show_intro()

    # Side panel setup
    display_app_header(main_txt = "Step 1",
                       sub_txt= "Choose type of analysis",
                       is_sidebar=True)

    selected_structure = st.sidebar.selectbox("", 
                                                ("Analyse 1 file", 
                                                "Compare 2 files",
                                                "Synthetic data comparison"))

    display_app_header(main_txt = "Step 2",
                    sub_txt= "Upload data",
                    is_sidebar=True)

    
    if selected_structure == "Analyse 1 file":

        # upload data
        st.session_state.data = upload_file()

        if st.session_state.data is not None:

            display_app_header(main_txt = "Step 3",
                            sub_txt= "Choose next step",
                            is_sidebar=True)
            
            # select subsection
            step_3 = st.sidebar.selectbox("",
            ("None", "EDA", "Preprocess and compare"))

            one_file_section(step_3, temp_folder, st.session_state.data)
           
    if selected_structure == "Compare 2 files":
        
        sweetviz_comparison(temp_folder, None, None, 0, text = "Step 3")
    
    if selected_structure == "Synthetic data comparison":
        
        table_evaluator_comparison(temp_folder)

def show_intro():

    # the very necessary reference expander
    intro_text = """
    Welcome to the DQW for structured data analysis.
    Structured data analysis is an important step 
    in AI model development or Data Analysis. This app 
    offers visualisation of descriptive statistics of a 
    csv input file. 
    <br> There are 3 options you can use: 
    <li> - Analyse 1 file using <a href = ""> pandas-profiling </a>
    <li> - Peprocess 1 file with <a href = "https://github.com/pycaret/pycaret"> PyCaret </a> 
    and compare with <a href = "https://github.com/fbdesignpro/sweetviz"> Sweetviz </a> and download preprocessing pipeline
    together with the datasets used
    <li> - Compare 2 files with Sweetviz
    <li> - Analyse synthetic data with <a href = "https://github.com/Baukebrenninkmeijer/table-evaluator"> table-evaluator </a> </li>
    <br> You can download pdf files/reports at the end of each analysis.
    """
    intro = st.expander("Click here for more info on this app section and packages used")
    with intro:
        sub_text(intro_text)

def upload_file():

    demo = st.sidebar.checkbox('Use demo data', value=False, help='Using the adult dataset')
    if demo:
        data = pd.read_csv('demo_data/tabular_demo.csv')
        return(data)
    else:
        data = st.sidebar.file_uploader("Upload dataset", 
                                type="csv") 

        if data:

            st.subheader('A preview of input data is below, please wait for data to be analyzed :bar_chart:')
            data = pd.read_csv(data)
            st.write(data.head(5))

            return(data)

        else:
            st.sidebar.warning("Please upload a dataset!")

            return(None)
        
            
def upload_2_files():
    """
    High level app logic when comparing 2 files
    """
    demo = st.sidebar.checkbox('Use demo data', value=False, help='Using the table-evaluator demo datasets')
    if demo:
        original = pd.read_csv('demo_data/real_test_sample.csv')
        comparison = pd.read_csv('demo_data/fake_test_sample.csv')
        indicator = 1
    else:
        original = st.sidebar.file_uploader("Upload reference dataset", 
                                            type="csv")

        if original:

            original = pd.read_csv(original)       
            indicator = 0                 

            comparison = st.sidebar.file_uploader("Upload comparison dataset", 
                                                    type="csv") 

            if comparison:                      
            
                comparison = pd.read_csv(comparison)
                indicator = 1

        else:
            st.sidebar.warning("Please upload a reference/original dataset.")
            indicator = 0
            return(None, None, indicator)

    # if data is available, continue with the app logic
    if indicator == 1:
        st.subheader('A preview of input files is below, please wait for data to be compared :bar_chart:')
        st.subheader('Reference data')
        st.write(original.head(5))
        st.subheader('Comparison data')
        st.write(comparison.head(5))

        return(original, comparison, 1)

def sweetviz_comparison(temp_folder, original, comparison, indicator, text):

    """
    Function to compare test and train data with sweetviz
    """
   
    # call high level function and get files
    original, comparison, indicator = upload_2_files()

    # use indicator to stop the app from running
    if indicator == 1: 

        sw = sv.compare([original, "Original"], [comparison, "Comparison"])

        sw.show_html(temp_folder+"/SWEETVIZ_REPORT.html", open_browser=False, layout='vertical', scale=1.0)

        display = open(temp_folder+"/SWEETVIZ_REPORT.html", 'r', encoding='utf-8')

        source_code = display.read()

        components.html(source_code, height=1200, scrolling=True)

        create_pdf_html(temp_folder+"/SWEETVIZ_REPORT.html",
                        text,
                        "sweetviz_dqw.pdf")

        return(sw)


def table_evaluator_comparison(temp_folder):

    """
    The portion of structured data app dedicated to file comparison with table-evaluator
    We have 2 options, plot differences or choose categorical column to analyse
    """

    # call high level function and get files
    original, comparison, indicator = upload_2_files()

    if indicator == 1: 
        # Side panel setup
        display_app_header(main_txt = "Step 3",
                        sub_txt= "Choose table-evaluator method",
                        is_sidebar=True)

        selected_method = st.sidebar.selectbox("", 
                                                ("Plot the differences", 
                                                "Compare model performance"))

        if selected_method == "Plot the differences":

            table_evaluator = TableEvaluator(original, comparison, temp_folder)
            table_evaluator.visual_evaluation()

            # Side panel setup
            display_app_header(main_txt = "Step 4",
                            sub_txt= "Download pdf report",
                            is_sidebar=True)

            with st.spinner("The pdf is being generated..."):
                create_pdf(original, comparison, temp_folder)
            st.success('Done! Please refer to sidebar, Step 4 for download.')

            zip = generate_zip_structured(temp_folder, original, comparison)

            with open(temp_folder+"/synthetic_data/report_files_dqw.zip", "rb") as fp:
                st.sidebar.download_button(
                        "⬇️",
                    data=fp,
                    file_name="te_compare_files_dqw.zip",
                    mime="application/zip"
                )

        else:
            
            # additional analysis part -------
            # insert an additional None column to options to stop the app 
            # from running on a wrong column
            dataset_columns = original.columns
            options_columns = dataset_columns.insert(0, 'None')
            
            evaluate_col = st.selectbox('Select the target column:', 
                                        options_columns, 
                                        index=0)
        
            if evaluate_col != 'None':

                table_evaluator = TableEvaluator(original, comparison, temp_folder)
                evaluate = table_evaluator.evaluate(target_col = evaluate_col)

            else:

                st.sidebar.warning('Please select a categorical column to analyse.')

        
def analyse_file(temp_folder, data):

    """
    The portion of structured data app dedicated to 1 file analysis with pandas-profiling
    """
    
    # generate a report and save it 
    pr = data.profile_report()
    st_profile_report(pr)
    pr.to_file(temp_folder+"/pandas_prof.html")
    
    create_pdf_html(temp_folder+"/pandas_prof.html",
                    "Step 4",
                    "pandas_profiling_dqw.pdf")

    return(pr)