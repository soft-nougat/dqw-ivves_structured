import json
import streamlit as st
import pandas as pd
import base64
import streamlit.components.v1 as components
from PIL import Image
import pandas as pd

import io
from PIL import Image
from pprint import pprint
from zipfile import ZipFile
import os
from os.path import basename
import webbrowser

import shutil

def session_state_init():

    """
    Initialize session state!
    """

    # create ss object
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'pr' not in st.session_state:
        st.session_state.pr = None
    if 'pdf_report' not in st.session_state:
        st.session_state.pdf_report = None
    if 'sw' not in st.session_state:
        st.session_state.sw = None
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None  
    if 'transformed_data' not in st.session_state:
        st.session_state.transformed_data = None 
    if 'download' not in st.session_state:
        st.session_state.download = None

def create_temp_folder():
    """
    Create a temporary folder per unique user session
    The folder will be deleted at the end of a successful or unsuccessful run of app
    """

    # get session id and create session specific folders
    id = _get_session()
    # now create a user specific folder
    temp_folder = 'DQW_Temp_'+ id
    
    if os.path.exists(temp_folder) == False:
        os.makedirs(temp_folder)
        os.makedirs(temp_folder+'/preprocessed_data')
        os.makedirs(temp_folder+'/synthetic_data')

    return(temp_folder)

def _get_session():

    """
    Get unique user session ID
    """
    
    try:
        from streamlit.script_run_context import get_script_run_ctx
    except ModuleNotFoundError:
        # streamlit < 1.4
        from streamlit.report_thread import (  # type: ignore
            get_report_ctx as get_script_run_ctx,
        )
          
    from streamlit.server.server import Server        
    # Ref: https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92
   
    ctx = get_script_run_ctx()
    if ctx is None:
        raise Exception("Failed to get the thread context")

    return ctx.session_id

def remove_folder_contents(path):

    shutil.rmtree(path)

def app_section_button(option1, option2, option3, option4):

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # current page
        col1.markdown(option1) 
    with col2:
        st.markdown(option2, unsafe_allow_html=True) 
    with col3:
        st.markdown(option3, unsafe_allow_html=True) 
    with col4:
        st.markdown(option4, unsafe_allow_html=True) 
      

def app_meta(icon):

    # Set website details
    st.set_page_config(page_title ="Data Quality Wrapper", 
                       page_icon=icon, 
                       layout='centered')
    
    # set sidebar width
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 300px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 300px;
        margin-left: -300px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
    The bg will be static and won't take resolution of device into account.
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# set background, use base64 to read local file
def get_base64_of_bin_file(bin_file):
    """
    function to read png file 
    ----------
    bin_file: png -> the background image in local folder
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    """
    function to display png as bg
    ----------
    png_file: png -> the background image in local folder
    """
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    st.App {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

# display app header and sidebar
# use HTML code to set div
def display_app_header(main_txt,sub_txt,is_sidebar = False):
    """
    function to display major headers at user interface
    ----------
    main_txt: str -> the major text to be displayed
    sub_txt: str -> the minor text to be displayed 
    is_sidebar: bool -> check if its side panel or major panel
    """

    html_temp = f"""
    <h2 style = "color:#F74369; text_align:center; font-weight: bold;"> {main_txt} </h2>
    <p style = "color:#BB1D3F; text_align:center;"> {sub_txt} </p>
    </div>
    """
    if is_sidebar:
        st.sidebar.markdown(html_temp, unsafe_allow_html = True)
    else: 
        st.markdown(html_temp, unsafe_allow_html = True)
        
def display_app_header_1(sub_txt,is_sidebar = False):
    """
    function to display major headers at user interface
    ----------
    main_txt: str -> the major text to be displayed
    sub_txt: str -> the minor text to be displayed 
    is_sidebar: bool -> check if its side panel or major panel
    """

    html_temp = f"""
    <p style = "color:#BB1D3F; text_align:center;"> {sub_txt} </p>
    </div>
    """
    if is_sidebar:
        st.sidebar.markdown(html_temp, unsafe_allow_html = True)
    else: 
        st.markdown(html_temp, unsafe_allow_html = True)


def get_input(data_input_mthd,ss_text,is_batch=False,text_column = "text"):
    """
    function get input from user either by uploading a csv file, pasting text
    or importing json files
    ----------
    ss_text: string
    is_batch: bool 
    text_column: str -> the columnn name for creating pd.DataFrame is _is_batch is False
    """
    if 'CSV' in data_input_mthd:
        uploaded_file = st.file_uploader("Choose a csv file to analyse", type="csv")

        if uploaded_file is not None:
            st.success('File upload successful')
            df = pd.read_csv(uploaded_file)
            return df,ss_text
        else:
            st.info('Please upload a csv file')
            return pd.DataFrame(),ss_text

    elif 'Copy-Paste text' in data_input_mthd: 
        ss_text = st.text_area("Type in text to analyse", ss_text)
        df = pd.DataFrame(data=[ss_text],columns=[text_column])
        return df,ss_text
    
    elif 'json' in data_input_mthd:
        uploaded_file = st.file_uploader("Choose a json file to analyse",
                                         type = "json")

        if uploaded_file is not None:
            st.success('File upload successful')
            data = json.load(uploaded_file)
            df = pd.json_normalize(data)
            #df = pd.read_json(uploaded_file)
            return df,ss_text
        else:
            st.info('Please upload a json file')
            return pd.DataFrame(),ss_text
 
    
def check_input_method(data_input_mthd):
    """
    function check user input method if uploading or pasting or using
    a json file
    Parameters
    ----------
    data_input_mthd: str -> the default displayed text for decision making
    """
    # ----------------------------------------------
    # session state init
    st.session_state['is_file_uploaded'] = False
    st.session_state['is_batch_process'] = False
    st.session_state['txt'] = 'Paste the text to analyze here'

    if 'Copy-Paste text' in data_input_mthd:
        df, st.session_state.txt = get_input(data_input_mthd,
                                            ss_text= st.session_state.txt)


    elif 'CSV' in data_input_mthd:
        df,st.session_state.txt= get_input(data_input_mthd,
                                        ss_text= st.session_state.txt,
                                        is_batch=True)
        if df.shape[0]>0:
            st.session_state.is_batch_process = True
            st.session_state.is_file_uploaded = True
    elif 'json' in data_input_mthd:
        df,st.session_state.txt= get_input(data_input_mthd,
                             ss_text= st.session_state.txt,
                             is_batch=True)
        if df.shape[0]>0:
            st.session_state.is_batch_process = True
            st.session_state.is_file_uploaded = True
        
    return df,st.session_state.txt

def zip_files(dirName, ext):
    with ZipFile(dirName+ext, 'w') as zipObj:
        # Iterate over all the files in directory
        for folderName, subfolders, filenames in os.walk(dirName):
            for filename in filenames:
                #create complete filepath of file in directory
                filePath = os.path.join(folderName, filename)
                # Add file to zip
                zipObj.write(filePath, basename(filePath))

def generate_zip_structured(temp_folder, original, comparison):
    """ A function to write files to disk and zip 'em """
    original.to_csv(temp_folder+'/synthetic_data/reference_file_dqw.csv', 
               index=False)
    comparison.to_csv(temp_folder+'/synthetic_data/comparison_file_dqw.csv', 
               index=False)
    # create a ZipFile object
    zipObj = ZipFile(temp_folder+'/synthetic_data/report_files_dqw.zip', 'w')
    # Add multiple files to the zip
    zipObj.write(temp_folder+'/synthetic_data/reference_file_dqw.csv')
    zipObj.write(temp_folder+'/synthetic_data/comparison_file_dqw.csv')
    zipObj.write(temp_folder+'/synthetic_data/table-evaluator_comparison_dqw.pdf')
    # close the Zip File
    zipObj.close()

def generate_zip_pp(temp_folder, original, X, X_train, X_test, y, y_train, y_test):
    """ A function to write pycaret files to disk and zip 'em """

    original.to_csv(temp_folder+'/preprocessed_data/original_file.csv', index=False)

    if y is not None:
        
        X.to_csv(temp_folder+'/preprocessed_data/transformed_file.csv', index=False)
        X_train.to_csv(temp_folder+'/preprocessed_data/x_train.csv', index=False)
        X_test.to_csv(temp_folder+'/preprocessed_data/x_test.csv', index=False)

        y.to_csv(temp_folder+'/preprocessed_data/labels.csv', index=False)
        y_train.to_csv(temp_folder+'/preprocessed_data/y_train.csv', index=False)
        y_test.to_csv(temp_folder+'/preprocessed_data/y_test.csv', index=False)

    else:

        X.to_csv(temp_folder+'/preprocessed_data/transformed_file.csv', index=False)

    # create a ZipFile object
    dirName = temp_folder+'/preprocessed_data'
    with ZipFile(temp_folder+'/preprocessed_data.zip', 'w') as zipObj:
        # Iterate over all the files in directory
        for folderName, subfolders, filenames in os.walk(dirName):
            for filename in filenames:
                #create complete filepath of file in directory
                filePath = os.path.join(folderName, filename)
                # Add file to zip
                zipObj.write(filePath, basename(filePath))

def sub_text(text):
    '''
    A function to neatly display text in app.
    Parameters
    ----------
    text : Just plain text.
    Returns
    -------
    Text defined by html5 code below.
    '''
    
    html_temp = f"""
    <p style = "color:#1F4E79; text_align:justify;"> {text} </p>
    </div>
    """
    
    st.markdown(html_temp, unsafe_allow_html = True)
    
def open_html(file_name, height, width):

    """
    Open a local html file with streamlit components
    """

    pipe = open(file_name, 'r', encoding='utf-8')
    source_code = pipe.read() 
    components.html(source_code, height = height, width = width, scrolling=True)