# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 17:04:46 2020

Script with defined app, including styling.

@author: TNIKOLIC
"""

import streamlit as st
from PIL import Image
from helper_functions import *
from tabular_eda.structured_data import *

# app setup 
try:

    # create ss object
    if 'data' not in st.session_state:
        st.session_state.data = None

    # app design
    app_meta('🏗️')
    set_bg_hack('dqw_background.png')

    # set logo in sidebar using PIL
    logo = Image.open('logo.png')
    st.sidebar.image(logo, 
                        use_column_width=True)
    
    # hide warning for st.pyplot() deprecation
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    # Main panel setup
    display_app_header(main_txt='Data Quality Wrapper',
                       sub_txt='Clean, describe, visualise and select data for AI models')

    st.markdown("""---""")
    # provide options to user to navigate to other dqw apps
    app_section_button('Tabular Data Section 🏗️',
    '[Audio Data Section 🎶](https://share.streamlit.io/soft-nougat/dqw-ivves_audio/main/app.py)',
    '[Text Data Section 📚](https://share.streamlit.io/soft-nougat/dqw-ivves_text/main/app.py)',
    '[Image Data Section 🖼️](https://emojifinder.com/crane)')
    st.markdown("""---""")
    
    structured_data_app()

except KeyError:
    st.error("Please select a key value from the dropdown to continue.")
    
except ValueError:
    st.error("Oops, something went wrong. Please check previous steps for inconsistent input.")
    
except TypeError:
    st.error("Oops, something went wrong. Please check previous steps for inconsistent input.")
