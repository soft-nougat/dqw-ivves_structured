# Welcome to the DQW Structured data repository! 🏗️

This repo contains the structured data DQW streamlit app code, however, the streamlit apps have been split into 5 for maintenance purposes:

- [Main Streamlit app 📊](https://share.streamlit.io/soft-nougat/dqw-ivves/app.py)
- [Tabular Data Section 🏗️](https://share.streamlit.io/soft-nougat/dqw-ivves_structured/main/app.py)
- [Audio Data Section 🎶](https://share.streamlit.io/soft-nougat/dqw-ivves_audio/main/app.py)
- [Text Data Section 📚](https://share.streamlit.io/soft-nougat/dqw-ivves_text/main/app.py)
- [Image Data Section 🖼️](https://share.streamlit.io/soft-nougat/dqw-ivves_images/main/app.py)


The packages used in the application are in the table below.

| App section                |     Description    |     Visualisation    |     Selection    |     Package             |
|----------------------------|--------------------|----------------------|------------------|-------------------------|
|     Synthetic tabular      |          x         |           x          |                  |     [table-evaluator](https://github.com/Baukebrenninkmeijer/table-evaluator)     |
|     Tabular                |          x         |           x          |                  |     [sweetviz](https://github.com/fbdesignpro/sweetviz)            |
|     Tabular                |          x         |           x          |                  |     [pandas-profiling](https://github.com/pandas-profiling/pandas-profiling)    |
|     Tabular, text          |                    |                      |         x        |     [PyCaret](https://github.com/pycaret/pycaret)             |

## Structured (tabular) data 

Key points addressed:
- Quantitative measures – number of rows and columns. 
- Qualitative measures – column types. 
- Descriptive statistics with NumPy for numeric columns, for example, count, mean, percentiles and standard deviation. For discrete columns, count, unique, top and frequency. 
- Explore missing data. 
- Examine outliers.  
- Mitigate class imbalance.
- Compare datasets, like train, test and evaluate data.
- Evaluate synthetic datasets.
- Create a quality report.

To complete the key points, 4 subsections are created:
- One file EDA with pandas-profiling
- One file preporcessing with PyCaret
- Two file comparison with Sweetviz 
- Synthetic data evaluation with table-evaluator
- In all the sections, there is an option to download a pdf/zip of the results

## How to run locally

1.	Installation process:

    Create virtual environment and activate it - https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
    
    Clone or download files from this repo
    
    Run pip install -r requirements.txt
    
    Run streamlit app.py to launch app

2.	Software dependencies:

    In requirements.txt

3.	Latest releases

    Use app.py