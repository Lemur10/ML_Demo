from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
from pycaret.classification import *
import pandas as pd
import streamlit.components.v1 as components
import os 

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("data.PNG") 
    st.title("Automated Preliminary Data Exploration")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
    st.write("After you upload, click on profiling for data profile")
    st.info("This project helps you to explore your data.")

if choice == "Upload":
    st.title("First Upload a .csv Dataset")
    file = st.file_uploader("To Begin, Upload a .csv Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None) 
        st.dataframe(df)

if choice == "Profiling": 
            # extract meta-data from the uploaded dataset
            st.header("Meta-data")
            row_count = df.shape[0]
            column_count = df.shape[1]
             
            # Use the duplicated() function to identify duplicate rows
            duplicates = df[df.duplicated()]
            duplicate_row_count =  duplicates.shape[0]
         
            missing_value_row_count = df[df.isna().any(axis=1)].shape[0]
         
            table_markdown = f"""
              | Description | Value | 
              |---|---|
              | Rows | {row_count} |
              | Columns | {column_count} |
              | Duplicated Rows | {duplicate_row_count} |
              | Rows with Missing Values | {missing_value_row_count} |
              """
            st.markdown(table_markdown)

            st.write(' ')
            st.write(' ')    
            st.write('Numeric columns: ')    
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            formatted_cols = ', '.join(numeric_cols)
            st.write(formatted_cols)

            numeric_cols = df.select_dtypes(include='number').columns.tolist()
         
            # add selection-box widget
            selected_num_col = st.selectbox("Which numeric column do you want to explore?", numeric_cols)

            st.header(f"{selected_num_col} - Statistics")
                 
            col_info = {}
            col_info["Unique Values"] = len(df[selected_num_col].unique())
            col_info["Rows with Missing Values"] = df[selected_num_col].isnull().sum()
            col_info["Rows with 0"] = df[selected_num_col].eq(0).sum()
            col_info["Rows with Negative Values"] = df[selected_num_col].lt(0).sum()
            col_info["Average Value"] = df[selected_num_col].mean()
            col_info["Standard Deviation"] = df[selected_num_col].std()
            col_info["Minimum Value"] = df[selected_num_col].min()
            col_info["Maximum Value"] = df[selected_num_col].max()
            col_info["Median"] = df[selected_num_col].median()
             
            info_df = pd.DataFrame(list(col_info.items()), columns=['Description', 'Value'])
             
            # display dataframe as a markdown table
            st.dataframe(info_df)







if choice == "Modelling": 
    target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=target, silent=True)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")
