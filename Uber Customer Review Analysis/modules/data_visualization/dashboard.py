import traceback

import streamlit as st
import pandas as pd
import plotly.express as px
from modules.data_preparation.data_preparation import prepare_data
from modules.data_visualization.review_length_pie_chart import create_review_length_pie_chart


def create_dashboard():
    st.title("Uber Customer Review Dashboard")
    st.write("This dashboard has the data visualizations for my Uber Customer Review Project")

    # Upload the data
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        # sidebar filters
        category = st.sidebar.selectbox("Word Count", data["word_count"].unique())

        # Filtered Data
        filtered_data = data[data['word_count'] == category]

        # Visualization
        fig = px.bar(filtered_data, x='subcategory', y='value', title="Filtered Data")
        st.plotly_chart(fig)


if __name__ == '__main__':
    try:
        create_dashboard()
    except Exception as e:
        print("An error has occurred:")
        print(e)
        traceback.print_exc()
