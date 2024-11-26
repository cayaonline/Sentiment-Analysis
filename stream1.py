import streamlit as st
import pandas as pd
from transformers import pipeline
from io import BytesIO
import time
import matplotlib.pyplot as plt

# Load the sentiment analysis pipeline
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device=0)

sentiment_analyzer = load_model()

# Sidebar
st.sidebar.title("Sentiment Analysis App")
st.sidebar.info("Upload an Excel file with a column named 'Text'. This app will analyze sentiments in the text and provide insights for APN clients.")

uploaded_file = st.sidebar.file_uploader("Upload an Excel file", type=["xlsx"])

# Main title
st.title("Sentiment Analysis Dashboard")

# Description
st.write("""
### About This App
This application performs sentiment analysis on uploaded text data, specifically tailored for APN clients. 
It processes the data to determine whether the sentiment is Positive, Neutral, or Negative, and visualizes the results in an easy-to-understand format.
""")

if uploaded_file:
    # Load and process the file
    try:
        df = pd.read_excel(uploaded_file)
        if "Text" not in df.columns:
            st.error("The uploaded file must contain a column named 'Text'.")
            st.stop()

        st.success("File uploaded successfully!")
        st.write("### Data Preview")
        st.write(df.head())

        # Perform sentiment analysis
        st.info("Performing sentiment analysis...")
        start_time = time.time()
        df['Sentiment'] = df['Text'].apply(lambda x: sentiment_analyzer(x)[0]['label'])
        end_time = time.time()
        df['Sentiment'] = df['Sentiment'].map({
            "LABEL_0": "Negative",
            "LABEL_1": "Neutral",
            "LABEL_2": "Positive"
        })

        st.success("Sentiment analysis complete!")
        st.write(f"Time taken: {end_time - start_time:.2f} seconds")

        # Display sentiment summary
        st.write("### Sentiment Distribution")
        sentiment_counts = df['Sentiment'].value_counts()

        # Display bar chart for overall sentiment distribution
        st.bar_chart(sentiment_counts)

        # Display detailed data visualization
        st.write("### Detailed Data with Sentiments")
        sentiment_groups = df.groupby(['Sentiment']).size().reset_index(name='Counts')

        fig, ax = plt.subplots()
        sentiment_groups.plot.bar(x='Sentiment', y='Counts', ax=ax, legend=False)
        ax.set_title("Sentiment Distribution by Counts")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Counts")
        st.pyplot(fig)

        # Allow download of results
        def to_excel(dataframe):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                dataframe.to_excel(writer, index=False)
            return output.getvalue()

        excel_data = to_excel(df)
        st.download_button(
            label="Download Results as Excel",
            data=excel_data,
            file_name="sentiment_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        st.error(f"Error loading file: {e}")