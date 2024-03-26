import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import io
import pickle

# Function to apply ML with error handling
def apply_ml(df, response_column):
    try:
        X_new = df[['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11']]
        X_new.dropna(inplace = True)
        with open(f'models/{response_column}_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        predictions = loaded_model.predict(X_new)
        df_out = X_new.copy()
        df_out[f"{response_column}"] = predictions
        return df_out
    except FileNotFoundError:
        st.error(f"Model for '{response_column}' not found. Please check the 'models' folder.")
    except KeyError:
        st.error("Required columns for prediction are missing in the data.")

# App layout
st.title("ML Response Variable Prediction")

# File upload and data validation
uploaded_file = st.file_uploader("Choose an Excel file")
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Response variable selection
    response_column = st.selectbox("Select the required response variable", ["R1", "R2", "R3"])

    if st.button("Apply ML"):
        try:
            df_out = apply_ml(df, response_column)
            st.table(df_out.head())

            # Plot with a legend
            fig, ax = plt.subplots()
            ax.plot(df_out[response_column], label=response_column)  # Add label for legend
            ax.legend()  # Display the legend
            st.pyplot(fig)

            # Download modified data
            buffer = io.BytesIO()
            df_out.to_excel(buffer)
            buffer.seek(0)
            st.download_button(label=' Download Excel File', data=buffer, file_name="data_with_response.xlsx")
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.write("Please upload an Excel file to use this app.")