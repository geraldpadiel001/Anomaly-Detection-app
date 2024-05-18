import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import cv2
import plotly.express as px
from PIL import Image

from datetime import datetime
import streamlit as st
import warnings
import tempfile
import os
import shutil
from collections import Counter

# Suppress InconsistentHashingWarning (use with caution)
warnings.filterwarnings("ignore", category=UserWarning, message="InconsistentHashingWarning")
st.title("Anomaly detection of illegal drugs and Substance System")

# Anomaly detection of illegal drugs and substance system
st.sidebar.image("back.png", width=300,)
st.balloons()
st.sidebar.subheader("Anomaly detection of illegal drugs and substance")

# Create a slider for transparency
transparency = st.sidebar.select_slider("Select a Drug", ["Cocaine", "Heroin", "Marijuana", "Methamphetamine", "LSD"])

menu = st.sidebar.radio("Menu", ["Home","Scan_objects"])
if menu == "Home":
    st.header('Survey Results')
    st.subheader('Classifying substances into categories based on their consuption')

    # --- LOAD DATAFRAME
    excel_file = 'Survey_Results.xlsx'
    sheet_name = 'DATA'

    df = pd.read_excel(excel_file, sheet_name=sheet_name, usecols='B:D', header=3)

    df_participants = pd.read_excel(excel_file, sheet_name=sheet_name, usecols='F:G', header=3)
    df_participants.dropna(inplace=True)

    # --- PLOT PIE CHART
    pie_chart = px.pie(df_participants,
                       title='Total No. of Participants',
                       values='Participants',
                       names='Departments')

    st.plotly_chart(pie_chart)

elif menu == "Scan_objects":
    # Create a folder to save the uploaded images
    UPLOAD_FOLDER = "uploads"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def save_uploaded_image(uploaded_file):
    if uploaded_file is not None:
        # Generate a new filename based on the original filename and current date and time
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        original_filename, file_extension = os.path.splitext(uploaded_file.name)
        new_filename = f"{original_filename}_{current_datetime}{file_extension}"

        # Save the uploaded image to the specified folder with the new filename
        image_path = os.path.join(UPLOAD_FOLDER, new_filename)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return image_path

def main():
    # Step 1: Load the dataset
    df = pd.read_csv('training_dataset.csv')

    # Step 2: Prepare the data
    X = df['Image']  # Image paths
    y = df['Label']  # Labels

    # Step 3: Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Feature Extraction
    # Define a function to extract features from image paths
    def extract_features(image_paths):
        features = [image.split('\\')[-1].split('.')[0] for image in image_paths]  # Extract file names without extensions
        return features

    # Extract features for training and testing data
    X_train_features = extract_features(X_train)
    X_test_features = extract_features(X_test)

    # Step 5: Model Training
    # For illustration purposes, let's use a simple SVM classifier with CountVectorizer for feature extraction
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train_features)
    X_test_vectorized = vectorizer.transform(X_test_features)

    clf = SVC(kernel='linear')
    clf.fit(X_train_vectorized, y_train)

    # Step 6: Model Evaluation
    y_pred = clf.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    stats_x_y = classification_report(y_test, y_pred)

    # Step 7: Apply Data Augmentation
    def apply_augmentation(image):
        # Apply transformations
        augmented_images = []
        augmented_images.append(cv2.flip(image, 1))  # Horizontal flip
        augmented_images.append(cv2.flip(image, 0))  # Vertical flip

        augmented_images.append(np.rot90(image))  # Rotate 90 degrees clockwise
        augmented_images.append(np.rot90(image, k=3))  # Rotate 90 degrees counterclockwise
        
        return augmented_images

    def draw_final_result(predictions):  
        #Statisticals concludes
        counted_predictions = Counter(predictions)  

        most_common_predictions = counted_predictions.most_common()
        most_common_prediction = most_common_predictions[0][0]

        return most_common_prediction

    # Allow user to upload an image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if st.button("Detect"):
        # Save the uploaded image to the folder
        image_path = save_uploaded_image(uploaded_file)
        
        if image_path:
            
            with st.spinner("Generating response..."):             
                # Step 8: Use Augmented Images for Prediction
                user_input_image_path = image_path
                  # Image path
                user_input_image = cv2.imread(user_input_image_path)  # Read the image

                # Apply data augmentation
                augmented_images = apply_augmentation(user_input_image)

                augmented_features = extract_features([user_input_image_path] * len(augmented_images))

                # Extract features from augmented images
                augmented_vectorized = vectorizer.transform(augmented_features)

                # Make predictions on augmented images
                augmented_predictions = clf.predict(augmented_vectorized)


            #Display an informational message indicating a cross-validation report using Streamlit's info function
            st.info("Cross-validation Report")

            # Split the stats_x_y string by newline characters to get individual rows and store them in a list
            rows = stats_x_y.split('\n')

            # Split each row by whitespace characters to separate the values and store them in a list of lists
            rows = [row.split() for row in rows[2:-5]]

            # Create a DataFrame from the list of lists, specifying column names
            df_stats_x_y = pd.DataFrame(rows, columns=['class', 'precision', 'recall', 'f1-score', 'support'])
            
            # Display the DataFrame as a table using Streamlit's table function
            st.table(df_stats_x_y)    

            # Compute the final result by determining the most common prediction from the augmented_predictions list
            final_result = draw_final_result(augmented_predictions)

            # Display an informational message indicating the model's response using Streamlit's info function
            st.info("Model Response:")

            # Concatenate the final result with a message and convert it to uppercase, then store it in the variable 'ans'
            ans = "Object / Substance From Picture Is: " + (str(final_result).upper())

            # Display the result message with a success indicator using Streamlit's success function
            st.success(ans)


        else:
            st.error("No image uploaded!")



    st.markdown("---")

    st.sidebar.header("About")

    st.sidebar.write("This Model Detects illegal and none illegal drugs from scanning pictures")
    
if __name__ == "__main__":
    main()

