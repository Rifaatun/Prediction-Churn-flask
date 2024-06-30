import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Function to train the model
def train_model(data):
    try:
        # Preprocessing data
        # Separate numerical and categorical data
        categorical_columns = ['Service_types', 'Packet_service', 'Media_transmisi', 'State', 'Partner', 'Type_contract', 'Complaint', 'Churn']
        numerical_columns = ['Bandwidth']

        # Encode categorical data
        label_encoders = {}
        for col in categorical_columns:
            label_encoders[col] = LabelEncoder()
            data[col] = label_encoders[col].fit_transform([data[col]])

        # Convert numerical data to float and normalize
        scaler = StandardScaler()
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

        # Define X and y for training
        X = data.drop('Churn', axis=1).values
        y = data['Churn'].values

        # Reshape X for LSTM
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Define BiLSTM model
        model = Sequential()
        model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(X.shape[1], 1)))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        print("Starting model training...")
        model.fit(X, y, epochs=10, batch_size=32)
        print("Model training completed.")

        # Save the model in .h5 format
        print("Saving model...")
        model.save('models/bilstm01.h5')
        print("Model saved successfully.")
        
        # Save the label encoders
        with open('models/label_encoders.pkl', 'wb') as f:
            pickle.dump(label_encoders, f)

        return 'Model trained and saved successfully.'

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Function to load the trained model and make predictions
def predict_model(input_data):
    try:
        # Load the trained model in .h5 format
        model = load_model('models/bilstm01.h5')

        # Preprocess input_data for prediction
        # Define categorical and numerical columns
        categorical_columns = ['Service_types', 'Packet_service', 'Media_transmisi', 'State', 'Partner', 'Type_contract', 'Complaint']
        numerical_columns = ['Bandwidth']

        # Encode categorical data
        label_encoders = {}
        for col in categorical_columns:
            label_encoders[col] = LabelEncoder()
            input_data[col] = label_encoders[col].fit_transform([input_data[col]])

        # Convert numerical data to float and normalize
        scaler = StandardScaler()
        input_data[numerical_columns] = scaler.fit_transform(input_data[numerical_columns])
        
        # Reshape input_data for LSTM model
        X = np.array(input_data).reshape(1, -1, 1)

        # Predict
        prediction = model.predict(X)

        # Interpret prediction
        result = 'Churn' if prediction[0][0] > 0.5 else 'Not Churn'

        return result

    except Exception as e:
        print(f'Error during prediction: {e}')
        return None

# Example input data (converted to DataFrame)
input_data = {
    'Service_types': ['Corporate'],
    'Packet_service': ['Permana Home'],
    'Media_transmisi': ['Fiber Optic'],
    'State': ['DKI Jakarta'],
    'Bandwidth': [100],  # Example numerical value for bandwidth
    'Partner': ['No'],
    'Type_contract': ['Yearly'],
    'Complaint': ['No'],
    'Churn': ['Yes']
}

# Convert dictionary to DataFrame
input_df = pd.DataFrame(input_data)

# Call the train_model function with DataFrame input
training_result = train_model(input_df)
print(f'Training result: {training_result}')

# Example input_data for prediction
input_data_predict = {
    'Service_types': 'Corporate',
    'Packet_service': 'Permana Home',
    'Media_transmisi': 'Fiber Optic',
    'State': 'DKI Jakarta',
    'Bandwidth': 100,  # Example numerical value for bandwidth
    'Partner': 'No',
    'Type_contract': 'Yearly',
    'Complaint': 'No'
}

# Convert dictionary to DataFrame
input_df_predict = pd.DataFrame([input_data_predict])

# Call predict_model function with input_data for prediction
prediction_result = predict_model(input_df_predict)

# Display the prediction result
print(f'Prediction result: {prediction_result}')