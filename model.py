import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

class IrisClassificationModel:
    def __init__(self):
        # Load the iris dataset
        self.iris = load_iris()
        self.X = self.iris.data
        self.y = self.iris.target
        self.classes = ['setosa', 'versicolor', 'virginica']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )
        
        # Scale the features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Initialize the model (using KNeighborsClassifier instead of RandomForest)
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.model_path = os.path.join(os.path.dirname(__file__), 'iris_model.joblib')
        self.scaler_path = os.path.join(os.path.dirname(__file__), 'iris_scaler.joblib')
    
    def train_model(self):
        # Train the model
        self.model.fit(self.X_train_scaled, self.y_train)
        return self
    
    def evaluate_model(self):
        # Make predictions
        y_pred = self.model.predict(self.X_test_scaled)
        
        # Print classification report
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=self.iris.target_names))
        
        # Create confusion matrix
        cm = np.zeros((3, 3), dtype=int)
        for true, pred in zip(self.y_test, y_pred):
            cm[true, pred] += 1
        
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.iris.target_names, 
                    yticklabels=self.iris.target_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        return self
    
    def predict(self, features):
        """
        Predict the Iris species for given features
        """
        # Scale the input features
        features_scaled = self.scaler.transform([features])
        
        # Make prediction
        prediction_index = self.model.predict(features_scaled)[0]
        
        # Return the species name
        return self.iris.target_names[prediction_index]
    
    def save_model(self):
        """
        Save the trained model and scaler to disk
        """
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"Model saved to {self.model_path}")
        print(f"Scaler saved to {self.scaler_path}")
        return self
    
    def load_model(self):
        """
        Load a pre-trained model and scaler from disk
        """
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
        else:
            raise FileNotFoundError("Model file not found")
        return self

# If script is run directly, train and save the model
if __name__ == '__main__':
    model = IrisClassificationModel()
    model.train_model().evaluate_model().save_model()
    print("Model trained and saved successfully!")
