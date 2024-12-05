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

class IrisClassificationModel:
    def __init__(self):
        # Load the iris dataset
        self.iris = load_iris()
        self.X = self.iris.data
        self.y = self.iris.target
        
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
    
    def save_model(self, model_path='iris_model.joblib', scaler_path='iris_scaler.joblib'):
        """
        Save the trained model and scaler to disk
        """
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        return self
    
    @classmethod
    def load_model(cls, model_path='iris_model.joblib', scaler_path='iris_scaler.joblib'):
        """
        Load a pre-trained model and scaler from disk
        """
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Create a new instance and replace model and scaler
        instance = cls()
        instance.model = model
        instance.scaler = scaler
        return instance

# If script is run directly, train and save the model
if __name__ == '__main__':
    model = IrisClassificationModel()
    model.train_model().evaluate_model().save_model()
    print("Model trained and saved successfully!")
