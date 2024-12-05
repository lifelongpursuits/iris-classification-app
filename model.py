import os
import joblib
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

class IrisClassificationModel:
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.model_path = os.path.join(os.path.dirname(__file__), 'iris_model.joblib')
        self.iris = load_iris()
        self.X = self.iris.data
        self.y = self.iris.target
        self.classes = ['setosa', 'versicolor', 'virginica']

    def train(self):
        """Train the model using the iris dataset"""
        X_train, _, y_train, _ = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, features):
        """Predict the iris species"""
        prediction = self.model.predict([features])[0]
        return self.classes[prediction]

    def save_model(self):
        """Save the trained model"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def load_model(self):
        """Load the trained model"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            raise FileNotFoundError("Model file not found")

# If script is run directly, train and save the model
if __name__ == '__main__':
    model = IrisClassificationModel()
    model.train()
    model.save_model()
