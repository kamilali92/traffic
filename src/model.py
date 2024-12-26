from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from src.data_preprocessing import load_data, preprocess_data, split_data, scale_features

def train_model():
    """Train a Random Forest Classifier to predict traffic accident severity."""
    # Load and preprocess data
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Scale the features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Classification Report: \n", classification_report(y_test, y_pred))
    
    return model

if __name__ == '__main__':
    train_model()
