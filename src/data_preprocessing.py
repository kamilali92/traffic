import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def load_data(file_path='data/Accidents_sampled.csv'):
    """Load the traffic accident dataset."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the dataset by handling missing values and encoding categorical features."""
    # Ensure you do not call the same function recursively
    # Fill missing numeric values with column mean
    numeric_cols = df.select_dtypes(include=['number']).columns  # Select only numeric columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Fill missing categorical values with the most frequent value (mode)
    categorical_cols = df.select_dtypes(include=['object']).columns  # Select only categorical columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
     
    # Convert categorical features like weather and road condition to numeric using label encoding
    label_encoder = LabelEncoder()
    df['Weather_Condition'] = label_encoder.fit_transform(df['Weather_Condition'].astype(str))

        # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Apply label encoding to each categorical column
    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col].astype(str))
  

    # Split the dataset into features (X) and target (y)
    features = ['Weather_Condition','Humidity(%)','Temperature(F)', 'Visibility(mi)']
    X = df[features]  # Drop non-essential columns
    y = df['Severity']
    
    return X, y

def split_data(X, y, test_size=0.2):
    """Split the data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=42)

def scale_features(X_train, X_test):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
