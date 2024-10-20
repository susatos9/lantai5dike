from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv('train.csv')

# Preprocessing
# Define numerical and categorical features
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Imputers
numerical_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# StandardScaler for numerical features
scaler = StandardScaler()

# OneHotEncoder for categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

# ColumnTransformer to apply transformations to respective columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_imputer, numerical_cols),
        ('cat', categorical_imputer, categorical_cols)
    ])

# Define the PCA
pca = PCA(n_components=0.95)  # Adjust n_components as needed

# Ridge Regression model
ridge_model = Ridge(alpha=1.0)  # Adjust alpha as needed

# Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('encoder', encoder),
    ('scaler', scaler),
    ('pca', pca),
    ('model', ridge_model)
])

# Split data into features and target
X = data.drop('Harga', axis=1)
y = data['Harga']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Predict on the testing set
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
