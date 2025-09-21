import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path

train_path = Path("data/raw/housing.csv") 
processed_train_path = Path("data/train/housing_train_processed.csv") 

housing = pd.read_csv(train_path)
pipeline = Pipeline([
    ('scaler', StandardScaler())
])

num_features = housing.drop("median_house_value", axis=1).select_dtypes(include=['float64', 'int']).columns
housing[num_features] = pipeline.fit_transform(housing[num_features])

processed_train_path.parent.mkdir(parents=True, exist_ok=True) 
housing.to_csv(processed_train_path, index=False)

