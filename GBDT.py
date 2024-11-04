import requests
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time


url = "https://tianchi-media.oss-cn-beijing.aliyuncs.com/DSW/7XGBoost/train.csv"
response = requests.get(url)

if response.status_code == 200:
    data = StringIO(response.text)
    df = pd.read_csv(data)
    
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
  
    if 'RainTomorrow' in categorical_cols:
        categorical_cols.remove('RainTomorrow')

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    

    X = df.drop('RainTomorrow', axis=1)
    y = LabelEncoder().fit_transform(df['RainTomorrow'].astype(str))
    X_processed = preprocessor.fit_transform(X)
    
    #Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    #print(df.head())
else:
    print("Failed to load data")

start_time = time.time()
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
clf = clf.fit(X_train, y_train)
predictions_train = clf.predict(X_train)
predictions_test = clf.predict(X_test)


# accuracy
accuracy_train = accuracy_score(y_train, predictions_train)
accuracy_test = accuracy_score(y_test, predictions_test)

# precision
precision_train = precision_score(y_train, predictions_train, average='binary')
precision_test = precision_score(y_test, predictions_test, average='binary')

# recall
recall_train = recall_score(y_train, predictions_train, average='binary')
recall_test = recall_score(y_test, predictions_test, average='binary')

# F1 score
f1_train = f1_score(y_train, predictions_train, average='binary')
f1_test = f1_score(y_test, predictions_test, average='binary')

print("Training Metrics:")
print(f"Accuracy: {accuracy_train}")
print(f"Precision: {precision_train}")
print(f"Recall: {recall_train}")
print(f"F1 Score: {f1_train}")

print("\nTest Metrics:")
print(f"Accuracy: {accuracy_test}")
print(f"Precision: {precision_test}")
print(f"Recall: {recall_test}")
print(f"F1 Score: {f1_test}")
end_time = time.time()
print('Time:', end_time - start_time)