import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Load the dataset
file_path = "Final_Cleaned_Dataset.csv"
data = pd.read_csv(file_path, low_memory=False)

# Remove statistical outliers
def remove_statistical_outliers(df, column, n_std=3):
    log_values = np.log1p(df[column])
    median = log_values.median()
    std = log_values.std()
    mask = (log_values - median).abs() <= (n_std * std)
    return df[mask]

# Remove outliers from Price
data = remove_statistical_outliers(data, 'Price', n_std=3)
data = data[data['Price'] > 0]

# Apply log transformation
data['Log_Price'] = np.log1p(data['Price'])

# Handle Cylinders
if 'Cylinders' in data.columns:
    data['Cylinders'] = data['Cylinders'].str.extract(r'(\d+)').astype(float)
    data['Cylinders'] = data['Cylinders'].fillna(data['Cylinders'].median())

# Add enhanced features
current_year = 2024
data['Age'] = current_year - data['Year']

# Enhanced depreciation calculation
def calculate_depreciation(row):
    age = row['Age']
    mileage = row['Odometer']
    base_price = row['Price']
    
    # Age-based depreciation (steeper in early years)
    if age <= 1:
        age_factor = 0.8  # 20% first year
    elif age <= 3:
        age_factor = 0.85 ** age  # 15% years 2-3
    elif age <= 5:
        age_factor = 0.88 ** age  # 12% years 4-5
    else:
        age_factor = 0.9 ** age  # 10% after year 5
    
    # Mileage-based depreciation
    mileage_factor = 1.0
    if mileage > 0:  # Only apply if mileage is available
        mileage_depreciation = (mileage / 100000) * 0.2  # 20% per 100k miles
        mileage_factor = max(0.5, 1 - mileage_depreciation)  # Floor at 50% value
    
    return base_price * age_factor * mileage_factor

# Apply enhanced depreciation
data['Depreciated_Price'] = data.apply(calculate_depreciation, axis=1)

# Add car segment feature based on initial price
data['Price_Segment'] = pd.qcut(data['Price'], q=5, labels=['Budget', 'Economy', 'Mid-range', 'Premium', 'Luxury'])

# Bucketize Year with smaller ranges
data['Year_Bucket'] = pd.cut(
    data['Year'],
    bins=[1980, 1990, 2000, 2005, 2010, 2015, 2020, 2025],
    labels=['1980s', '1990s', '2000-2005', '2005-2010', '2010-2015', '2015-2020', '2020+']
)

# Create interaction features
if 'Type' in data.columns:
    if data['Type'].dtype == 'object':
        data['Type'] = data['Type'].fillna('Unknown')
        le_type = LabelEncoder()
        le_type.fit(data['Type'])
        data['Type'] = le_type.transform(data['Type'])

# Scale Year with reduced importance
scaler = StandardScaler()
data['Year_Scaled'] = scaler.fit_transform(data[['Year']]) * 0.5  # Reduce year impact

# Encode categorical features
categorical_features = ['Make', 'Model', 'Condition', 'Fuel', 'Title_status', 
                       'Transmission', 'Drive', 'Type', 'Paint_color', 'Year_Bucket', 'Price_Segment']
data[categorical_features] = data[categorical_features].astype('object')
label_encoders = {}
for feature in categorical_features:
    if feature in data.columns:
        data[feature] = data[feature].fillna('Unknown')
        le = LabelEncoder()
        le.fit(data[feature])
        data[feature] = le.transform(data[feature])
        label_encoders[feature] = le

# Define features and target
features = ['Year_Scaled', 'Make', 'Model', 'Condition', 'Cylinders', 'Fuel', 'Odometer', 
            'Title_status', 'Transmission', 'Drive', 'Type', 'Paint_color', 'Age', 
            'Price_Segment']
target = 'Depreciated_Price'

X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:\nRMSE: ${rmse:,.2f}\nR²: {r2:.4f}")

# Save the model and encoders
joblib.dump(model, "car_price_model.pkl")
joblib.dump(label_encoders, "car_label_encoders.pkl")
print("Model and encoders saved successfully.")

# Example prediction
example_car = pd.DataFrame({
    'Year': [2020],
    'Make': ['Ford'],
    'Model': ['Focus'],
    'Condition': ['Used'],
    'Cylinders': [4.0],
    'Fuel': ['Gasoline'],
    'Odometer': [30000],
    'Title_status': ['Clean'],
    'Transmission': ['Automatic'],
    'Drive': ['FWD'],
    'Type': ['Sedan'],
    'Paint_color': ['Red']
})

# Add Age feature
example_car['Age'] = current_year - example_car['Year']

# Determine Price_Segment based on similar cars in the dataset
# Determine Price_Segment based on similar cars in the dataset
similar_cars = data[
    (data['Make'] == label_encoders['Make'].transform(['Ford'])[0]) & 
    (data['Model'] == label_encoders['Model'].transform(['Focus'])[0]) & 
    (abs(data['Year'] - 2020) <= 2)  # Cars within 2 years
]
if len(similar_cars) > 0:
    median_price = similar_cars['Price'].median()
    # Get the bin edges from the original price segmentation
    bins = pd.qcut(data['Price'], q=5, retbins=True)[1]  # Get the bin edges
    segment_mapping = {0: 'Budget', 1: 'Economy', 2: 'Mid-range', 3: 'Premium', 4: 'Luxury'}
    segment_index = pd.cut([median_price], bins=bins, labels=False)[0]
    example_car['Price_Segment'] = segment_mapping[segment_index]
else:
    # Default to Mid-range if no similar cars found
    example_car['Price_Segment'] = 'Mid-range'

# First encode categorical features
for feature in categorical_features:
    if feature in example_car.columns:
        if feature in label_encoders:
            example_car[feature] = example_car[feature].apply(
                lambda x: label_encoders[feature].transform([x])[0] if x in label_encoders[feature].classes_ else -1
            )

# Then add Year_Scaled using the same scaler
example_car['Year_Scaled'] = scaler.transform(example_car[['Year']]) * 0.5  # Apply same scaling factor

# Predict price
predicted_price = model.predict(example_car[features])[0]
print(f"\nExample Prediction:")
print(f"Predicted Price for 2020 Ford Focus: ${predicted_price:,.2f}")

# Plot feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.title("Feature Importance")
plt.ylabel("Importance")
plt.xlabel("Feature")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()
