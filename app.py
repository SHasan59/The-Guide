import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import openai 
import os
from sklearn.preprocessing import StandardScaler
import re

openai.api_key = "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"

# Page Configuration
st.set_page_config(
    page_title="Car Price Prediction & Assistant",
    page_icon="🚗",
    layout="wide"
)

# Cache functions
@st.cache_data
def load_datasets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    original_data_path = os.path.join(current_dir, "Final_Cleaned_Dataset.csv")
    processed_data_path = os.path.join(current_dir, "Processed_Dataset_Encoded.csv")

    original_data = pd.read_csv(original_data_path, low_memory=False)
    processed_data = pd.read_csv(processed_data_path)

    return original_data, processed_data 



@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "car_price_model.pkl")
    encoders_path = os.path.join(current_dir, "car_label_encoders.pkl")

    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)

    return model, encoders


# Initialize data and models
try:
    original_data, data = load_datasets()
    model, label_encoders = load_model()
except Exception as e:
    st.error(f"Error loading data or models: {str(e)}")
    st.stop()


# Create main layout
col1, col2 = st.columns([2, 1])

# Define prediction interface
def create_prediction_interface():
    st.title("Car Price Prediction Tool 🚗")
    st.write("This app predicts car prices and visualizes price trends over the years.")

    # Sidebar inputs
    st.sidebar.header("Car Details")

    # Year slider
    year = st.sidebar.slider("Year", min_value=1980, max_value=2035, value=2022)

    # Make selection
    make_options = sorted(original_data['Make'].dropna().unique())
    make = st.sidebar.selectbox("Make", options=make_options)

    # Model selection (filtered by make)
    filtered_models = sorted(original_data[original_data['Make'] == make]['Model'].dropna().unique())
    model_name = st.sidebar.selectbox("Model", options=filtered_models)

    # Other inputs
    condition = st.sidebar.selectbox("Condition", sorted(original_data['Condition'].fillna('Unknown').unique()))
    cylinders = st.sidebar.selectbox("Cylinders", sorted(original_data['Cylinders'].fillna('Unknown').unique()))
    fuel = st.sidebar.selectbox("Fuel Type", sorted(original_data['Fuel'].fillna('Unknown').unique()))
    odometer = st.sidebar.number_input("Odometer (miles)", min_value=0, value=20000)
    title_status = st.sidebar.selectbox("Title Status", sorted(original_data['Title_status'].fillna('Unknown').unique()))
    transmission = st.sidebar.selectbox("Transmission", sorted(original_data['Transmission'].fillna('Unknown').unique()))
    drive = st.sidebar.selectbox("Drive Type", sorted(original_data['Drive'].fillna('Unknown').unique()))
    car_type = st.sidebar.selectbox("Type", sorted(original_data['Type'].fillna('Unknown').unique()))
    paint_color = st.sidebar.selectbox("Paint Color", sorted(original_data['Paint_color'].fillna('Unknown').unique()))
    mpg = st.sidebar.number_input("Miles per Gallon (MPG)", min_value=0, value=30)

    return year, make, model_name, condition, cylinders, fuel, odometer, title_status, transmission, drive, car_type, paint_color, mpg

def prepare_input(year, make, model_name, condition, cylinders, fuel, odometer, title_status,
                  transmission, drive, car_type, paint_color, mpg):
    input_data = pd.DataFrame({
        'Year': [year], 'Make': [make], 'Model': [model_name], 'Condition': [condition],
        'Cylinders': [cylinders], 'Fuel': [fuel], 'Odometer': [odometer],
        'Title_status': [title_status], 'Transmission': [transmission], 'Drive': [drive],
        'Type': [car_type], 'Paint_color': [paint_color]
    })

    encoded_data = input_data.copy()

    # Add Age feature
    current_year = 2024
    encoded_data['Age'] = current_year - encoded_data['Year']

    # Scale Year for Year_Scaled feature
    scaler = StandardScaler()
    encoded_data['Year_Scaled'] = scaler.fit_transform(encoded_data[['Year']]) * 0.5

    # Add Price_Segment based on similar cars
    similar_cars = data[
        (data['Make'] == make) & 
        (abs(data['Year'] - year) <= 2)
    ]
    if len(similar_cars) > 0:
        median_price = similar_cars['Price'].median()
        segment_bins = pd.qcut(data['Price'], q=5, labels=False)
        segment_mapping = {0: 'Budget', 1: 'Economy', 2: 'Mid-range', 3: 'Premium', 4: 'Luxury'}
        encoded_data['Price_Segment'] = segment_mapping[pd.cut([median_price], 
                                                    bins=pd.qcut(data['Price'], q=5).categories,
                                                    labels=False)[0]]
    else:
        encoded_data['Price_Segment'] = 'Mid-range'

    # Handle Cylinders
    if 'Cylinders' in encoded_data.columns:
        encoded_data['Cylinders'] = encoded_data['Cylinders'].str.extract(r'(\d+)').astype(float).fillna(-1)

    # Encode categorical features
    for feature in label_encoders:
        if feature in encoded_data.columns:
            try:
                encoded_data[feature] = encoded_data[feature].astype(str)
                encoded_data[feature] = label_encoders[feature].transform(encoded_data[feature])
            except:
                encoded_data[feature] = -1

    # Ensure all required columns are present
    required_columns = ['Year_Scaled', 'Make', 'Model', 'Condition', 'Cylinders', 'Fuel', 
                       'Odometer', 'Title_status', 'Transmission', 'Drive', 'Type', 
                       'Paint_color', 'Age', 'Price_Segment']
    
    for col in required_columns:
        if col not in encoded_data.columns:
            encoded_data[col] = 0

    return encoded_data[required_columns]

def create_assistant_section():
    st.header("🤖 Car Shopping Assistant")
    st.write("Ask me anything about cars! For example: 'What's a good car under $30,000 with low mileage?'")

    # Initialize session state for assistant responses
    if "assistant_responses" not in st.session_state:
        st.session_state.assistant_responses = []  # Store assistant's responses

    # Chat input
    prompt = st.chat_input("Ask about car recommendations...")
    if prompt:  # If the user submits a query
        # Process query to provide recommendations
        try:
            response = handle_car_recommendation(prompt)  # Use your model for recommendations
            st.session_state.assistant_responses.append(response)  # Store response
        except Exception as e:
            response = f"Sorry, I encountered an error: {str(e)}"

        # Display the latest response
        with st.chat_message("assistant"):
            st.markdown(response)

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.assistant_responses = []
        st.experimental_rerun()


def handle_car_recommendation(prompt):
    """
    Processes the user's query and provides car recommendations with proper type handling.
    """
    prompt = prompt.lower()
    
    # Budget parsing (keeping the same logic)
    budget = None
    numbers = re.findall(r'\$?(\d+(?:,\d+)?(?:\.\d+)?)[k]?\s?(?:thousand)?', prompt)
    
    for num in numbers:
        try:
            num = num.replace(',', '')
            if 'k' in num:
                num = float(num.replace('k', '')) * 1000
            budget = float(num)
            if budget < 1000:
                budget *= 1000
            break
        except ValueError:
            continue

    if budget is None:
        return "I can help you find a car! Could you please specify your budget? For example, you can say 'under $30,000' or '$25k'."

    # Query the model for car price predictions
    filtered_data = original_data.copy()
    
    # Ensure Make column is string type and handle NaN values
    filtered_data['Make'] = filtered_data['Make'].fillna('Unknown').astype(str)
    
    # Get all unique makes from the dataset
    all_makes = filtered_data['Make'].str.lower().unique()
    
    # Brand filtering - check if any car brand is mentioned in the prompt
    mentioned_brands = [make for make in all_makes if make in prompt and make != 'unknown']
    if mentioned_brands:
        filtered_data = filtered_data[filtered_data['Make'].str.lower().isin(mentioned_brands)]
    
    # Remove any rows with unknown or invalid makes
    filtered_data = filtered_data[filtered_data['Make'] != 'Unknown']
    
    # Additional filters
    if 'low mileage' in prompt:
        filtered_data = filtered_data[filtered_data['Odometer'] < 50000]
    if 'electric' in prompt:
        filtered_data = filtered_data[filtered_data['Fuel'] == 'electric']
    if 'suv' in prompt:
        filtered_data = filtered_data[filtered_data['Type'].str.contains('SUV', na=False)]
    
    # Sort by year and mileage
    filtered_data = filtered_data.sort_values(['Year', 'Odometer'], ascending=[False, True])
    
    recommendations = []
    seen_models = set()
    market_adjustment = 1.4

    for _, row in filtered_data.iterrows():
        model_key = f"{row['Make']} {row['Model']}"
        if model_key in seen_models:
            continue
            
        input_data = prepare_input(
            row['Year'], row['Make'], row['Model'], row['Condition'],
            'Not Specified', row['Fuel'], row['Odometer'], row['Title_status'],
            row['Transmission'], row['Drive'], row['Type'], row['Paint_color'], 
            row.get('MPG', 30)
        )

        base_price = model.predict(input_data)[0] + 8000
        
        # Price adjustments
        year_factor = (row['Year'] - 1980) / 40
        mileage_factor = max(0, 1 - (row['Odometer'] / 200000))
        
        predicted_price = base_price * market_adjustment
        predicted_price *= (1 + year_factor * 0.5)
        predicted_price *= (1 + mileage_factor * 0.3)

        if predicted_price <= budget and predicted_price >= 5000:
            drive_text = f"- Drive: {row['Drive']}\n" if pd.notna(row['Drive']) else ""
            transmission_text = f"- Transmission: {row['Transmission']}\n" if pd.notna(row['Transmission']) else ""
            
            recommendations.append(
                f"**{int(row['Year'])} {row['Make']} {row['Model']}**\n" +
                f"- Estimated price: ${predicted_price:,.2f}\n" +
                f"- Mileage: {row['Odometer']:,.0f} miles\n" +
                drive_text +
                transmission_text +
                f"- Fuel: {row['Fuel']}"
            )
            seen_models.add(model_key)

        if len(recommendations) >= 3:
            break

    if recommendations:
        response = f"Based on your budget of ${budget:,.0f}, here are some recommendations:\n\n"
        response += "\n\n".join(recommendations)
        response += "\n\nWould you like more specific details about any of these cars?"
    else:
        response = f"I couldn't find any cars within your ${budget:,.0f} budget. Would you like to try a higher budget?"

    return response

def prepare_input(year, make, model_name, condition, cylinders, fuel, odometer, title_status,
                  transmission, drive, car_type, paint_color, mpg):
    input_data = pd.DataFrame({
        'Year': [year], 'Make': [make], 'Model': [model_name], 'Condition': [condition],
        'Fuel': [fuel], 'Odometer': [odometer],
        'Title_status': [title_status], 'Transmission': [transmission], 'Drive': [drive],
        'Type': [car_type], 'Paint_color': [paint_color]
    })
    
    # Remove Cylinders from input data since it's causing issues
    encoded_data = input_data.copy()

    # Rest of the preparation logic remains the same...
    current_year = 2024
    encoded_data['Age'] = current_year - encoded_data['Year']
    
    scaler = StandardScaler()
    encoded_data['Year_Scaled'] = scaler.fit_transform(encoded_data[['Year']]) * 0.5

    similar_cars = data[
        (data['Make'] == make) & 
        (abs(data['Year'] - year) <= 2)
    ]
    if len(similar_cars) > 0:
        median_price = similar_cars['Price'].median()
        segment_bins = pd.qcut(data['Price'], q=5, labels=False)
        segment_mapping = {0: 'Budget', 1: 'Economy', 2: 'Mid-range', 3: 'Premium', 4: 'Luxury'}
        encoded_data['Price_Segment'] = segment_mapping[pd.cut([median_price], 
                                                    bins=pd.qcut(data['Price'], q=5).categories,
                                                    labels=False)[0]]
    else:
        encoded_data['Price_Segment'] = 'Mid-range'

    # Encode categorical features
    for feature in label_encoders:
        if feature in encoded_data.columns:
            try:
                encoded_data[feature] = encoded_data[feature].astype(str)
                encoded_data[feature] = label_encoders[feature].transform(encoded_data[feature])
            except:
                encoded_data[feature] = -1

    # Add Cylinders column with a default value
    encoded_data['Cylinders'] = -1

    # Ensure all required columns are present
    required_columns = ['Year_Scaled', 'Make', 'Model', 'Condition', 'Cylinders', 'Fuel', 
                       'Odometer', 'Title_status', 'Transmission', 'Drive', 'Type', 
                       'Paint_color', 'Age', 'Price_Segment']
    
    for col in required_columns:
        if col not in encoded_data.columns:
            encoded_data[col] = 0

    return encoded_data[required_columns]


# Main interface
with col1:
    # Get input values
    inputs = create_prediction_interface()

    # Predict price when button is clicked
    if st.sidebar.button("Predict Price"):
        input_data = prepare_input(*inputs)
        predicted_price = model.predict(input_data)[0]
        st.sidebar.success(f"Predicted Price: ${predicted_price:,.2f}")

        # Historical predictions only (up to 2024)
        selected_year = inputs[0]
        historical_years = list(range(1980, 2025))
        historical_prices = []

        for year in historical_years:
            year_inputs = list(inputs)
            year_inputs[0] = year  # Update year
            year_input_data = prepare_input(*year_inputs)
            historical_prices.append(model.predict(year_input_data)[0])

        # Plot historical prices
        st.header("Historical Price Predictions")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(historical_years, historical_prices, marker="o", linestyle="-", color="blue", alpha=0.6)

        # Highlight selected year
        if selected_year in historical_years:
            ax.axvline(selected_year, color="red", linestyle="--", label="Selected Year")

        # Format the chart
        ax.set_title(f"Price Trend of {inputs[1]} {inputs[2]} (1980–2024)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Price ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        st.pyplot(fig)

        # Warning for invalid years
        if selected_year > 2024:
            st.warning("The model only supports predictions up to the year 2024.")

    # Historical trends
    st.header("Price Trends Over the Years")
    price_trend = data.groupby("Year")["Price"].mean().reset_index()
    st.line_chart(price_trend, x="Year", y="Price", use_container_width=True)

# Assistant interface
with col2:
    create_assistant_section()
