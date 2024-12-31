from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestRegressor
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' before importing pyplot

import matplotlib.pyplot as plt
from datetime import datetime
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
dataset_path = "/Users/shruthi2003/Desktop/Crop_Lai_Pred/Crop Production data.csv"  # Change this to your dataset file path
dataset = pd.read_csv(dataset_path)
# Load models
classification_model_path = '/Users/shruthi2003/Desktop/Crop_Lai_Pred/static/model/Estimation of LAI.hdf5'
lai_model_path = '/Users/shruthi2003/Desktop/Crop_Lai_Pred/static/model/lai_regression_model.pkl'  # Path to the ML model (not deep learning)
classification_model = load_model(classification_model_path)

# Load your trained machine learning model for LAI prediction
lai_model = joblib.load(lai_model_path)  # Load a pre-trained RandomForestRegressor, XGBoost, etc.

class_labels = ['Jute', 'Maize', 'Rice', 'Sugarcane', 'Wheat']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    app.logger.info("Predict endpoint hit.")
    
    if 'image' not in request.files:
        app.logger.error("No image file provided.")
        return jsonify({'error': 'No image file provided'}), 400

    state = "Andhra Pradesh"
    district = "KURNOOL"
    crop = "Rice"

    image = request.files['image']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Ensure upload folder exists
    image.save(image_path)
    app.logger.info(f"Image saved to {image_path}.")

    try:
        # Predict crop
        predicted_crop = predict_image(image_path, classification_model)
        app.logger.info(f"Predicted Crop: {predicted_crop}")

        # Predict LAI
        lai_prediction = predict_lai_with_image(image_path)
        lai_prediction = float(lai_prediction)
        app.logger.info(f"Predicted LAI: {lai_prediction}")

        # Reference yield
        reference_yield = get_average_yield(dataset, state, district, predicted_crop)
        app.logger.info(f"Reference Yield: {reference_yield}")

        # Predict yield
        predicted_yield = predict_yield(image_path, reference_yield)

        # Plot graph
        graph_path = plot_yield_trends(dataset, state, district, crop)
        graph_url = request.url_root + graph_path.replace("static/", "static/")
        app.logger.info(f"Graph saved at {graph_path}")
    except Exception as e:
        app.logger.error(f"Error: {e}")
        return jsonify({'error': str(e)}), 400

    return jsonify({
        'predicted_crop': predicted_crop,
        'lai_prediction': lai_prediction,
        'reference_yield': reference_yield,
        'predicted_yield': predicted_yield,
        'graph': graph_url
    })

def get_average_yield(dataset, state, district, crop):
    # Filter the data based on the given state, district, and crop
    filtered_data = dataset[
        (dataset["State_Name"] == state) &
        (dataset["District_Name"] == district) &
        (dataset["Crop"] == crop)
    ]
    
    # Check if there is any data matching the given criteria
    if filtered_data.empty:
        raise ValueError(f"No data found for {crop} in {state}, {district} in the dataset.")
    
    # Get the latest year of data available for the crop
    latest_year = filtered_data["Crop_Year"].max()
    latest_year_data = filtered_data[filtered_data["Crop_Year"] == latest_year]
    
    # Check if there is any data for the latest year
    if latest_year_data.empty:
        raise ValueError(f"No data found for {crop} in {state}, {district} in the latest available year ({latest_year}).")

    # Ensure 'Production' and 'Area' columns are numeric to avoid errors
    latest_year_data["Production"] = pd.to_numeric(latest_year_data["Production"], errors='coerce')
    latest_year_data = latest_year_data.copy()  # Make a copy of the subset
    latest_year_data["Area"] = pd.to_numeric(latest_year_data["Area"], errors='coerce')


    # Handle cases where 'Production' or 'Area' might be NaN
    latest_year_data = latest_year_data.dropna(subset=["Production", "Area"])

    # Calculate Yield: Yield = Production / Area
    latest_year_data["Yield"] = latest_year_data["Production"] / (latest_year_data["Area"] + 1e-6)

    # Calculate the average yield
    average_yield = latest_year_data["Yield"].mean()

    # Return the calculated average yield
    return average_yield
# Generate synthetic data
def generate_synthetic_data():
    np.random.seed(42)
    data = {
        "NDVI": np.random.uniform(0.2, 0.9, 100),
        "EVI": np.random.uniform(0.1, 0.8, 100),
        "GNDVI": np.random.uniform(0.2, 0.9, 100),
        "RENDVI": np.random.uniform(0.2, 0.7, 100),
        "WDRVI": np.random.uniform(0.1, 0.6, 100),
        "CI": np.random.uniform(0.3, 1.0, 100),
        "NDRE": np.random.uniform(0.1, 0.7, 100),
        "LAI": np.random.uniform(0.5, 6.0, 100)
    }
    df = pd.DataFrame(data)
    return df

# Evaluate models and train best one
def evaluate_models(df):
    X = df[["NDVI", "EVI", "GNDVI", "RENDVI", "WDRVI", "CI", "NDRE"]]
    y = df["LAI"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        results[name] = {"MSE": mse, "RMSE": rmse, "R^2": r2}
        print(f"{name}: RMSE = {rmse:.4f}")

    best_model_name = max(results, key=lambda x: results[x]["R^2"])
    best_model = models[best_model_name]

    return best_model

# Calculate LAI from vegetation indices
def calculate_lai_from_indices(indices_df, model):
    all_features = ['NDVI', 'EVI', 'GNDVI', 'RENDVI', 'WDRVI', 'CI', 'NDRE']
    indices_of_interest = ['NDVI', 'GNDVI', 'WDRVI']

    feature_data = {}
    for feature in indices_of_interest:
        feature_data[feature] = indices_df.loc[indices_df['VI'] == feature, 'Flowering'].values
    features_df = pd.DataFrame(feature_data)
    for missing_feature in set(all_features) - set(features_df.columns):
        features_df[missing_feature] = 0
    features_df = features_df[all_features]
    lai_predictions = model.predict(features_df)
    indices_df.loc[indices_df['VI'].isin(indices_of_interest), 'LAI'] = lai_predictions

    mean_lai = lai_predictions.mean()
    print(f"Calculated Mean LAI: {mean_lai:.4f}")

    return mean_lai

# Estimate yield range based on LAI and reference yield
def estimate_yield_range(lai, reference_yield, predicted_lai):
    scaling_factor = predicted_lai / lai.mean()
    estimated_yield = reference_yield * scaling_factor
    print(f"Estimated Yield: {estimated_yield:.4f}")
    lower_bound = estimated_yield * 0.9
    upper_bound = estimated_yield * 1.1

    print(f"Predicted LAI: {predicted_lai:.4f}")
    print(f"Estimated Yield Range: {lower_bound:.4f} - {upper_bound:.4f}")
    return lower_bound, upper_bound

# Main function to integrate in your Flask code
def predict_yield(image_path, reference_yield):
    # Step 1: Generate synthetic data
    synthetic_data = generate_synthetic_data()

    # Step 2: Train models and select the best model
    best_model = evaluate_models(synthetic_data)

    # Step 3: Read indices table (example, you can adjust as needed)
    indices_table_path="/Users/shruthi2003/Desktop/Crop_Lai_Pred/static/data/Vegetation_Indices_Rice.xlsx"
    indices_table = pd.read_excel(indices_table_path, engine='openpyxl')


    # Step 4: Calculate LAI from indices using the best model
    predicted_lai = calculate_lai_from_indices(indices_table, best_model)

    # Step 5: Estimate yield range
    yield_range = estimate_yield_range(synthetic_data["LAI"], reference_yield, predicted_lai)

    return (yield_range[0]+yield_range[1])/2


def predict_image(image_path, model):
    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return class_labels[predicted_class]

def calculate_vegetation_indices(image):
    image = image.astype(np.float32) / 255.0

    R = image[:, :, 2]
    G = image[:, :, 1]
    B = image[:, :, 0]

    NDVI = (R - G) / (R + G + 1e-6)
    EVI = 2.5 * (R - G) / (R + 6 * G - 7.5 * B + 1 + 1e-6)
    GNDVI = (G - R) / (G + R + 1e-6)
    RENDVI = (R - B) / (R + B + 1e-6)
    WDRVI = 0.1 * (R - G) / (R + G + 1e-6)
    CI = R / (G + 1e-6)
    NDRE = (R - B) / (R + B + 1e-6)

    indices = {
        "NDVI": np.mean(NDVI),
        "EVI": np.mean(EVI),
        "GNDVI": np.mean(GNDVI),
        "RENDVI": np.mean(RENDVI),
        "WDRVI": np.mean(WDRVI),
        "CI": np.mean(CI),
        "NDRE": np.mean(NDRE),
    }
    return indices

def predict_lai_with_image(image_path):
    # Read the image and extract vegetation indices
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found. Please check the path.")

    indices = calculate_vegetation_indices(img)
    print("Extracted Vegetation Indices:", indices)

    feature_vector = np.array([indices[key] for key in ["NDVI", "EVI", "GNDVI", "RENDVI", "WDRVI", "CI", "NDRE"]]).reshape(1, -1)

    # Predict LAI using the pre-trained machine learning model
    lai_prediction = lai_model.predict(feature_vector)[0]

    if lai_prediction < 0 or lai_prediction > 6:
        raise ValueError(
            "The uploaded image is unable to predict LAI. Please provide a proper image."
        )

    print(f"Predicted LAI: {lai_prediction:.4f}")
    return lai_prediction

def plot_yield_trends(dataset, state, district, crop):
    # Validate dataset columns
    required_columns = {"State_Name", "District_Name", "Crop", "Crop_Year", "Production", "Area"}
    if not required_columns.issubset(dataset.columns):
        raise ValueError(f"Dataset is missing required columns: {required_columns - set(dataset.columns)}")

    # Filter data
    filtered_data = dataset[
        (dataset["State_Name"] == state) &
        (dataset["District_Name"] == district) &
        (dataset["Crop"] == crop)
    ]

    if filtered_data.empty:
        raise ValueError("No data available for the specified state, district, and crop.")

    # Calculate average yield
    avg_yield = filtered_data.groupby("Crop_Year").apply(
    lambda x: (x["Production"] / x["Area"].replace(0, np.nan)).mean()
    ).sort_index()


    # Plot
    plt.figure(figsize=(4,4))
    plt.plot(avg_yield.index, avg_yield.values, marker='o', label="Average Yield")
    plt.title("Yield Trends Over the Years")
    plt.xlabel("Year")
    plt.ylabel("Average Yield")
    plt.legend()
    plt.grid()

    # Save the plot to a file
    os.makedirs("static", exist_ok=True)
    graph_filename = f"static/yield_trends_{uuid.uuid4().hex[:8]}.png"
    plt.savefig(graph_filename)
    plt.close()

    return graph_filename
    
if __name__ == '__main__':
    app.run(debug=True,port=5003)
