
# Soil Analysis and Irrigation Prediction

## Project Overview

This project is created under guidance of **Dr. S. Dutta, Assistant Professor,Dept. of Computer Science and Engineering, NIT Jamshedpur** with the course code 1507 and aims to predict whether irrigation is needed based on soil moisture, temperature, and humidity data. By using machine learning techniques and feature engineering, the model provides insights into optimal irrigation management, improving agricultural practices.


## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Work](#future-work)

## Installation

To run this project, you need to install the required Python packages. You can install them using the following command:

```bash
pip install -r requirements.txt
```

The key dependencies include:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can also manually install the libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. **Data Loading**: The project starts by loading a dataset (`modified_irrigation_dataset.csv`) which contains the following features:
   - `Moisture`: Soil moisture content
   - `Temperature`: Ambient temperature
   - `Humidity`: Air humidity level
   - `Irrigation_Needed`: The target label, indicating whether irrigation is required

2. **Running the Notebook**: The code is implemented in a Jupyter Notebook (`soil_analysis.ipynb`). You can open the notebook and run it cell by cell to reproduce the results.

3. **Model Training**: The model uses several machine learning algorithms such as Logistic Regression, Random Forest, and Support Vector Machine (SVM) to predict whether irrigation is needed.

4. **Feature Engineering**: Non-linear interaction terms and outlier handling are incorporated into the dataset for better model performance.

## Data

The dataset used in this project includes features related to soil and environmental conditions:
- **Moisture**: A key predictor of irrigation needs.
- **Temperature**: Affects evaporation rates and soil moisture levels.
- **Humidity**: Related to water retention in the atmosphere and soil.

Additional engineered features include:
- **Moisture_Temp_Interaction**: An interaction term between moisture and temperature.
- **Humidity_Squared**: A non-linear transformation of the humidity feature.

The dataset also includes some label noise and outliers for model robustness testing.

## Model Training

### Data Preprocessing
- **Outlier Handling**: Introduces synthetic outliers in 5% of the dataset to test model resilience.
- **Non-linear Terms**: New interaction features (e.g., `Moisture_Temp_Interaction`) were created to capture complex relationships in the data.
- **Label Noise**: Random noise was added to 5% of the labels to challenge the model's ability to generalize.

### Standardization
The features are standardized using `StandardScaler` to ensure the models train efficiently.

### Model Training
- **Logistic Regression**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**

The data is split into 80% training and 20% testing, and the models are evaluated on this split.

## Evaluation

After training the models, the following metrics are used to evaluate the performance:
- **Accuracy**
- **Confusion Matrix**
- **ROC Curve and AUC Score**
- **Precision-Recall Curve**
- **Classification Report** (including precision, recall, F1-score)

## Results

The project compares the performance of Logistic Regression, Random Forest, and SVM models. Detailed evaluation metrics are provided, including confusion matrices and ROC curves. Based on these results, a recommendation is made regarding the best-performing model for irrigation prediction.

## Future Work

- **Hyperparameter Tuning**: Optimizing the hyperparameters of the models using Grid Search or Random Search.
- **Incorporating More Features**: Adding additional features like soil type, crop type, and weather forecasts to improve the accuracy of irrigation predictions.
- **Time-Series Analysis**: Since irrigation needs are often influenced by time, incorporating time-series data could enhance prediction accuracy.
