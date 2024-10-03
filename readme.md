# Insurance Purchase Prediction Model

This project implements a neural network using TensorFlow and Keras to predict whether a person will purchase insurance based on their age and affordability. The model has been improved with feature scaling, additional hidden layers, early stopping, and performance evaluation to enhance its accuracy and avoid overfitting.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Improvements](#model-improvements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Model Visualization](#model-visualization)
- [Contributing](#contributing)
- [License](#license)

## Overview
The project solves a binary classification problem: predicting whether a person will purchase insurance. The dataset includes features such as age and affordability. The target variable (`bought_insurance`) is binary, indicating whether a person bought insurance or not.

### Key Features:
- Feature scaling applied to numerical data.
- Neural network with two hidden layers for better learning.
- Early stopping to avoid overfitting.
- Model evaluation and performance visualization.

## Dataset
The dataset used in this project is `insurance_data.csv`, which contains the following columns:
- `age`: The age of the person.
- `affordibility`: A measure of the person's financial capability to afford insurance.
- `bought_insurance`: A binary variable (0 or 1) indicating whether the person purchased insurance.

### Sample Data:

| age | affordibility | bought_insurance |
|-----|---------------|------------------|
| 25  | 0.35          | 1                |
| 45  | 0.80          | 0                |
| 32  | 0.60          | 1                |

## Model Improvements
The model is significantly improved from the initial version with the following changes:
- **Feature Scaling**: Applied `StandardScaler` to normalize the age and affordability features.
- **Hidden Layers**: Added two hidden layers with `ReLU` activation functions to capture complex patterns in the data.
- **Early Stopping**: Early stopping is used to halt training once the model stops improving on validation data, preventing overfitting.
- **Model Evaluation**: The model is evaluated on test data after training to check its generalization ability.

## Installation
To get started with this project, you will need to have the following installed:
- Python 3.x
- TensorFlow
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

Install the required libraries using:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/insurance-prediction.git
   cd insurance-prediction
   ```

2. Add your `insurance_data.csv` dataset in the root directory of the project.

3. Run the model training script:

```bash
python model_training.py
```

### Code Walkthrough
```python
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf 
from tensorflow import keras 
import pandas as pd 
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("insurance_data.csv")

# Feature scaling for better convergence
scaler = StandardScaler()
df['age_scaled'] = scaler.fit_transform(df[['age']])
df['affordibility_scaled'] = scaler.fit_transform(df[['affordibility']])

# Split the dataset
X = df[['age_scaled', 'affordibility_scaled']]
y = df['bought_insurance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Define the improved model architecture
model = keras.Sequential([
    keras.layers.Dense(16, input_shape=(2,), activation='relu'),   # First hidden layer
    keras.layers.Dense(8, activation='relu'),                      # Second hidden layer
    keras.layers.Dense(1, activation='sigmoid')                    # Output layer
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Early stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train, y_train, 
    epochs=1000, 
    validation_split=0.2, 
    callbacks=[early_stopping]
)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
```

## Results
After training, the model achieved the following:
- **Test accuracy**: The accuracy on the test set will be displayed after training. 

By using early stopping, the model avoids overfitting and ensures better generalization to unseen data.

## Model Visualization
### Loss over Epochs:
A plot of training and validation loss over time is generated to visualize the learning process.

```python
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

### Accuracy over Epochs:
Similarly, a plot of training and validation accuracy helps analyze performance improvements.

```python
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

## Contributing
Contributions are welcome! If you have any improvements or suggestions, please feel free to submit a pull request or open an issue.