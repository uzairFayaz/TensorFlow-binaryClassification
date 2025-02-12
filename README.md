# Binary Classification using TensorFlow

## Overview
This project demonstrates how to perform binary classification using TensorFlow and Keras. The example uses the IMDB movie review dataset to classify sentiment (positive or negative) using a Convolutional Neural Network (CNN).

## Prerequisites
Ensure you have the following installed:
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

You can install the required packages using:
```sh
pip install tensorflow numpy matplotlib
```

## Dataset
The IMDB dataset consists of 50,000 movie reviews labeled as positive or negative. We use only the top 10,000 frequent words and pad sequences to a fixed length of 200.

## Model Architecture
- **Embedding Layer**: Converts words into dense vectors.
- **Conv1D Layer**: Extracts features from text sequences.
- **GlobalMaxPooling1D Layer**: Reduces dimensionality.
- **Dense Layers**: Fully connected layers for classification.
- **Sigmoid Activation**: Outputs probabilities for binary classification.

## Usage

### 1. Clone the Repository
```sh
git clone https://github.com/your-repo/binary-classification-tensorflow.git
cd binary-classification-tensorflow
```

### 2. Run the Script
```sh
python binary_classification.py
```

## Code Implementation
### 1. Import Libraries
```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
```

### 2. Load and Preprocess Data
```python
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

num_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
maxlen = 200
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
```

### 3. Build the Model
```python
model = keras.Sequential([
    keras.layers.Embedding(input_dim=num_words, output_dim=32, input_length=maxlen),
    keras.layers.Conv1D(32, 3, activation="relu"),
    keras.layers.GlobalMaxPooling1D(),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
```

### 4. Train the Model
```python
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))
```

### 5. Evaluate the Model
```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

### 6. Make Predictions
```python
predictions = model.predict(x_test[:5])
print((predictions > 0.5).astype(int))
```

## Results
After training for 5 epochs, the model achieves an accuracy of around **85%** on the test set.

## License
This project is open-source and available under the [MIT License](LICENSE).

