# Cat-Dog Image Classification using SVM

This project implements a Support Vector Machine (SVM) classifier to classify images of cats and dogs from the Kaggle Cats vs Dogs dataset.

## Features

- **Multiple SVM Kernels**: Linear, RBF, and Polynomial kernels for comparison
- **Image Preprocessing**: Resize, normalize, and flatten images for SVM input
- **Comprehensive Evaluation**: Accuracy, confusion matrix, and classification report
- **Visual Results**: Sample predictions with images
- **Model Persistence**: Save and load trained models
- **Modular Design**: Clean, well-commented, and reusable code

## Requirements

- Python 3.7+
- Required packages (see `requirements.txt`):
  - numpy
  - pandas
  - opencv-python
  - matplotlib
  - scikit-learn
  - joblib

## Installation

### Quick Start (Recommended)
Run the complete setup script that handles everything automatically:
```bash
python setup_kaggle_project.py
```

### Manual Installation
1. Clone or download this project
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Setup

### Option 1: Automatic Kaggle Dataset Download (Recommended)
The project includes an automatic downloader for the Kaggle Cats vs Dogs dataset:

```bash
python download_kaggle_dataset.py
```

This will:
- Install Kaggle CLI if needed
- Guide you through setting up Kaggle credentials
- Download the official dataset
- Organize it into the required folder structure

**Note:** You'll need a Kaggle account and API token. The script will guide you through this process.

### Option 2: Manual Kaggle Dataset Setup
1. Download the [Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats) dataset from Kaggle
2. Extract the dataset and organize it in the following structure:
   ```
   dataset/
   ├── train/
   │   ├── cats/
   │   │   ├── cat.0.jpg
   │   │   ├── cat.1.jpg
   │   │   └── ...
   │   └── dogs/
   │       ├── dog.0.jpg
   │       ├── dog.1.jpg
   │       └── ...
   └── test/
       ├── cats/
       └── dogs/
   ```

### Option 3: Create Sample Dataset Structure
If you don't have the full dataset, create the folder structure above and place your cat and dog images in the respective folders.

## Usage

### Basic Usage
Run the main script:
```bash
python svm_cat_dog_classifier.py
```

### Custom Configuration
You can modify the classifier parameters in the `main()` function:

```python
# Initialize classifier with custom parameters
classifier = CatDogClassifier(
    img_size=64,      # Image size (default: 64x64)
    grayscale=True    # Use grayscale images (default: True)
)
```

### Programmatic Usage
```python
from svm_cat_dog_classifier import CatDogClassifier

# Initialize classifier
classifier = CatDogClassifier(img_size=64, grayscale=True)

# Load and preprocess data
X, y = classifier.load_and_preprocess_data("dataset")

# Train model with specific kernel
classifier.train_model(X, y, kernel='linear')

# Evaluate model
classifier.evaluate_model(kernel='linear')

# Show sample predictions
classifier.show_sample_predictions(kernel='linear', num_samples=5)

# Save model
classifier.save_model("my_model.pkl", kernel='linear')
```

## Project Structure

```
SCT_ML_3/
├── svm_cat_dog_classifier.py  # Main classifier script
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── dataset/                  # Dataset folder (create this)
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   └── test/
│       ├── cats/
│       └── dogs/
└── *.pkl                     # Saved models (generated after training)
```

## Class Documentation

### CatDogClassifier

The main classifier class with the following methods:

#### `__init__(img_size=64, grayscale=True)`
Initialize the classifier with image size and color mode settings.

#### `load_and_preprocess_data(dataset_path)`
Load images from the dataset, preprocess them, and return feature matrix and labels.

#### `train_model(X, y, kernel='linear')`
Train an SVM model with the specified kernel ('linear', 'rbf', 'poly').

#### `evaluate_model(kernel='linear')`
Print evaluation metrics for the trained model.

#### `compare_kernels()`
Compare performance of different kernels.

#### `show_sample_predictions(kernel='linear', num_samples=5)`
Display sample predictions with images.

#### `save_model(filepath, kernel='linear')`
Save the trained model to a file.

#### `load_model(filepath)`
Load a previously saved model.

## Output

The script will:
1. Load and preprocess images from the dataset
2. Train SVM models with different kernels
3. Print evaluation metrics for each kernel
4. Compare kernel performances
5. Show sample predictions with images
6. Save the best performing model

### Sample Output
```
Cat-Dog Image Classification using SVM
==================================================
Loading and preprocessing data...
Loading cats images...
Loading dogs images...
Loaded 25000 images
Class distribution: Cats=12500, Dogs=12500

Training SVM model with linear kernel...
Training completed!
Accuracy: 0.8234

==================================================
EVALUATION RESULTS - LINEAR KERNEL
==================================================

Accuracy Score: 0.8234

Confusion Matrix:
[[1234  266]
 [ 189 1311]]

Classification Report:
              precision    recall  f1-score   support

         Cat       0.87      0.82      0.84      1500
         Dog       0.83      0.87      0.85      1500

    accuracy                           0.84      3000
   macro avg       0.85      0.85      0.85      3000
weighted avg       0.85      0.84      0.85      3000
```

## Performance Notes

- **Image Size**: Smaller images (64x64) train faster but may lose detail
- **Grayscale vs Color**: Grayscale reduces feature dimensionality and training time
- **Kernel Selection**: 
  - Linear: Fastest, good for linearly separable data
  - RBF: Good for non-linear data, moderate speed
  - Polynomial: Can capture complex patterns, slower training

## Troubleshooting

### Common Issues

1. **Dataset not found**: Ensure the dataset folder structure is correct
2. **Memory errors**: Reduce image size or use grayscale images
3. **Slow training**: Use linear kernel or reduce dataset size
4. **Import errors**: Install all required packages from requirements.txt

### Performance Optimization

- Use grayscale images for faster processing
- Reduce image size for memory efficiency
- Consider using a subset of the dataset for testing
- Use linear kernel for faster training

## License

This project is for educational purposes. Please respect the original dataset licenses.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project. 