#!/usr/bin/env python3
"""
Flask Web Application for SVM Cat-Dog Classification
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import base64
import io
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = 64
GRAYSCALE = True

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for model and results
classifier = None
model_results = {}
sample_images = []
_initialized = False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for SVM prediction"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Convert to grayscale if specified
    if GRAYSCALE:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape(IMG_SIZE, IMG_SIZE, 1)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    # Flatten
    img_flat = img.flatten()
    
    return img_flat

def load_trained_models():
    """Load trained SVM models with improved parameters"""
    global classifier, model_results
    
    # Clear any existing models
    model_results.clear()
    
    try:
        print("Loading and training models...")
        
        # Initialize classifier
        from svm_cat_dog_classifier import CatDogClassifier  # May not exist after cleanup
        classifier = CatDogClassifier(img_size=IMG_SIZE, grayscale=GRAYSCALE)
        print("+ Classifier initialized")

        # Load dataset and train models
        X, y = classifier.load_and_preprocess_data("dataset")
        print(f"+ Data loaded: X shape: {X.shape}, y shape: {y.shape}")

        # Train all kernels with optimized parameters
        kernels = ['linear', 'rbf', 'poly']
        for kernel in kernels:
            print(f"Training {kernel} kernel...")
            classifier.train_model(X, y, kernel)

            # Verify the model was stored
            if kernel in classifier.kernel_results:
                model_results[kernel] = classifier.kernel_results[kernel]
                print(f"+ {kernel} kernel stored in model_results")
                print(f"  - Keys: {list(model_results[kernel].keys())}")
            else:
                print(f"X {kernel} kernel not found in classifier.kernel_results")

        print(f"+ Models loaded successfully! model_results keys: {list(model_results.keys())}")
        
    except Exception as e:
        print(f"X Error loading/training models with CatDogClassifier: {str(e)}")
        import traceback
        traceback.print_exc()

        # First, if dataset exists, train compact fallback models for all kernels
        dataset_base = os.path.join(os.getcwd(), 'dataset')
        if os.path.exists(os.path.join(dataset_base, 'train')) and os.path.exists(os.path.join(dataset_base, 'test')):
            print("Attempting fallback: training compact models from existing dataset...")
            try:
                def collect_xy(folder):
                    X, y = [], []
                    cat_dir = os.path.join(folder, 'cats')
                    dog_dir = os.path.join(folder, 'dogs')
                    for subdir, label in [(cat_dir, 0), (dog_dir, 1)]:
                        if os.path.exists(subdir):
                            for fname in os.listdir(subdir):
                                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                                    path = os.path.join(subdir, fname)
                                    arr = preprocess_image(path)
                                    if arr is not None:
                                        X.append(arr)
                                        y.append(label)
                    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

                X_train, y_train = collect_xy(os.path.join(dataset_base, 'train'))
                X_test, y_test = collect_xy(os.path.join(dataset_base, 'test'))
                print(f"+ Dataset prepared. Train: {X_train.shape}, Test: {X_test.shape}")

                kernels = ['linear', 'rbf', 'poly']
                model_results.clear()
                for k in kernels:
                    print(f"Training fallback {k} SVM (probability=True)...")
                    clf = make_pipeline(StandardScaler(with_mean=True), SVC(kernel=k, probability=True, gamma='scale'))
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
                    model_results[k] = {
                        'model': clf,
                        'accuracy': acc,
                        'precision': prec,
                        'recall': rec,
                        'f1_score': f1,
                        'y_test': y_test,
                        'y_pred': y_pred,
                    }
                    print(f"+ {k} trained. Acc={acc:.4f}")
                print("+ Fallback training complete. Kernels:", list(model_results.keys()))
                print("+ Model results structure:", {k: list(v.keys()) for k, v in model_results.items()})
                return  # Exit here if training succeeded
            except Exception as train_error:
                print(f"X Fallback training failed: {train_error}")
                import traceback
                traceback.print_exc()

        # If no dataset or training failed, try loading a pre-trained single model and wrap it
        try:
            pkl_path = os.path.join(os.getcwd(), 'enhanced_svm_model.pkl')
            if os.path.exists(pkl_path):
                print(f"Attempting fallback: loading pre-trained model from {pkl_path} ...")
                obj = joblib.load(pkl_path)

                class PretrainedModelWrapper:
                    def __init__(self, model, scaler=None):
                        self.model = model
                        self.scaler = scaler
                    def _transform(self, X):
                        arr = np.asarray(X, dtype=np.float32)
                        if self.scaler is not None:
                            arr = self.scaler.transform(arr)
                        return arr
                    def predict(self, X):
                        arr = self._transform(X)
                        return self.model.predict(arr)
                    def predict_proba(self, X):
                        arr = self._transform(X)
                        if hasattr(self.model, 'predict_proba'):
                            return self.model.predict_proba(arr)
                        if hasattr(self.model, 'decision_function'):
                            scores = self.model.decision_function(arr)
                            # Convert to probabilities with sigmoid/softmax
                            if scores.ndim == 1:
                                probs_pos = 1.0 / (1.0 + np.exp(-scores))
                                probs = np.vstack([1 - probs_pos, probs_pos]).T
                            else:
                                e = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                                probs = e / np.sum(e, axis=1, keepdims=True)
                            return probs
                        # Fallback uniform
                        return np.tile([0.5, 0.5], (np.asarray(X).shape[0], 1))

                if isinstance(obj, dict):
                    mdl = obj.get('model', None)
                    sclr = obj.get('scaler', None)
                    wrapped = PretrainedModelWrapper(mdl, sclr)
                else:
                    wrapped = obj  # already a model

                model_results.clear()
                # Assume RBF if unknown
                model_results['rbf'] = { 'model': wrapped }
                print("+ Fallback model loaded successfully. Available kernels:", list(model_results.keys()))
                print("+ Model results structure:", {k: list(v.keys()) for k, v in model_results.items()})
            else:
                print("X Fallback model not found at enhanced_svm_model.pkl")
                raise
        except Exception as fallback_error:
            print(f"X Failed to load fallback model: {fallback_error}")
            import traceback
            traceback.print_exc()
            raise

def get_sample_images():
    """Get sample images from dataset for gallery"""
    global sample_images
    sample_images = []
    
    # Get sample cat images
    cat_folder = "dataset/train/cats"
    if os.path.exists(cat_folder):
        cat_files = [f for f in os.listdir(cat_folder) if f.endswith(('.jpg', '.jpeg', '.png'))][:6]
        for cat_file in cat_files:
            sample_images.append({
                'path': os.path.join(cat_folder, cat_file),
                'label': 'Cat',
                'type': 'cat'
            })
    
    # Get sample dog images
    dog_folder = "dataset/train/dogs"
    if os.path.exists(dog_folder):
        dog_files = [f for f in os.listdir(dog_folder) if f.endswith(('.jpg', '.jpeg', '.png'))][:6]
        for dog_file in dog_files:
            sample_images.append({
                'path': os.path.join(dog_folder, dog_file),
                'label': 'Dog',
                'type': 'dog'
            })
    
    return sample_images



def create_confusion_matrix_plot():
    """Create confusion matrix visualization"""
    try:
        # Use RBF kernel results if available
        if 'rbf' not in model_results:
            print("RBF kernel not available for confusion matrix")
            return None
        if 'y_test' not in model_results['rbf'] or 'y_pred' not in model_results['rbf']:
            # Metrics not available (e.g., when using fallback pre-trained model only)
            print("RBF kernel missing y_test or y_pred for confusion matrix")
            return None

        y_true = model_results['rbf']['y_test']
        y_pred = model_results['rbf']['y_pred']
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Cat', 'Dog'], 
                    yticklabels=['Cat', 'Dog'])
        plt.title('Confusion Matrix (RBF Kernel)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)
        plt.close()
        
        # Convert to base64
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        print("Confusion matrix created successfully")
        return img_str
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_dataset_stats():
    """Get dataset statistics"""
    stats = {}
    
    folders = {
        'train_cats': 'dataset/train/cats',
        'train_dogs': 'dataset/train/dogs',
        'test_cats': 'dataset/test/cats',
        'test_dogs': 'dataset/test/dogs'
    }
    
    for key, folder in folders.items():
        if os.path.exists(folder):
            count = len([f for f in os.listdir(folder) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            stats[key] = count
        else:
            stats[key] = 0
    
    return stats

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/test')
def test():
    """Test endpoint"""
    return jsonify({'status': 'ok', 'message': 'Server is running'})

def ensure_models_loaded():
    """Ensure models are loaded before processing requests"""
    global _initialized
    if not _initialized:
        print("Models not initialized, loading now...")
        try:
            load_trained_models()
            get_sample_images()
            _initialized = True
            print("+ Models successfully initialized")
        except Exception as e:
            print(f"X Failed to initialize models: {e}")
            _initialized = False
            raise
    else:
        print("+ Models already initialized")

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'models_loaded': len(model_results) > 0,
        'models_available': list(model_results.keys()) if model_results else [],
        'model_results_keys': [list(model_results[k].keys()) if k in model_results else [] for k in ['linear', 'rbf', 'poly']],
        'upload_folder': UPLOAD_FOLDER,
        'upload_folder_exists': os.path.exists(UPLOAD_FOLDER)
    })

@app.route('/test_upload')
def test_upload_page():
    """Test upload page"""
    return app.send_static_file('test_upload_simple.html')

@app.route('/debug_upload')
def debug_upload_page():
    """Debug upload page"""
    return app.send_static_file('test_upload_debug.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    # Models should already be loaded at startup
    if not model_results or len(model_results) == 0:
        print("ERROR: Models not loaded! This should not happen.")
        return jsonify({'error': 'Models not loaded. Please restart the application.'})
    
    print(f"=== UPLOAD REQUEST START ===")
    print(f"Upload request received: {request.method}")
    print(f"Files in request: {list(request.files.keys())}")
    print(f"Request headers: {dict(request.headers)}")
    
    try:
        if 'file' not in request.files:
            print("No 'file' key in request.files")
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        print(f"File received: {file.filename}")
        print(f"File content type: {file.content_type}")
        print(f"File size: {len(file.read())} bytes")
        file.seek(0)  # Reset file pointer
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"upload_{timestamp}_{file.filename}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            print(f"Saving file to: {filepath}")
            
            # Ensure upload folder exists
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            
            file.save(filepath)
            
            # Verify file was saved
            if not os.path.exists(filepath):
                print(f"File was not saved to {filepath}")
                return jsonify({'error': 'Failed to save uploaded file'})
            
            print(f"File saved successfully. File size: {os.path.getsize(filepath)} bytes")
            
            # Preprocess image
            print("Preprocessing image...")
            img_processed = preprocess_image(filepath)
            if img_processed is None:
                print("Image preprocessing failed")
                return jsonify({'error': 'Failed to process image'})
            
            print(f"Image preprocessed successfully. Shape: {img_processed.shape}")
            
            # Check if models are loaded
            if not model_results:
                print("Models not loaded!")
                return jsonify({'error': 'Models not loaded. Please restart the application.'})
            
            print(f"Models loaded: {list(model_results.keys())}")
            
            # Make predictions with all available kernels
            predictions = {}
            for kernel in ['linear', 'rbf', 'poly']:
                if kernel in model_results:
                    try:
                        # Get the actual model from the dictionary
                        model = model_results[kernel]['model']
                        print(f"Making prediction with {kernel} kernel...")
                        
                        # Make prediction
                        prediction = model.predict([img_processed])[0]
                        
                        # Get confidence using probability estimates
                        try:
                            confidence = model.predict_proba([img_processed])[0].max() * 100
                        except:
                            confidence = 50.0  # Fallback confidence
                        
                        # Map prediction to label
                        label = "🐱 Cat" if prediction == 0 else "🐶 Dog"
                        
                        # Keep reasonable clipping to avoid extremes if model reports poorly calibrated values
                        confidence = max(confidence, 5.0)
                        confidence = min(confidence, 99.0)
                        
                        predictions[kernel] = {
                            'label': label,
                            'confidence': round(confidence, 1)
                        }
                        print(f"+ {kernel} prediction: {label} ({confidence:.1f}%)")
                        
                    except Exception as e:
                        print(f"Model prediction error for {kernel}: {str(e)}")
                        predictions[kernel] = 'Error'
                else:
                    print(f"Kernel {kernel} not available")
                    predictions[kernel] = 'Not Available'
            
            # Convert image to base64 for display
            with open(filepath, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode()
            
            # Clean up uploaded file
            os.remove(filepath)
            
            print(f"Upload successful. Predictions: {predictions}")
            print(f"=== UPLOAD REQUEST END ===")
            
            return jsonify({
                'success': True,
                'image': img_data,
                'predictions': predictions
            })
        
        print(f"Invalid file type: {file.filename}")
        return jsonify({'error': 'Invalid file type'})
    except Exception as e:
        print(f"Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"=== UPLOAD REQUEST END WITH ERROR ===")
        return jsonify({'error': f'Server error: {str(e)}'})

@app.route('/predict_sample/<sample_type>/<int:index>')
def predict_sample(sample_type, index):
    """Predict on sample images"""
    # Check if models are already loaded
    if not model_results or len(model_results) == 0:
        print("Models not loaded, initializing now...")
        ensure_models_loaded()
    
    if index >= len(sample_images):
        return jsonify({'error': 'Sample index out of range'})
    
    sample = sample_images[index]
    if sample['type'] != sample_type:
        return jsonify({'error': 'Invalid sample type'})
    
    # Preprocess image
    img_processed = preprocess_image(sample['path'])
    if img_processed is None:
        return jsonify({'error': 'Failed to process image'})
    
    # Make prediction with RBF kernel (best performing)
    model = model_results['rbf']['model']
    pred = model.predict([img_processed])[0]
    prediction = 'Cat' if pred == 0 else 'Dog'
    
    # Get confidence using probability estimates
    try:
        probabilities = model.predict_proba([img_processed])[0]
        if prediction == 'Cat':
            confidence = probabilities[0]  # Probability of being a cat
        else:
            confidence = probabilities[1]  # Probability of being a dog
        
        # Keep realistic confidence (no artificial boosting)
        confidence = max(confidence, 0.25)  # Minimum 25%
        confidence = min(confidence, 0.90)  # Maximum 90%
        
        confidence_normalized = confidence
    except:
        confidence_normalized = 0.5
    
    # Convert image to base64
    with open(sample['path'], 'rb') as img_file:
        img_data = base64.b64encode(img_file.read()).decode()
    
    return jsonify({
        'success': True,
        'image': img_data,
        'true_label': sample['label'],
        'prediction': prediction,
        'confidence': confidence_normalized
    })

@app.route('/api/model_info')
def model_info():
    """Get model information and results"""
    # Check if models are already loaded
    if not model_results or len(model_results) == 0:
        print("Models not loaded, initializing now...")
        ensure_models_loaded()
    
    print(f"=== MODEL INFO REQUEST ===")
    print(f"Model results keys: {list(model_results.keys())}")
    print(f"Model results content: {model_results}")
    
    try:
        # Create confusion matrix plot
        cm_plot = create_confusion_matrix_plot()
        
        # Get dataset stats
        stats = get_dataset_stats()
        print(f"Dataset stats: {stats}")
        
        # Prepare kernel comparison data
        kernel_comparison = []
        for kernel, results in model_results.items():
            print(f"Processing kernel {kernel}: {results}")
            if 'accuracy' in results and 'precision' in results and 'recall' in results and 'f1_score' in results:
                kernel_comparison.append({
                    'kernel': kernel.upper(),
                    'accuracy': f"{results['accuracy']:.4f}",
                    'precision': f"{results['precision']:.4f}",
                    'recall': f"{results['recall']:.4f}",
                    'f1_score': f"{results['f1_score']:.4f}"
                })
            else:
                print(f"Missing metrics for kernel {kernel}: {list(results.keys())}")
        
        print(f"Kernel comparison: {kernel_comparison}")
        
        response_data = {
            'kernel_comparison': kernel_comparison,
            'confusion_matrix': cm_plot,
            'dataset_stats': stats,
            'best_kernel': 'RBF',
            'best_accuracy': f"{model_results['rbf']['accuracy']:.4f}" if 'rbf' in model_results else "N/A"
        }
        
        print(f"Response data: {response_data}")
        print(f"=== MODEL INFO REQUEST END ===")
        
        return jsonify(response_data)
    except Exception as e:
        print(f"Error in model_info: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'})

@app.route('/api/sample_images')
def get_samples():
    """Get sample images for gallery"""
    # Check if models are already loaded
    if not sample_images or len(sample_images) == 0:
        print("Sample images not loaded, initializing now...")
        ensure_models_loaded()
    
    samples = []
    for i, sample in enumerate(sample_images):
        with open(sample['path'], 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
        
        samples.append({
            'index': i,
            'type': sample['type'],
            'label': sample['label'],
            'image': img_data
        })
    
    return jsonify({'samples': samples})

@app.route('/metrics')
def get_metrics():
    """Get model metrics for the dashboard"""
    try:
        # Models should already be loaded at startup
        if not model_results or len(model_results) == 0:
            print("ERROR: Models not loaded! This should not happen.")
            return jsonify({'error': 'Models not loaded. Please restart the application.'})
        
        # Get dataset stats
        stats = get_dataset_stats()
        print(f"Dataset stats: {stats}")
        
        # Prepare kernel comparison data
        kernel_comparison = []
        best_kernel = None
        best_accuracy = 0.0
        
        for kernel, results in model_results.items():
            print(f"Processing kernel {kernel}: {list(results.keys())}")
            if 'accuracy' in results and 'precision' in results and 'recall' in results and 'f1_score' in results:
                kernel_comparison.append({
                    'kernel': kernel.upper(),
                    'accuracy': f"{results['accuracy']:.4f}",
                    'precision': f"{results['precision']:.4f}",
                    'recall': f"{results['recall']:.4f}",
                    'f1_score': f"{results['f1_score']:.4f}"
                })
                
                # Track best performing kernel
                if results['accuracy'] > best_accuracy:
                    best_accuracy = results['accuracy']
                    best_kernel = kernel.upper()
            else:
                print(f"Missing metrics for kernel {kernel}: {list(results.keys())}")
        
        print(f"Kernel comparison: {kernel_comparison}")
        
        # Create confusion matrix plot
        try:
            cm_plot = create_confusion_matrix_plot()
        except Exception as cm_error:
            print(f"Error creating confusion matrix: {cm_error}")
            cm_plot = None
        
        response_data = {
            'dataset_stats': stats,
            'kernel_comparison': kernel_comparison,
            'confusion_matrix': cm_plot,
            'best_kernel': best_kernel,
            'best_accuracy': f"{best_accuracy:.4f}" if best_kernel else "N/A"
        }
        
        print(f"Metrics endpoint response: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in metrics endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'})

if __name__ == '__main__':
    # Direct run (development): initialize immediately
    print("=== STARTING APP ===")
    print("Calling load_trained_models()...")
    load_trained_models()
    print(f"Models loaded: {list(model_results.keys())}")
    print("Calling get_sample_images()...")
    get_sample_images()
    print("Web application starting...")
    print("Access the application at: http://localhost:5001")
    print("Network access: http://192.168.179.179:5001")
    # Development server only; for production use waitress
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False, threaded=True)