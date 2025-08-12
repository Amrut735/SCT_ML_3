import streamlit as st
import numpy as np
import cv2
import joblib
import os
from PIL import Image
import io
import base64
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="SVM Cat/Dog Classifier",
    page_icon="🐱🐶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .prediction-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Global variables
IMG_SIZE = 64
GRAYSCALE = True
model_results = {}

def preprocess_image(image_path):
    """Preprocess image for SVM input"""
    try:
        if GRAYSCALE:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.flatten().astype(np.float32) / 255.0
        return img
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def load_trained_models():
    """Load or train SVM models"""
    global model_results
    
    if not model_results:
        st.info("Training models... This may take a few minutes on first run.")
        
        # Check if dataset exists
        dataset_base = os.path.join(os.getcwd(), 'dataset')
        if os.path.exists(os.path.join(dataset_base, 'train')) and os.path.exists(os.path.join(dataset_base, 'test')):
            try:
                # Collect training data
                X_train, y_train = [], []
                X_test, y_test = [], []
                
                # Load training data
                for label, folder in enumerate(['cats', 'dogs']):
                    train_path = os.path.join(dataset_base, 'train', folder)
                    test_path = os.path.join(dataset_base, 'test', folder)
                    
                    if os.path.exists(train_path):
                        for fname in os.listdir(train_path)[:100]:  # Limit for demo
                            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                                path = os.path.join(train_path, fname)
                                arr = preprocess_image(path)
                                if arr is not None:
                                    X_train.append(arr)
                                    y_train.append(label)
                    
                    if os.path.exists(test_path):
                        for fname in os.listdir(test_path)[:25]:  # Limit for demo
                            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                                path = os.path.join(test_path, fname)
                                arr = preprocess_image(path)
                                if arr is not None:
                                    X_test.append(arr)
                                    y_test.append(label)
                
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                X_test = np.array(X_test)
                y_test = np.array(y_test)
                
                if len(X_train) > 0 and len(X_test) > 0:
                    # Train models
                    kernels = ['linear', 'rbf', 'poly']
                    for kernel in kernels:
                        st.write(f"Training {kernel} SVM...")
                        clf = make_pipeline(StandardScaler(), SVC(kernel=kernel, probability=True, gamma='scale'))
                        clf.fit(X_train, y_train)
                        
                        y_pred = clf.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)
                        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
                        
                        model_results[kernel] = {
                            'model': clf,
                            'accuracy': acc,
                            'precision': prec,
                            'recall': rec,
                            'f1_score': f1
                        }
                    
                    st.success(f"Models trained successfully! Available: {list(model_results.keys())}")
                else:
                    st.error("No training data found")
                    
            except Exception as e:
                st.error(f"Error training models: {str(e)}")
        else:
            st.error("Dataset not found. Please ensure 'dataset/train' and 'dataset/test' folders exist.")
    
    return model_results

def predict_image(image_array, model):
    """Make prediction using trained model"""
    try:
        # Preprocess the uploaded image
        if GRAYSCALE:
            img = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            img = image_array
        
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.flatten().astype(np.float32) / 255.0
        img = img.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(img)[0]
        confidence = model.predict_proba(img)[0].max()
        
        label = "🐱 Cat" if prediction == 0 else "🐶 Dog"
        confidence_pct = confidence * 100
        
        return label, confidence_pct
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Error", 0

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🐱🐶 SVM Cat/Dog Classifier</h1>
        <p>Machine Learning-powered image classification using Support Vector Machines</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Home", "Image Classification", "Model Performance", "About"]
    )
    
    if page == "Home":
        show_home_page()
    elif page == "Image Classification":
        show_classification_page()
    elif page == "Model Performance":
        show_performance_page()
    elif page == "About":
        show_about_page()

def show_home_page():
    st.header("🏠 Welcome to SVM Cat/Dog Classifier")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 What This App Does
        
        This application uses **Support Vector Machines (SVM)** with different kernels to classify images as either cats or dogs.
        
        **Features:**
        - 🖼️ Image upload and classification
        - 🤖 Multiple SVM kernels (Linear, RBF, Polynomial)
        - 📊 Performance metrics and comparisons
        - 🎨 Beautiful, responsive interface
        
        **How it works:**
        1. Upload an image of a cat or dog
        2. The app processes it using trained SVM models
        3. Get predictions from all three kernel types
        - Compare accuracy and confidence levels
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Getting Started
        
        1. **Navigate** to "Image Classification" in the sidebar
        2. **Upload** an image of a cat or dog
        3. **View** predictions from all SVM models
        4. **Check** "Model Performance" for detailed metrics
        
        ### 🔬 Technical Details
        
        - **Image Processing**: 64x64 grayscale conversion
        - **ML Framework**: Scikit-learn SVM implementation
        - **Kernels**: Linear, RBF (Radial Basis Function), Polynomial
        - **Training**: Automatic model training from dataset
        
        ### 📁 Dataset Requirements
        
        The app expects a dataset structure:
        ```
        dataset/
        ├── train/
        │   ├── cats/
        │   └── dogs/
        └── test/
            ├── cats/
            └── dogs/
        ```
        """)

def show_classification_page():
    st.header("🖼️ Image Classification")
    
    # Load models
    models = load_trained_models()
    
    if not models:
        st.warning("Please wait for models to load or check dataset availability.")
        return
    
    # File upload
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image of a cat or dog to classify"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Make predictions
        st.subheader("🔮 Predictions")
        
        col1, col2, col3 = st.columns(3)
        
        for i, (kernel, model_data) in enumerate(models.items()):
            with [col1, col2, col3][i]:
                label, confidence = predict_image(image_array, model_data['model'])
                
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>{kernel.upper()} Kernel</h3>
                    <h2>{label}</h2>
                    <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                    <p><strong>Accuracy:</strong> {model_data['accuracy']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)

def show_performance_page():
    st.header("📊 Model Performance")
    
    models = load_trained_models()
    
    if not models:
        st.warning("No models available. Please train models first.")
        return
    
    # Performance metrics
    st.subheader("📈 Model Comparison")
    
    metrics_data = []
    for kernel, data in models.items():
        metrics_data.append({
            'Kernel': kernel.upper(),
            'Accuracy': f"{data['accuracy']:.3f}",
            'Precision': f"{data['precision']:.3f}",
            'Recall': f"{data['recall']:.3f}",
            'F1-Score': f"{data['f1_score']:.3f}"
        })
    
    # Display metrics table
    st.dataframe(metrics_data, use_container_width=True)
    
    # Performance visualization
    st.subheader("📊 Performance Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy comparison
        kernels = list(models.keys())
        accuracies = [models[k]['accuracy'] for k in kernels]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(kernels, accuracies, color=['#667eea', '#764ba2', '#f093fb'])
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy Comparison')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom')
        
        st.pyplot(fig)
    
    with col2:
        # F1-Score comparison
        f1_scores = [models[k]['f1_score'] for k in kernels]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(kernels, f1_scores, color=['#667eea', '#764ba2', '#f093fb'])
        ax.set_ylabel('F1-Score')
        ax.set_title('Model F1-Score Comparison')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, f1 in zip(bars, f1_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{f1:.3f}', ha='center', va='bottom')
        
        st.pyplot(fig)

def show_about_page():
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This is a **Machine Learning web application** that demonstrates the use of Support Vector Machines (SVM) for image classification tasks.
    
    ### 🔬 Technical Implementation
    
    - **Backend**: Python with Streamlit framework
    - **Machine Learning**: Scikit-learn SVM implementation
    - **Image Processing**: OpenCV and PIL for image manipulation
    - **Data Visualization**: Matplotlib and Seaborn for charts
    
    ### 🏗️ Architecture
    
    The application uses a **modular design** with:
    - **Data preprocessing** pipeline for image standardization
    - **Multiple SVM kernels** for comparison (Linear, RBF, Polynomial)
    - **Performance evaluation** with comprehensive metrics
    - **User-friendly interface** for easy interaction
    
    ### 📚 Learning Objectives
    
    This project demonstrates:
    - SVM implementation for computer vision tasks
    - Image preprocessing techniques
    - Model performance evaluation
    - Web application development with ML integration
    
    ### 🚀 Future Enhancements
    
    Potential improvements include:
    - Real-time video classification
    - Additional ML algorithms (CNN, Random Forest)
    - Model fine-tuning capabilities
    - API endpoints for external access
    
    ### 👨‍💻 Developer
    
    Built with ❤️ for educational and research purposes.
    """)

if __name__ == "__main__":
    main() 