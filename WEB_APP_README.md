# 🐱🐶 SVM Cat-Dog Classifier Web Application

## 🚀 **Interactive Web Application**

The SVM Cat-Dog Classifier is now available as a **fully interactive web application** with upload functionality, real-time predictions, and comprehensive model visualization!

## 🌐 **Access the Web App**

**URL:** http://localhost:5000

The web application is currently running and accessible at the above URL.

## ✨ **Features**

### 📤 **Image Upload & Classification**
- **Upload any image** (.jpg, .jpeg, .png)
- **Auto-resize and preprocess** images (64x64 pixels, grayscale)
- **Real-time SVM prediction** with confidence scores
- **Multiple kernel results** (Linear, RBF, Polynomial)

### 📊 **Model Information Panel**
- **Dataset Statistics**: Training/Test image counts
- **Kernel Performance Comparison**: Accuracy, Precision, Recall, F1-Score
- **Best Model Highlight**: RBF kernel with ~58% accuracy
- **Confusion Matrix**: Visual representation of model performance

### 🖼️ **Sample Image Gallery**
- **12 sample images** from the Kaggle dataset (6 cats, 6 dogs)
- **Click to test**: Instant predictions on sample images
- **True vs Predicted**: Compare actual vs predicted labels
- **Confidence visualization**: Visual confidence bars

### 🎯 **Prediction Results**
- **Cat 🐱 or Dog 🐶** classification
- **Confidence percentage** for each prediction
- **Multiple kernel predictions** for comparison
- **Visual confidence bars** with color coding

## 🛠️ **Technical Details**

### **Model Performance**
- **Dataset**: Real Kaggle Cats vs Dogs (2000 images)
- **Best Accuracy**: ~58% (RBF kernel)
- **Feature Engineering**: Pixel values, histograms, edge detection
- **Preprocessing**: 64x64 resize, grayscale conversion, normalization

### **Web Technologies**
- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Real-time**: AJAX for seamless user experience
- **Visualization**: Matplotlib, Seaborn for charts

## 📋 **How to Use**

1. **Open your browser** and go to http://localhost:5000
2. **Upload an image** using the file selector
3. **Click "Classify Image"** to get predictions
4. **View results** with confidence scores
5. **Try sample images** by clicking on the gallery
6. **Explore model info** in the information panel

## 🔧 **Running the Application**

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web application
python app.py

# Access at: http://localhost:5000
```

## 📁 **Project Structure**

```
SCT_ML_3/
├── app.py                          # Flask web application
├── templates/
│   └── index.html                  # Main web interface
├── svm_cat_dog_classifier.py       # Core SVM classifier
├── improved_svm_classifier.py      # Enhanced version with better accuracy
├── dataset/                        # Kaggle dataset (organized)
│   ├── train/cats/                 # Training cat images
│   ├── train/dogs/                 # Training dog images
│   ├── test/cats/                  # Test cat images
│   └── test/dogs/                  # Test dog images
├── uploads/                        # Temporary upload folder
└── requirements.txt                # Python dependencies
```

## 🎨 **User Interface**

### **Modern Design**
- **Responsive layout** that works on all devices
- **Gradient backgrounds** with professional styling
- **Interactive elements** with hover effects
- **Loading animations** for better UX

### **Visual Elements**
- **Emoji indicators** (🐱 for cats, 🐶 for dogs)
- **Color-coded confidence bars** (green for high confidence)
- **Professional data tables** for model metrics
- **Image galleries** with hover effects

## 🔍 **Model Insights**

### **Why ~58% Accuracy?**
- **Real-world data**: Complex lighting, angles, breeds
- **Raw pixel features**: No CNN or advanced feature extraction
- **Standard SVM**: Baseline performance expected
- **Realistic benchmark**: Demonstrates real ML challenges

### **Kernel Comparison**
- **RBF Kernel**: Best performance (57.81%)
- **Polynomial Kernel**: Moderate performance (54.37%)
- **Linear Kernel**: Basic performance (49.69%)

## 🌟 **Key Achievements**

✅ **Complete web application** with upload functionality  
✅ **Real-time predictions** with confidence scores  
✅ **Interactive sample gallery** for testing  
✅ **Comprehensive model visualization**  
✅ **Professional UI/UX design**  
✅ **Real Kaggle dataset integration**  
✅ **Multiple SVM kernel comparison**  
✅ **Responsive design** for all devices  

## 🚀 **Next Steps**

The web application is **fully functional** and ready for use! You can:

1. **Test with your own images** by uploading them
2. **Explore the sample gallery** to see predictions
3. **Analyze model performance** through the information panel
4. **Compare different kernels** and their results

**Enjoy exploring the SVM Cat-Dog Classifier! 🎉** 