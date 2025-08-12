#!/usr/bin/env python3
"""
Debug script to test fallback training logic
"""

import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import cv2

def preprocess_image(image_path):
    """Preprocess image for SVM prediction"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Resize
    img = cv2.resize(img, (64, 64))
    
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape(64, 64, 1)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    # Flatten
    img_flat = img.flatten()
    
    return img_flat

def main():
    print("=== DEBUG TRAINING ===")
    
    # Check dataset structure
    dataset_base = os.path.join(os.getcwd(), 'dataset')
    print(f"Dataset base: {dataset_base}")
    print(f"Dataset exists: {os.path.exists(dataset_base)}")
    
    if os.path.exists(dataset_base):
        train_path = os.path.join(dataset_base, 'train')
        test_path = os.path.join(dataset_base, 'test')
        print(f"Train path: {train_path}")
        print(f"Test path: {test_path}")
        print(f"Train exists: {os.path.exists(train_path)}")
        print(f"Test exists: {os.path.exists(test_path)}")
        
        if os.path.exists(train_path):
            train_cats = os.path.join(train_path, 'cats')
            train_dogs = os.path.join(train_path, 'dogs')
            print(f"Train cats: {train_cats} - {os.path.exists(train_cats)}")
            print(f"Train dogs: {train_dogs} - {os.path.exists(train_dogs)}")
            
            if os.path.exists(train_cats):
                cat_files = [f for f in os.listdir(train_cats) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                print(f"Cat files found: {len(cat_files)}")
                if cat_files:
                    print(f"Sample cat file: {cat_files[0]}")
            
            if os.path.exists(train_dogs):
                dog_files = [f for f in os.listdir(train_dogs) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                print(f"Dog files found: {len(dog_files)}")
                if dog_files:
                    print(f"Sample dog file: {dog_files[0]}")
    
    # Try to collect data
    print("\n=== COLLECTING DATA ===")
    try:
        def collect_xy(folder):
            X, y = [], []
            cat_dir = os.path.join(folder, 'cats')
            dog_dir = os.path.join(folder, 'dogs')
            print(f"  Collecting from {folder}")
            print(f"  Cat dir: {cat_dir} - {os.path.exists(cat_dir)}")
            print(f"  Dog dir: {dog_dir} - {os.path.exists(dog_dir)}")
            
            for subdir, label in [(cat_dir, 0), (dog_dir, 1)]:
                if os.path.exists(subdir):
                    files = [f for f in os.listdir(subdir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    print(f"    {subdir}: {len(files)} files")
                    for fname in files[:5]:  # Limit to first 5 for testing
                        path = os.path.join(subdir, fname)
                        arr = preprocess_image(path)
                        if arr is not None:
                            X.append(arr)
                            y.append(label)
                            print(f"      ✓ {fname} -> shape {arr.shape}")
                        else:
                            print(f"      ✗ {fname} -> failed")
                else:
                    print(f"    {subdir}: does not exist")
            
            if X:
                return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)
            else:
                return np.array([]), np.array([])

        X_train, y_train = collect_xy(os.path.join(dataset_base, 'train'))
        X_test, y_test = collect_xy(os.path.join(dataset_base, 'test'))
        
        print(f"\nData shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_test: {y_test.shape}")
        
        if len(X_train) > 0 and len(X_test) > 0:
            print("\n=== TRAINING MODELS ===")
            kernels = ['linear', 'rbf', 'poly']
            models = {}
            
            for k in kernels:
                print(f"Training {k} SVM...")
                try:
                    clf = make_pipeline(StandardScaler(with_mean=True), SVC(kernel=k, probability=True, gamma='scale'))
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
                    
                    models[k] = {
                        'model': clf,
                        'accuracy': acc,
                        'precision': prec,
                        'recall': rec,
                        'f1_score': f1,
                        'y_test': y_test,
                        'y_pred': y_pred,
                    }
                    print(f"  ✓ {k} trained. Acc={acc:.4f}")
                except Exception as e:
                    print(f"  ✗ {k} failed: {e}")
            
            print(f"\nModels trained: {list(models.keys())}")
            for k, v in models.items():
                print(f"  {k}: {list(v.keys())}")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 