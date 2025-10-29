"""
Streamlit App for NEU Metal Surface Defects Detection
Run: streamlit run app.py
"""

import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import json
import traceback

import tensorflow as tf
from tensorflow import keras

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Page configuration
st.set_page_config(
    page_title="NEU Defect Detection",
    page_icon="🔧",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2c3e50;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
    }
    .stApp {
        max-width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Classes
CLASSES = ['Crazing', 'Inclusion', 'Patches', 
           'Pitted_surface', 'Rolled-in_scale', 'Scratches']

# Helper functions
@st.cache_resource
def load_model(model_path):
    """Load trained model with error handling"""
    try:
        model = keras.models.load_model(model_path)
        return model, None
    except Exception as e:
        return None, str(e)

def preprocess_image(image, target_size=(200, 200)):
    """Preprocess image for prediction"""
    try:
        img = cv2.resize(image, target_size)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def predict_defect(model, image):
    """Predict defect class"""
    try:
        preprocessed = preprocess_image(image)
        if preprocessed is None:
            return None, None, None
        
        predictions = model.predict(preprocessed, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        return CLASSES[class_idx], confidence, predictions[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None

def plot_predictions(predictions):
    """Plot prediction probabilities"""
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#3498db' if i == np.argmax(predictions) else '#95a5a6' 
              for i in range(len(predictions))]
    bars = ax.barh(CLASSES, predictions, color=colors)
    ax.set_xlabel('Confidence', fontweight='bold')
    ax.set_title('Prediction Probabilities', fontweight='bold', fontsize=14)
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}',
                ha='left', va='center', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    return fig

def get_available_models(results_dir='results'):
    """Get list of available trained models"""
    available_models = []
    
    if not os.path.exists(results_dir):
        return available_models
    
    for file in os.listdir(results_dir):
        if file.endswith('_best.h5'):
            model_name = file.replace('_best.h5', '')
            model_path = os.path.join(results_dir, file)
            # Check if file is not empty
            if os.path.getsize(model_path) > 0:
                available_models.append(model_name)
    
    return sorted(available_models)

def load_results(results_dir='results'):
    """Load training results"""
    results = {}
    
    if not os.path.exists(results_dir):
        return results
    
    for file in os.listdir(results_dir):
        if file.endswith('_results.json'):
            model_name = file.replace('_results.json', '')
            try:
                with open(os.path.join(results_dir, file), 'r') as f:
                    results[model_name] = json.load(f)
            except Exception as e:
                st.warning(f"Could not load results for {model_name}: {e}")
    
    return results

def plot_confusion_matrix(cm, title):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=CLASSES,
               yticklabels=CLASSES,
               ax=ax)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_xlabel('Predicted Label', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🔧 NEU Metal Surface Defect Detection</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Check if results directory exists
    if not os.path.exists('results'):
        st.error("⚠️ Results directory not found! Please train models first by running: `python train_models.py`")
        st.stop()
    
    # Get available models
    available_models = get_available_models('results')
    
    if not available_models:
        st.error("⚠️ No trained models found in 'results/' directory!")
        st.info("Please train models first by running: `python train_models.py`")
        st.stop()
    
    # Show available models
    st.sidebar.success(f"✅ Found {len(available_models)} trained model(s)")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", 
                           ["🏠 Home", 
                            "🔍 Single Image Prediction", 
                            "📊 Batch Prediction",
                            "📈 Model Performance",
                            "ℹ️ About Dataset"])
    
    # Page 1: Home
    if page == "🏠 Home":
        st.header("Welcome to NEU Defect Detection System")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 6 Defect Classes</h3>
                <p>Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🤖 {len(available_models)} Trained Model(s)</h3>
                <p>{', '.join(available_models)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 High Accuracy</h3>
                <p>Transfer learning models achieve 95%+ accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("Sample Defect Types")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**Crazing**")
            st.caption("Fine cracks on the surface")
        
        with col2:
            st.info("**Inclusion**")
            st.caption("Foreign material embedded in surface")
        
        with col3:
            st.info("**Patches**")
            st.caption("Irregular surface areas")
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.info("**Pitted Surface**")
            st.caption("Small cavities or pits")
        
        with col5:
            st.info("**Rolled-in Scale**")
            st.caption("Oxide scale pressed into surface")
        
        with col6:
            st.info("**Scratches**")
            st.caption("Linear marks on surface")
        
        st.markdown("---")
        
        # Show system info
        st.subheader("System Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Available Models:**")
            for model in available_models:
                st.write(f"✅ {model}")
        
        with col2:
            st.write("**TensorFlow Version:**")
            st.write(f"🔢 {tf.__version__}")
            
            # Check GPU
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                st.write(f"🎮 GPU Available: {len(gpus)} device(s)")
            else:
                st.write("💻 Running on CPU")
    
    # Page 2: Single Image Prediction
    elif page == "🔍 Single Image Prediction":
        st.header("Single Image Defect Detection")
        
        if not available_models:
            st.error("No trained models available!")
            return
        
        # Model selection
        selected_model = st.selectbox("Select Model", available_models)
        
        # File uploader
        uploaded_file = st.file_uploader("Upload a defect image", 
                                        type=['jpg', 'jpeg', 'png', 'bmp'])
        
        if uploaded_file is not None:
            try:
                # Read image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    st.error("Could not read image. Please upload a valid image file.")
                    return
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Input Image")
                    st.image(image, caption="Uploaded Image", use_container_width=True, 
                            clamp=True, channels='GRAY')
                
                with col2:
                    st.subheader("Prediction Results")
                    
                    # Load model and predict
                    model_path = os.path.join('results', f'{selected_model}_best.h5')
                    
                    with st.spinner('Loading model and making prediction...'):
                        model, error = load_model(model_path)
                        
                        if model is None:
                            st.error(f"Could not load model: {error}")
                            return
                        
                        predicted_class, confidence, all_predictions = predict_defect(model, image)
                        
                        if predicted_class is None:
                            st.error("Prediction failed!")
                            return
                    
                    # Display results
                    st.success(f"**Predicted Class:** {predicted_class}")
                    st.metric("Confidence", f"{confidence*100:.2f}%")
                    
                    # Plot probabilities
                    st.subheader("Class Probabilities")
                    fig = plot_predictions(all_predictions)
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Error processing image: {e}")
                st.error(traceback.format_exc())
    
    # Page 3: Batch Prediction
    elif page == "📊 Batch Prediction":
        st.header("Batch Image Defect Detection")
        
        if not available_models:
            st.error("No trained models available!")
            return
        
        # Model selection
        selected_model = st.selectbox("Select Model", available_models)
        
        # Multiple file uploader
        uploaded_files = st.file_uploader("Upload multiple defect images", 
                                         type=['jpg', 'jpeg', 'png', 'bmp'],
                                         accept_multiple_files=True)
        
        if uploaded_files:
            st.info(f"Uploaded {len(uploaded_files)} images")
            
            if st.button("Predict All"):
                model_path = os.path.join('results', f'{selected_model}_best.h5')
                
                with st.spinner('Loading model...'):
                    model, error = load_model(model_path)
                    
                    if model is None:
                        st.error(f"Could not load model: {error}")
                        return
                
                results_data = []
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process each image
                for idx, uploaded_file in enumerate(uploaded_files):
                    try:
                        status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {uploaded_file.name}")
                        
                        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                        
                        if image is None:
                            results_data.append({
                                'Image': uploaded_file.name,
                                'Predicted Class': 'Error',
                                'Confidence': 'N/A'
                            })
                            continue
                        
                        predicted_class, confidence, _ = predict_defect(model, image)
                        
                        if predicted_class is None:
                            results_data.append({
                                'Image': uploaded_file.name,
                                'Predicted Class': 'Error',
                                'Confidence': 'N/A'
                            })
                        else:
                            results_data.append({
                                'Image': uploaded_file.name,
                                'Predicted Class': predicted_class,
                                'Confidence': f"{confidence*100:.2f}%"
                            })
                        
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                        
                    except Exception as e:
                        st.warning(f"Error processing {uploaded_file.name}: {e}")
                        results_data.append({
                            'Image': uploaded_file.name,
                            'Predicted Class': 'Error',
                            'Confidence': 'N/A'
                        })
                
                status_text.text("Prediction Complete!")
                
                # Display results table
                st.success("✅ Batch Prediction Complete!")
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )
                
                # Class distribution (excluding errors)
                valid_predictions = results_df[results_df['Predicted Class'] != 'Error']
                
                if len(valid_predictions) > 0:
                    st.subheader("Prediction Distribution")
                    class_counts = valid_predictions['Predicted Class'].value_counts()
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    class_counts.plot(kind='bar', ax=ax, color='#3498db')
                    ax.set_title('Distribution of Predicted Classes', fontweight='bold', fontsize=14)
                    ax.set_xlabel('Defect Class', fontweight='bold')
                    ax.set_ylabel('Count', fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='y')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
    
    # Page 4: Model Performance
    elif page == "📈 Model Performance":
        st.header("Model Performance Comparison")
        
        # Load all results
        results = load_results('results')
        
        if not results:
            st.warning("⚠️ No model results found. Models may still be training or results files are missing.")
            st.info("After training completes, result JSON files will appear here.")
            return
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Overall Comparison", "🎯 Detailed Metrics", "🔥 Confusion Matrices"])
        
        with tab1:
            st.subheader("Model Accuracy Comparison")
            
            # Extract accuracies
            model_names = list(results.keys())
            accuracies = [results[name]['test_accuracy'] for name in model_names]
            
            # Create comparison dataframe
            comparison_df = pd.DataFrame({
                'Model': model_names,
                'Test Accuracy': accuracies,
                'Test Loss': [results[name]['test_loss'] for name in model_names]
            })
            comparison_df = comparison_df.sort_values('Test Accuracy', ascending=False)
            
            # Display table
            st.dataframe(comparison_df, use_container_width=True)
            
            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
            bars = ax.bar(comparison_df['Model'], comparison_df['Test Accuracy'], 
                         color=colors[:len(comparison_df)])
            ax.set_title('Model Comparison - Test Accuracy', fontweight='bold', fontsize=14)
            ax.set_ylabel('Test Accuracy', fontweight='bold')
            ax.set_ylim([0.8, 1.0])
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Best model highlight
            best_model = comparison_df.iloc[0]['Model']
            best_accuracy = comparison_df.iloc[0]['Test Accuracy']
            st.success(f"🏆 **Best Model:** {best_model} with {best_accuracy*100:.2f}% accuracy")
        
        with tab2:
            st.subheader("Detailed Classification Metrics")
            
            # Model selector
            selected_model = st.selectbox("Select Model for Detailed View", 
                                         list(results.keys()))
            
            if selected_model:
                model_results = results[selected_model]
                
                # Overall metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Test Accuracy", 
                             f"{model_results['test_accuracy']*100:.2f}%")
                
                with col2:
                    st.metric("Test Loss", 
                             f"{model_results['test_loss']:.4f}")
                
                with col3:
                    report = model_results['classification_report']
                    st.metric("Macro Avg F1-Score", 
                             f"{report['macro avg']['f1-score']:.4f}")
                
                # Per-class metrics
                st.subheader("Per-Class Performance")
                
                class_metrics = []
                for class_name in CLASSES:
                    if class_name in report:
                        class_metrics.append({
                            'Class': class_name,
                            'Precision': report[class_name]['precision'],
                            'Recall': report[class_name]['recall'],
                            'F1-Score': report[class_name]['f1-score'],
                            'Support': int(report[class_name]['support'])
                        })
                
                metrics_df = pd.DataFrame(class_metrics)
                st.dataframe(metrics_df, use_container_width=True)
                
                # Visualize per-class metrics
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                metrics_to_plot = ['Precision', 'Recall', 'F1-Score']
                colors = ['#3498db', '#2ecc71', '#e74c3c']
                
                for idx, metric in enumerate(metrics_to_plot):
                    axes[idx].barh(metrics_df['Class'], metrics_df[metric], 
                                  color=colors[idx])
                    axes[idx].set_title(metric, fontweight='bold', fontsize=12)
                    axes[idx].set_xlim([0, 1])
                    axes[idx].grid(True, alpha=0.3, axis='x')
                
                plt.tight_layout()
                st.pyplot(fig)
        
        with tab3:
            st.subheader("Confusion Matrices")
            
            # Select model
            selected_model = st.selectbox("Select Model for Confusion Matrix", 
                                         list(results.keys()), 
                                         key='cm_selector')
            
            if selected_model:
                cm = np.array(results[selected_model]['confusion_matrix'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Raw Confusion Matrix**")
                    fig = plot_confusion_matrix(cm, f'{selected_model}')
                    st.pyplot(fig)
                
                with col2:
                    st.write("**Normalized Confusion Matrix**")
                    # Normalized confusion matrix
                    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    
                    fig_norm, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                               xticklabels=CLASSES,
                               yticklabels=CLASSES,
                               ax=ax)
                    ax.set_title(f'{selected_model} - Normalized', fontweight='bold', fontsize=14)
                    ax.set_ylabel('True Label', fontweight='bold')
                    ax.set_xlabel('Predicted Label', fontweight='bold')
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                    plt.tight_layout()
                    st.pyplot(fig_norm)
    
    # Page 5: About Dataset
    elif page == "ℹ️ About Dataset":
        st.header("About NEU Metal Surface Defects Dataset")
        
        st.markdown("""
        ### Dataset Overview
        
        The NEU Metal Surface Defects Database is a comprehensive dataset for 
        surface defect detection in hot-rolled steel strips.
        
        **Key Features:**
        - **Total Images:** 1,800 grayscale images
        - **Image Size:** 200×200 pixels
        - **Classes:** 6 types of typical surface defects
        - **Distribution:** 300 samples per class
        
        ### Defect Classes
        
        1. **Crazing (Cr)**: Network of fine cracks on the surface
        2. **Inclusion (In)**: Foreign material embedded in the steel surface
        3. **Patches (Pa)**: Irregular areas with different texture/appearance
        4. **Pitted Surface (PS)**: Surface with small cavities or pits
        5. **Rolled-in Scale (RS)**: Oxide scale pressed into the surface during rolling
        6. **Scratches (Sc)**: Linear marks caused by sharp objects
        
        ### Dataset Split
        
        - **Training Set:** 70% (1,260 images)
        - **Validation Set:** 15% (270 images)
        - **Test Set:** 15% (270 images)
        
        ### Models Available
        """)
        
        # Show which models are available
        for model in available_models:
            st.write(f"✅ **{model}**")
        
        st.markdown("""
        
        ### Data Augmentation
        
        Applied augmentation techniques:
        - Rotation (±20°)
        - Width/Height shifts (20%)
        - Shear transformation
        - Zoom (20%)
        - Horizontal/Vertical flips
        
        ### Citation
        
        If you use this dataset or models, please cite:
        
        ```
        @misc{NEU_surface_defect_database,
            title={NEU surface defect database},
            author={Kechen Song and Yunhui Yan},
            year={2013}
        }
        ```
        
        ### References
        
        - Original Dataset: [NEU Surface Defect Database](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html)
        - Paper: Song, K., & Yan, Y. (2013). A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects.
        """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {e}")
        st.error(traceback.format_exc())
        st.info("Please check the console for more details.")