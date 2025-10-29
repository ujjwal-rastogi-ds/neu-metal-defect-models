"""
CNN Models for NEU Metal Surface Defects Detection
Includes: Custom CNN, Transfer Learning (VGG16, ResNet50, EfficientNet)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support

import warnings
warnings.filterwarnings('ignore')

class NEUDefectCNN:
    """CNN Model Training and Evaluation for NEU Dataset"""
    
    def __init__(self, data_dir, img_height=200, img_width=200, batch_size=32):
        """
        Initialize CNN training
        
        Args:
            data_dir: Path to split dataset (train/val/test)
            img_height: Image height
            img_width: Image width
            batch_size: Batch size for training
        """
        self.data_dir = data_dir
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.classes = ['Crazing', 'Inclusion', 'Patches', 
                       'Pitted_surface', 'Rolled-in_scale', 'Scratches']
        self.num_classes = len(self.classes)
        
        # Create results directory
        self.results_dir = 'results'
        os.makedirs(self.results_dir, exist_ok=True)
        
    def create_data_generators(self, augmentation=True):
        """Create data generators with augmentation"""
        
        if augmentation:
            # Training data augmentation
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        # Validation and test data (no augmentation)
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        self.train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'train'),
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            color_mode='grayscale',
            class_mode='categorical',
            shuffle=True
        )
        
        self.val_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'val'),
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            color_mode='grayscale',
            class_mode='categorical',
            shuffle=False
        )
        
        self.test_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'test'),
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            color_mode='grayscale',
            class_mode='categorical',
            shuffle=False
        )
        
        print("✓ Data generators created successfully")
        print(f"  Training samples: {self.train_generator.samples}")
        print(f"  Validation samples: {self.val_generator.samples}")
        print(f"  Test samples: {self.test_generator.samples}")
    
    def build_custom_cnn(self):
        """Build custom CNN architecture"""
        
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(self.img_height, self.img_width, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_transfer_learning_model(self, base_model_name='VGG16'):
        """
        Build transfer learning model
        
        Args:
            base_model_name: 'VGG16', 'ResNet50', or 'EfficientNetB0'
        """
        
        # Convert grayscale to RGB for transfer learning
        input_layer = layers.Input(shape=(self.img_height, self.img_width, 1))
        x = layers.Conv2D(3, (1, 1), padding='same')(input_layer)
        
        # Load pre-trained base model
        if base_model_name == 'VGG16':
            base_model = VGG16(weights='imagenet', include_top=False, 
                              input_shape=(self.img_height, self.img_width, 3))
        elif base_model_name == 'ResNet50':
            base_model = ResNet50(weights='imagenet', include_top=False,
                                 input_shape=(self.img_height, self.img_width, 3))
        elif base_model_name == 'EfficientNetB0':
            base_model = EfficientNetB0(weights='imagenet', include_top=False,
                                       input_shape=(self.img_height, self.img_width, 3))
        else:
            raise ValueError("Unsupported base model")
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        output_layer = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=input_layer, outputs=output_layer)
        
        return model
    
    def train_model(self, model, model_name, epochs=50, initial_lr=0.001):
        """
        Train model with callbacks
        
        Args:
            model: Keras model
            model_name: Name for saving model
            epochs: Number of epochs
            initial_lr: Initial learning rate
        """
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=initial_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        model.summary()
        
        # Callbacks
        model_path = os.path.join(self.results_dir, f'{model_name}_best.h5')
        
        callbacks = [
            ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            CSVLogger(
                os.path.join(self.results_dir, f'{model_name}_training.csv')
            )
        ]
        
        # Train model
        history = model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"\n✓ Training completed for {model_name}")
        print(f"  Best model saved: {model_path}")
        
        return history, model
    
    def evaluate_model(self, model, model_name):
        """Evaluate model on test set"""
        
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}")
        
        # Predictions
        test_loss, test_acc = model.evaluate(self.test_generator, verbose=0)
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Get predictions
        y_pred = model.predict(self.test_generator, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = self.test_generator.classes
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_classes, 
                                   target_names=self.classes))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        
        # Calculate per-class metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_classes, average=None
        )
        
        metrics_df = pd.DataFrame({
            'Class': self.classes,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        print("\nPer-Class Metrics:")
        print(metrics_df.to_string(index=False))
        
        # Save results
        results = {
            'model_name': model_name,
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            'classification_report': classification_report(
                y_true, y_pred_classes, target_names=self.classes, output_dict=True
            ),
            'confusion_matrix': cm.tolist()
        }
        
        with open(os.path.join(self.results_dir, f'{model_name}_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        return cm, metrics_df, test_acc
    
    def plot_training_history(self, history, model_name):
        """Plot training history"""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
        axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[1].set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{model_name}_history.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm, model_name):
        """Plot confusion matrix"""
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.classes,
                   yticklabels=self.classes,
                   cbar_kws={'label': 'Count'})
        plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{model_name}_confusion_matrix.png'),
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_models(self, results_dict):
        """Compare multiple models"""
        
        model_names = list(results_dict.keys())
        accuracies = [results_dict[name]['accuracy'] for name in model_names]
        
        # Bar plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, accuracies, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        plt.title('Model Comparison - Test Accuracy', fontsize=14, fontweight='bold')
        plt.ylabel('Test Accuracy')
        plt.ylim([0.8, 1.0])
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create comparison table
        comparison_df = pd.DataFrame({
            'Model': model_names,
            'Test Accuracy': accuracies
        })
        comparison_df = comparison_df.sort_values('Test Accuracy', ascending=False)
        
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        print(comparison_df.to_string(index=False))
        print("="*60)
        
        return comparison_df


def main():
    """Main training pipeline"""
    
    # Configuration
    # Change all backslashes to forward slashes
    DATA_DIR = r"C:\Users\ujjwa\Desktop\nue_cnn\data\neu_split"  # Change to your split dataset path
    
    EPOCHS = 50
    BATCH_SIZE = 32
    
    print("="*60)
    print("NEU DEFECT DETECTION - MODEL TRAINING")
    print("="*60)
    
    # Initialize
    neu_cnn = NEUDefectCNN(DATA_DIR, batch_size=BATCH_SIZE)
    
    # Create data generators
    neu_cnn.create_data_generators(augmentation=True)
    
    # Dictionary to store results
    all_results = {}
    
    # 1. Train Custom CNN
    print("\n" + "="*60)
    print("1. CUSTOM CNN")
    print("="*60)
    custom_model = neu_cnn.build_custom_cnn()
    history_custom, custom_model = neu_cnn.train_model(
        custom_model, 'CustomCNN', epochs=EPOCHS
    )
    neu_cnn.plot_training_history(history_custom, 'CustomCNN')
    cm_custom, metrics_custom, acc_custom = neu_cnn.evaluate_model(custom_model, 'CustomCNN')
    neu_cnn.plot_confusion_matrix(cm_custom, 'CustomCNN')
    all_results['CustomCNN'] = {'accuracy': acc_custom}
    
    # 2. Train VGG16
    print("\n" + "="*60)
    print("2. VGG16 TRANSFER LEARNING")
    print("="*60)
    vgg16_model = neu_cnn.build_transfer_learning_model('VGG16')
    history_vgg16, vgg16_model = neu_cnn.train_model(
        vgg16_model, 'VGG16', epochs=EPOCHS
    )
    neu_cnn.plot_training_history(history_vgg16, 'VGG16')
    cm_vgg16, metrics_vgg16, acc_vgg16 = neu_cnn.evaluate_model(vgg16_model, 'VGG16')
    neu_cnn.plot_confusion_matrix(cm_vgg16, 'VGG16')
    all_results['VGG16'] = {'accuracy': acc_vgg16}
    
    # 3. Train ResNet50
    print("\n" + "="*60)
    print("3. RESNET50 TRANSFER LEARNING")
    print("="*60)
    resnet_model = neu_cnn.build_transfer_learning_model('ResNet50')
    history_resnet, resnet_model = neu_cnn.train_model(
        resnet_model, 'ResNet50', epochs=EPOCHS
    )
    neu_cnn.plot_training_history(history_resnet, 'ResNet50')
    cm_resnet, metrics_resnet, acc_resnet = neu_cnn.evaluate_model(resnet_model, 'ResNet50')
    neu_cnn.plot_confusion_matrix(cm_resnet, 'ResNet50')
    all_results['ResNet50'] = {'accuracy': acc_resnet}
    
    # 4. Train EfficientNetB0
    print("\n" + "="*60)
    print("4. EFFICIENTNET-B0 TRANSFER LEARNING")
    print("="*60)
    effnet_model = neu_cnn.build_transfer_learning_model('EfficientNetB0')
    history_effnet, effnet_model = neu_cnn.train_model(
        effnet_model, 'EfficientNetB0', epochs=EPOCHS
    )
    neu_cnn.plot_training_history(history_effnet, 'EfficientNetB0')
    cm_effnet, metrics_effnet, acc_effnet = neu_cnn.evaluate_model(effnet_model, 'EfficientNetB0')
    neu_cnn.plot_confusion_matrix(cm_effnet, 'EfficientNetB0')
    all_results['EfficientNetB0'] = {'accuracy': acc_effnet}
    
    # Compare all models
    comparison_df = neu_cnn.compare_models(all_results)
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETED!")
    print(f"All results saved in: {neu_cnn.results_dir}")
    print("="*60)
    
    return comparison_df


if __name__ == "__main__":
    main()