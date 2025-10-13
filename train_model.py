import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import zipfile
import os
import numpy as np

class BloodGroupClassifier:
    def __init__(self, dataset_path, img_size=(128, 128), batch_size=32):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = None
        
    def extract_dataset(self, zip_path, extract_to):
        """Extract dataset from zip file"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"[OK] Dataset extracted to {extract_to}")
        except Exception as e:
            print(f"[ERROR] Error extracting dataset: {e}")
            return False
        return True
    
    def create_data_generators(self, validation_split=0.2):
        """Create optimized data generators with augmentation"""
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            brightness_range=[0.8, 1.2],
            validation_split=validation_split
        )
        
        # Validation data generator (only rescaling)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Training generator
        self.train_generator = train_datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Validation generator
        self.val_generator = val_datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        # Store class names
        self.class_names = list(self.train_generator.class_indices.keys())
        print(f"[OK] Classes found: {self.class_names}")
        print(f"[OK] Training samples: {self.train_generator.samples}")
        print(f"[OK] Validation samples: {self.val_generator.samples}")
        
    def build_optimized_model(self):
        """Build an optimized CNN model"""
        self.model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Fourth Convolutional Block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Fully Connected Layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(len(self.class_names), activation='softmax')
        ])
        
        # Compile with optimized settings
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        print("[OK] Model built and compiled successfully")
        print(f"[OK] Total parameters: {self.model.count_params():,}")
        
    def get_callbacks(self):
        """Define training callbacks for optimization"""
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
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
            ModelCheckpoint(
                'best_blood_group_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        return callbacks
        
    def train_model(self, epochs=50):
        """Train the model with callbacks"""
        print("[INFO] Starting model training...")
        
        # Calculate steps (cap for quick training)
        steps_per_epoch = max(1, min(10, self.train_generator.samples // self.batch_size))
        validation_steps = max(1, min(5, self.val_generator.samples // self.batch_size))
        
        # Train model
        self.history = self.model.fit(
            self.train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=self.val_generator,
            validation_steps=validation_steps,
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        print("[OK] Training completed!")
        
    def evaluate_model(self):
        """Evaluate model performance"""
        print("[INFO] Evaluating model...")
        
        # Evaluate on validation set
        val_loss, val_accuracy, val_top_k = self.model.evaluate(
            self.val_generator,
            steps=self.val_generator.samples // self.batch_size,
            verbose=0
        )
        
        print(f"[OK] Validation Results:")
        print(f"   Accuracy: {val_accuracy*100:.2f}%")
        print(f"   Top-K Accuracy: {val_top_k*100:.2f}%")
        print(f"   Loss: {val_loss:.4f}")
        
        return val_accuracy, val_loss
        
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("[WARN] No training history available")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def save_model(self, filepath='blood_group_model.h5'):
        """Save the trained model"""
        if self.model is None:
            print("❌ No model to save")
            return
            
        self.model.save(filepath)
        print(f"[OK] Model saved to {filepath}")
        
        # Save class names
        class_names_path = filepath.replace('.h5', '_classes.txt')
        with open(class_names_path, 'w') as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")
        print(f"[OK] Class names saved to {class_names_path}")

def main():
    """Main training pipeline"""
    print("Blood Group Classification Model Training")
    print("=" * 50)
    
    # Configuration
    cwd = os.getcwd()
    ZIP_PATH = os.path.join(cwd, "archive.zip")  # Optional: place zip here if needed
    EXTRACT_PATH = os.path.join(cwd, "dataset")
    DATASET_PATH = os.path.join(cwd, "dataset")
    
    # Initialize classifier
    classifier = BloodGroupClassifier(
        dataset_path=DATASET_PATH,
        img_size=(128, 128),
        batch_size=32
    )
    
    # Extract dataset if zip exists; otherwise assume dataset folder already present
    if os.path.exists(ZIP_PATH):
        if not classifier.extract_dataset(ZIP_PATH, EXTRACT_PATH):
            return
    else:
        print(f"[INFO] No zip found at {ZIP_PATH}. Using existing dataset folder: {DATASET_PATH}")
    
    # Create data generators
    classifier.create_data_generators(validation_split=0.2)
    
    # Build model
    classifier.build_optimized_model()
    
    # Display model architecture
    classifier.model.summary()
    
    # Train model
    classifier.train_model(epochs=1)
    
    # Evaluate model
    classifier.evaluate_model()
    
    # Plot training history
    classifier.plot_training_history()
    
    # Save model
    classifier.save_model('blood_group_model.h5')
    
    print("\n[OK] Training pipeline completed successfully!")

if __name__ == "__main__":
    main()