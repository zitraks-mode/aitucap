import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import seaborn as sns
import os
from PIL import Image

CSV_FILE = 'archive\Data_Entry_2017.csv'
IMAGE_DIR = 'archive\images'
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2
MULTI_LABEL = True 
print("Loading CSV labels...")
df = pd.read_csv(CSV_FILE)

print(f"\nTotal images: {len(df)}")
print(f"\nFirst few rows:")
print(df[['Image Index', 'Finding Labels']].head(10))

def parse_labels(label_string):
    if pd.isna(label_string) or label_string == '':
        return []
    return [label.strip() for label in str(label_string).split('|')]

df['labels_list'] = df['Finding Labels'].apply(parse_labels)

all_diseases = set()
for labels in df['labels_list']:
    all_diseases.update(labels)
class_names = sorted(list(all_diseases))
num_classes = len(class_names)

print(f"\nUnique diseases detected: {num_classes}")
print(f"Disease classes: {class_names}")

print("\nClass distribution:")
label_counts = df['Finding Labels'].value_counts()
print(label_counts.head(15))


if MULTI_LABEL:
    mlb = MultiLabelBinarizer(classes=class_names)
    y_encoded = mlb.fit_transform(df['labels_list'])
    print(f"\nUsing MULTI-LABEL classification")
else:
    df['primary_label'] = df['labels_list'].apply(lambda x: x[0] if len(x) > 0 else 'No Finding')
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    y_encoded = np.array([label_to_idx[label] for label in df['primary_label']])
    print(f"\nUsing SINGLE-LABEL classification")

train_df, val_df = train_test_split(
    df, 
    test_size=VALIDATION_SPLIT, 
    random_state=42,
    stratify=df['Finding Labels'] if not MULTI_LABEL else None
)

print(f"\nTraining samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

class XRayDataGenerator(keras.utils.Sequence):    
    def __init__(self, dataframe, image_dir, batch_size, img_size, 
                 class_names, shuffle=True, augment=False, multi_label=True):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.shuffle = shuffle
        self.augment = augment
        self.multi_label = multi_label
        self.indexes = np.arange(len(self.df))
        
        if multi_label:
            self.mlb = MultiLabelBinarizer(classes=class_names)
            self.mlb.fit([class_names])
        else:
            self.label_to_idx = {label: idx for idx, label in enumerate(class_names)}
        
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.df))
        batch_indexes = self.indexes[start_idx:end_idx]
        
        X, y = self._generate_batch(batch_indexes)
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def _load_image(self, image_path):
        try:
            img = Image.open(image_path).convert('L')
            img = img.resize(self.img_size)
            img_array = np.array(img, dtype=np.float32)
            img_array = img_array / 255.0  
            img_array = np.expand_dims(img_array, axis=-1)  
            
            if self.augment:
                img_array = self._augment_image(img_array)
            
            return img_array
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return np.zeros((*self.img_size, 1), dtype=np.float32)
    
    def _augment_image(self, img):
        """Apply random augmentations"""
        if np.random.random() > 0.5:
            angle = np.random.uniform(-5, 5)
            img = tf.image.rot90(img, k=int(angle/90))
        
        if np.random.random() > 0.5:
            img = tf.image.random_brightness(img, 0.2)
        
        if np.random.random() > 0.5:
            img = tf.image.random_contrast(img, 0.8, 1.2)
        
        return img
    
    def _generate_batch(self, batch_indexes):
        batch_size = len(batch_indexes)
        X = np.zeros((batch_size, *self.img_size, 1), dtype=np.float32)
        
        if self.multi_label:
            y = np.zeros((batch_size, self.num_classes), dtype=np.float32)
        else:
            y = np.zeros((batch_size,), dtype=np.int32)
        
        for i, idx in enumerate(batch_indexes):
            img_name = self.df.loc[idx, 'Image Index']
            img_path = os.path.join(self.image_dir, img_name)
            X[i] = self._load_image(img_path)
            
            labels_list = self.df.loc[idx, 'labels_list']
            if self.multi_label:
                y[i] = self.mlb.transform([labels_list])[0]
            else:
                primary_label = labels_list[0] if len(labels_list) > 0 else 'No Finding'
                y[i] = self.label_to_idx.get(primary_label, 0)
        
        if not self.multi_label:
            y = keras.utils.to_categorical(y, num_classes=self.num_classes)
        
        return X, y


train_generator = XRayDataGenerator(
    train_df, IMAGE_DIR, BATCH_SIZE, IMG_SIZE, 
    class_names, shuffle=True, augment=True, multi_label=MULTI_LABEL
)

val_generator = XRayDataGenerator(
    val_df, IMAGE_DIR, BATCH_SIZE, IMG_SIZE, 
    class_names, shuffle=False, augment=False, multi_label=MULTI_LABEL
)

print("\nData generators created successfully!")



def build_custom_cnn(num_classes, multi_label=True):
    """
    Custom CNN architecture for X-ray classification
    """
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
        
        
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        
        
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='sigmoid' if multi_label else 'softmax')
    ])
    
    return model



model = build_custom_cnn(num_classes, multi_label=MULTI_LABEL)
model.summary()



if MULTI_LABEL:
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
else:
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'best_xray_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1
    )
]



print("\n" + "="*50)
print("TRAINING X-RAY DISEASE CLASSIFIER")
print("="*50)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)



print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)


results = model.evaluate(val_generator)
print(f"\nValidation Loss: {results[0]:.4f}")
print(f"Validation Accuracy: {results[1]:.4f}")
print(f"Validation Precision: {results[2]:.4f}")
print(f"Validation Recall: {results[3]:.4f}")


if results[2] + results[3] > 0:
    f1_score = 2 * (results[2] * results[3]) / (results[2] + results[3])
    print(f"Validation F1-Score: {f1_score:.4f}")


print("\nGenerating predictions for detailed metrics...")
y_true = []
y_pred = []

for i in range(len(val_generator)):
    images, labels = val_generator[i]
    predictions = model.predict(images, verbose=0)
    
    if MULTI_LABEL:
        
        y_pred.extend((predictions > 0.5).astype(int))
        y_true.extend(labels.astype(int))
    else:
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(np.argmax(labels, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)



if MULTI_LABEL:
    print("\n" + "="*50)
    print("PER-CLASS METRICS (Multi-label)")
    print("="*50)
    
    for i, disease in enumerate(class_names):
        if y_true[:, i].sum() > 0:  
            report = classification_report(
                y_true[:, i], 
                y_pred[:, i], 
                target_names=['Absent', 'Present'],
                digits=4,
                zero_division=0
            )
            print(f"\n{disease}:")
            print(report)
else:
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - X-ray Disease Classification', fontsize=14, pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved as 'confusion_matrix.png'")



fig, axes = plt.subplots(2, 2, figsize=(15, 12))


axes[0, 0].plot(history.history['accuracy'], label='Train')
axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
axes[0, 0].set_title('Model Accuracy', fontsize=12)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)


axes[0, 1].plot(history.history['loss'], label='Train')
axes[0, 1].plot(history.history['val_loss'], label='Validation')
axes[0, 1].set_title('Model Loss', fontsize=12)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)


axes[1, 0].plot(history.history['precision'], label='Train')
axes[1, 0].plot(history.history['val_precision'], label='Validation')
axes[1, 0].set_title('Model Precision', fontsize=12)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)


axes[1, 1].plot(history.history['recall'], label='Train')
axes[1, 1].plot(history.history['val_recall'], label='Validation')
axes[1, 1].set_title('Model Recall', fontsize=12)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("\nTraining history saved as 'training_history.png'")



model.save('xray_disease_classifier_final.keras')
print("\nFinal model saved as 'xray_disease_classifier_final.keras'")


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('xray_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("TFLite model saved as 'xray_model.tflite'")


with open('class_names.txt', 'w') as f:
    f.write('\n'.join(class_names))
print("Class names saved as 'class_names.txt'")



def predict_single_xray(image_path, model, class_names, multi_label=True, threshold=0.5):
    """
    Predict disease from a single X-ray image
    """
    
    img = Image.open(image_path).convert('L')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    
    
    predictions = model.predict(img_array, verbose=0)[0]
    
    print(f"\n{'='*60}")
    print(f"X-RAY PREDICTION: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    if multi_label:
        
        print(f"\nDetected diseases (threshold={threshold}):")
        detected = []
        for i, class_name in enumerate(class_names):
            prob = predictions[i]
            if prob >= threshold:
                detected.append((class_name, prob))
                print(f"  ✓ {class_name}: {prob:.2%}")
        
        if not detected:
            print("  No diseases detected above threshold")
        
        print("\nAll probabilities:")
        sorted_indices = np.argsort(predictions)[::-1]
        for idx in sorted_indices:
            prob = predictions[idx]
            bar = '█' * int(prob * 40)
            print(f"{class_names[idx]:20s} {prob:6.2%} {bar}")
        
        return detected, predictions
    else:
        
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        
        print(f"\nDiagnosis: {class_names[predicted_idx]}")
        print(f"Confidence: {confidence:.2%}\n")
        print("All probabilities:")
        sorted_indices = np.argsort(predictions)[::-1]
        for idx in sorted_indices:
            prob = predictions[idx]
            bar = '█' * int(prob * 40)
            print(f"{class_names[idx]:20s} {prob:6.2%} {bar}")
        
        return class_names[predicted_idx], confidence, predictions



print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
print("\nTo make predictions on new X-ray images, use:")
print("predict_single_xray('path/to/xray.png', model, class_names, multi_label=MULTI_LABEL)")
print("\nExample:")
print("predict_single_xray('00000001_000.png', model, class_names, multi_label=True)")