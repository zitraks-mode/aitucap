import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


DATA_DIR = 'face_dataset'  
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2


DISEASE_FOLDERS = [
    'Carcinoma',
    'Dermatitis',
    'Eczema',
    'Fungi',
    'Keratoses',
    'Keratosis',
    'Nevi',
    'Melanoma',
    'Psoriasis',
    'Warts'
]


print("Loading Skin Disease Dataset...")
print(f"Expected folder structure: {DATA_DIR}/[{', '.join(DISEASE_FOLDERS)}]/")


train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'  
)


val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)


class_names = train_ds.class_names
num_classes = len(class_names)

print(f"\n{'='*60}")
print(f"DATASET INFORMATION")
print(f"{'='*60}")
print(f"Classes detected: {class_names}")
print(f"Number of classes: {num_classes}")


print(f"\nDataset structure:")
import os
for class_name in class_names:
    class_dir = os.path.join(DATA_DIR, class_name)
    if os.path.exists(class_dir):
        num_images = len([f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        print(f"  {class_name:20s}: {num_images:4d} images")


total_batches = tf.data.experimental.cardinality(train_ds).numpy()
val_batches = tf.data.experimental.cardinality(val_ds).numpy()
approx_train_images = total_batches * BATCH_SIZE
approx_val_images = val_batches * BATCH_SIZE

print(f"\nApproximate split:")
print(f"  Training images:   ~{approx_train_images}")
print(f"  Validation images: ~{approx_val_images}")
print(f"{'='*60}\n")




data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),  
    layers.RandomRotation(0.2),  
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
], name='data_augmentation')


normalization_layer = layers.Rescaling(1./255)


def prepare_dataset(ds, augment=False):
    """Prepare dataset with normalization and optional augmentation"""
    ds = ds.map(lambda x, y: (normalization_layer(x), y))
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y))
    return ds

train_ds = prepare_dataset(train_ds, augment=True)
val_ds = prepare_dataset(val_ds, augment=False)


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



print("Generating sample images visualization...")

plt.figure(figsize=(15, 10))
for images, labels in train_ds.take(1):
    for i in range(min(9, BATCH_SIZE)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy())
        class_idx = np.argmax(labels[i])
        plt.title(f"{class_names[class_idx]}")
        plt.axis('off')

plt.tight_layout()
plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
print("✓ Sample images saved as 'sample_images.png'\n")



def build_custom_cnn(num_classes):
    """
    Custom CNN architecture for skin disease classification
    """
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        
        
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
        layers.Dense(num_classes, activation='softmax')
    ], name='CustomCNN')
    
    return model

def build_transfer_learning_model(num_classes):
    """
    Transfer learning model using EfficientNetB3
    (Works well for skin lesion classification)
    """
    
    base_model = keras.applications.EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        pooling='avg'
    )
    
    
    base_model.trainable = False
    
    
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name='EfficientNetB3_Transfer')
    
    return model, base_model

def build_mobilenet_model(num_classes):
    """
    Lightweight model using MobileNetV2
    (Good balance between speed and accuracy)
    """
    base_model = keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        pooling='avg'
    )
    
    base_model.trainable = False
    
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name='MobileNetV2_Transfer')
    
    return model, base_model



print("Building model...")






model, base_model = build_transfer_learning_model(num_classes)




model.summary()

print(f"\n✓ Model built: {model.name}")
print(f"  Total parameters: {model.count_params():,}")



model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
    ]
)


callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'best_skin_disease_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1
    ),
    keras.callbacks.CSVLogger(
        'training_log.csv',
        separator=',',
        append=False
    )
]



print("\n" + "="*60)
print("PHASE 1: INITIAL TRAINING")
print("="*60)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)




if 'base_model' in locals():
    print("\n" + "="*60)
    print("PHASE 2: FINE-TUNING")
    print("="*60)
    
    
    base_model.trainable = True
    
    
    if 'EfficientNet' in model.name:
        fine_tune_at = 200  
    else:
        fine_tune_at = 100
    
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    print(f"Fine-tuning from layer {fine_tune_at} onwards")
    print(f"Trainable layers: {sum([1 for layer in model.layers if layer.trainable])}")
    
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
        ]
    )
    
    
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS + 20,
        initial_epoch=len(history.history['loss']),
        callbacks=callbacks,
        verbose=1
    )
    
    
    for key in history.history.keys():
        history.history[key].extend(history_fine.history[key])



print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)


results = model.evaluate(val_ds, verbose=0)

print(f"\nValidation Metrics:")
print(f"  Loss:           {results[0]:.4f}")
print(f"  Accuracy:       {results[1]:.4f}")
print(f"  Precision:      {results[2]:.4f}")
print(f"  Recall:         {results[3]:.4f}")
print(f"  AUC:            {results[4]:.4f}")
print(f"  Top-3 Accuracy: {results[5]:.4f}")


if results[2] + results[3] > 0:
    f1_score = 2 * (results[2] * results[3]) / (results[2] + results[3])
    print(f"  F1-Score:       {f1_score:.4f}")



print("\n" + "="*60)
print("GENERATING DETAILED METRICS")
print("="*60)


y_true = []
y_pred = []
y_pred_proba = []

for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred_proba.extend(predictions)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_pred_proba = np.array(y_pred_proba)


print("\nClassification Report:")
print("="*60)
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


print("\nPer-Class Accuracy:")
print("="*60)
for i, class_name in enumerate(class_names):
    class_mask = (y_true == i)
    if class_mask.sum() > 0:
        class_acc = (y_pred[class_mask] == i).sum() / class_mask.sum()
        print(f"  {class_name:20s}: {class_acc:.4f} ({class_mask.sum()} samples)")



print("\nGenerating confusion matrix...")

cm = confusion_matrix(y_true, y_pred)


cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))


sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'},
            ax=ax1)
ax1.set_title('Confusion Matrix (Counts)', fontsize=14, pad=15)
ax1.set_ylabel('True Label', fontsize=12)
ax1.set_xlabel('Predicted Label', fontsize=12)
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
plt.setp(ax1.get_yticklabels(), rotation=0)


sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Percentage'},
            ax=ax2)
ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, pad=15)
ax2.set_ylabel('True Label', fontsize=12)
ax2.set_xlabel('Predicted Label', fontsize=12)
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
plt.setp(ax2.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrix saved as 'confusion_matrix.png'")



print("\nGenerating training history plots...")


metrics_to_plot = ['accuracy', 'loss', 'precision', 'recall', 'auc', 'top_3_accuracy']
available_metrics = [m for m in metrics_to_plot if m in history.history]

num_metrics = len(available_metrics)
cols = 3
rows = (num_metrics + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
axes = axes.flatten() if num_metrics > 1 else [axes]

for idx, metric in enumerate(available_metrics):
    ax = axes[idx]
    ax.plot(history.history[metric], label='Train', linewidth=2)
    ax.plot(history.history[f'val_{metric}'], label='Validation', linewidth=2)
    ax.set_title(f'Model {metric.replace("_", " ").title()}', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.legend()
    ax.grid(True, alpha=0.3)


for idx in range(num_metrics, len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("✓ Training history saved as 'training_history.png'")



print("\nAnalyzing misclassifications...")


misclassified_indices = np.where(y_true != y_pred)[0]
num_misclassified = len(misclassified_indices)

print(f"\nMisclassification Summary:")
print(f"  Total misclassified: {num_misclassified} / {len(y_true)} ({num_misclassified/len(y_true)*100:.2f}%)")


print("\nMost Common Misclassifications:")
misclassification_pairs = {}
for idx in misclassified_indices:
    true_class = class_names[y_true[idx]]
    pred_class = class_names[y_pred[idx]]
    pair = (true_class, pred_class)
    misclassification_pairs[pair] = misclassification_pairs.get(pair, 0) + 1


sorted_pairs = sorted(misclassification_pairs.items(), key=lambda x: x[1], reverse=True)
for (true_class, pred_class), count in sorted_pairs[:10]:
    print(f"  {true_class:20s} → {pred_class:20s}: {count:3d} times")



print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

model.save('skin_disease_classifier_final.keras')
print("✓ Final model saved as 'skin_disease_classifier_final.keras'")


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('skin_disease_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("✓ TFLite model saved as 'skin_disease_model.tflite'")


with open('class_names.txt', 'w') as f:
    f.write('\n'.join(class_names))
print("✓ Class names saved as 'class_names.txt'")


import json
with open('training_history.json', 'w') as f:
    json.dump(history.history, f, indent=2)
print("✓ Training history saved as 'training_history.json'")



def predict_single_image(image_path, model, class_names):
    """
    Predict skin disease from a single image
    """
    
    img = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)
    
    
    predictions = model.predict(img_array, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    
    
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    
    
    print(f"\n{'='*60}")
    print(f"SKIN DISEASE PREDICTION: {image_path}")
    print(f"{'='*60}")
    print(f"\nPrimary Diagnosis: {class_names[predicted_idx]}")
    print(f"Confidence: {confidence:.2%}\n")
    
    print("Top 3 Predictions:")
    for i, idx in enumerate(top_3_idx, 1):
        prob = predictions[0][idx]
        bar = '█' * int(prob * 40)
        print(f"{i}. {class_names[idx]:20s} {prob:6.2%} {bar}")
    
    print("\nAll Probabilities:")
    sorted_indices = np.argsort(predictions[0])[::-1]
    for idx in sorted_indices:
        prob = predictions[0][idx]
        bar = '█' * int(prob * 40)
        print(f"   {class_names[idx]:20s} {prob:6.2%} {bar}")
    
    return class_names[predicted_idx], confidence, predictions[0]

def predict_batch(image_paths, model, class_names):
    """
    Predict multiple skin disease images
    """
    results = []
    for img_path in image_paths:
        disease, conf, probs = predict_single_image(img_path, model, class_names)
        results.append({
            'image': img_path,
            'predicted_disease': disease,
            'confidence': conf,
            'probabilities': probs
        })
    return results



print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\nFinal Validation Accuracy: {results[1]:.2%}")
print(f"Final Validation Loss: {results[0]:.4f}")

print("\n" + "="*60)
print("FILES GENERATED")
print("="*60)
print("  ✓ skin_disease_classifier_final.keras  - Trained model")
print("  ✓ skin_disease_model.tflite            - Mobile model")
print("  ✓ best_skin_disease_model.keras        - Best checkpoint")
print("  ✓ class_names.txt                      - Class labels")
print("  ✓ confusion_matrix.png                 - Confusion matrix")
print("  ✓ training_history.png                 - Training curves")
print("  ✓ training_history.json                - History data")
print("  ✓ training_log.csv                     - Training log")
print("  ✓ sample_images.png                    - Sample images")

print("\n" + "="*60)
print("NEXT STEPS")
print("="*60)
print("1. Review the confusion matrix to identify problematic classes")
print("2. Use the prediction tools to test on new images:")
print("   predict_single_image('path/to/image.jpg', model, class_names)")
print("3. Deploy the model or continue fine-tuning")

print("\n" + "="*60)
