import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


DATA_DIR = 'eyes_dataset'  
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2


EYE_CONDITIONS = [
    'cataract',
    'diabetes',
    'glaucoma',
    'normal'
]


print("Loading Eye Disease Dataset...")
print(f"Expected folder structure: {DATA_DIR}/[{', '.join(EYE_CONDITIONS)}]/")


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


print("Class Distribution Analysis:")
class_counts = []
for class_name in class_names:
    class_dir = os.path.join(DATA_DIR, class_name)
    if os.path.exists(class_dir):
        count = len([f for f in os.listdir(class_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        class_counts.append(count)

max_count = max(class_counts)
min_count = min(class_counts)
imbalance_ratio = max_count / min_count if min_count > 0 else 0

print(f"  Most samples: {max_count}")
print(f"  Least samples: {min_count}")
print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")

if imbalance_ratio > 3:
    print("  ⚠ Warning: Significant class imbalance detected!")
    print("  Consider using class weights or data augmentation.")
print()





data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),       
    layers.RandomRotation(0.1),            
    layers.RandomZoom(0.1),                
    layers.RandomTranslation(0.05, 0.05),  
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
        plt.title(f"{class_names[class_idx]}", fontsize=12, fontweight='bold')
        plt.axis('off')

plt.suptitle('Sample Eye Images from Training Set', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('sample_eye_images.png', dpi=300, bbox_inches='tight')
print("✓ Sample images saved as 'sample_eye_images.png'\n")



def build_custom_cnn(num_classes):
    """
    Custom CNN architecture for eye disease classification
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
        layers.Dropout(0.3),
        
        
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
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ], name='CustomCNN_Eye')
    
    return model

def build_densenet_model(num_classes):
    """
    Transfer learning model using DenseNet121
    (Excellent for retinal images - used in medical literature)
    """
    
    base_model = keras.applications.DenseNet121(
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
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name='DenseNet121_Transfer')
    
    return model, base_model

def build_inception_model(num_classes):
    """
    Transfer learning model using InceptionV3
    (Great for multi-scale features in retinal images)
    """
    base_model = keras.applications.InceptionV3(
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
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name='InceptionV3_Transfer')
    
    return model, base_model

def build_efficientnet_model(num_classes):
    """
    Transfer learning model using EfficientNetB4
    (State-of-the-art efficiency and accuracy)
    """
    base_model = keras.applications.EfficientNetB4(
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
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name='EfficientNetB4_Transfer')
    
    return model, base_model



print("Building model...")






model, base_model = build_densenet_model(num_classes)







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
        keras.metrics.TopKCategoricalAccuracy(k=min(3, num_classes), name='top_k_accuracy')
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
        'best_eye_disease_model.keras',
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
    
    
    if 'DenseNet' in model.name:
        fine_tune_at = 250  
    elif 'Inception' in model.name:
        fine_tune_at = 200
    elif 'EfficientNet' in model.name:
        fine_tune_at = 300
    else:
        fine_tune_at = 150
    
    
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
            keras.metrics.TopKCategoricalAccuracy(k=min(3, num_classes), name='top_k_accuracy')
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
print(f"  Loss:              {results[0]:.4f}")
print(f"  Accuracy:          {results[1]:.4f}")
print(f"  Precision:         {results[2]:.4f}")
print(f"  Recall:            {results[3]:.4f}")
print(f"  AUC:               {results[4]:.4f}")
if num_classes >= 3:
    print(f"  Top-{min(3, num_classes)} Accuracy: {results[5]:.4f}")


if results[2] + results[3] > 0:
    f1_score = 2 * (results[2] * results[3]) / (results[2] + results[3])
    print(f"  F1-Score:          {f1_score:.4f}")



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
print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))


print("\nPer-Class Accuracy:")
print("="*60)
for i, class_name in enumerate(class_names):
    class_mask = (y_true == i)
    if class_mask.sum() > 0:
        class_acc = (y_pred[class_mask] == i).sum() / class_mask.sum()
        class_support = class_mask.sum()
        print(f"  {class_name:20s}: {class_acc:.4f} ({class_support:3d} samples)")


print("\nSensitivity & Specificity per Class:")
print("="*60)
for i, class_name in enumerate(class_names):
    
    tp = np.sum((y_true == i) & (y_pred == i))
    fn = np.sum((y_true == i) & (y_pred != i))
    tn = np.sum((y_true != i) & (y_pred != i))
    fp = np.sum((y_true != i) & (y_pred == i))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"  {class_name:20s}: Sensitivity={sensitivity:.4f}, Specificity={specificity:.4f}")


print("\n⚠️  Critical Misclassifications for Screening:")
print("="*60)


if 'normal' in class_names:
    normal_idx = class_names.index('normal')
    
    for i, class_name in enumerate(class_names):
        if i != normal_idx:
            disease_as_normal = np.sum((y_true == i) & (y_pred == normal_idx))
            print(f"  {class_name:20s} → normal: {disease_as_normal:3d} times")



print("\nGenerating confusion matrix...")

cm = confusion_matrix(y_true, y_pred)


cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))


sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'},
            ax=ax1,
            annot_kws={'size': 14})
ax1.set_title('Confusion Matrix (Counts)', fontsize=14, pad=15)
ax1.set_ylabel('True Label', fontsize=12)
ax1.set_xlabel('Predicted Label', fontsize=12)
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
plt.setp(ax1.get_yticklabels(), rotation=0)


sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Percentage'},
            ax=ax2,
            annot_kws={'size': 14})
ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, pad=15)
ax2.set_ylabel('True Label', fontsize=12)
ax2.set_xlabel('Predicted Label', fontsize=12)
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
plt.setp(ax2.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrix saved as 'confusion_matrix.png'")



print("\nGenerating training history plots...")


metrics_to_plot = ['accuracy', 'loss', 'precision', 'recall', 'auc']
if num_classes >= 3:
    metrics_to_plot.append('top_k_accuracy')

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

if num_misclassified > 0:
    
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

model.save('eye_disease_classifier_final.keras')
print("✓ Final model saved as 'eye_disease_classifier_final.keras'")


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('eye_disease_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("✓ TFLite model saved as 'eye_disease_model.tflite'")


with open('class_names.txt', 'w') as f:
    f.write('\n'.join(class_names))
print("✓ Class names saved as 'class_names.txt'")


import json
with open('training_history.json', 'w') as f:
    json.dump(history.history, f, indent=2)
print("✓ Training history saved as 'training_history.json'")


with open('evaluation_report.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("EYE DISEASE CLASSIFIER - EVALUATION REPORT\n")
    f.write("="*60 + "\n\n")
    f.write(f"Model: {model.name}\n")
    f.write(f"Classes: {', '.join(class_names)}\n")
    f.write(f"Training samples: ~{approx_train_images}\n")
    f.write(f"Validation samples: ~{approx_val_images}\n\n")
    f.write("Overall Metrics:\n")
    f.write(f"  Accuracy:  {results[1]:.4f}\n")
    f.write(f"  Precision: {results[2]:.4f}\n")
    f.write(f"  Recall:    {results[3]:.4f}\n")
    f.write(f"  F1-Score:  {f1_score:.4f}\n\n")
    f.write(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))
    
    
    f.write("\n" + "="*60 + "\n")
    f.write("CRITICAL SCREENING METRICS\n")
    f.write("="*60 + "\n\n")
    if 'normal' in class_names:
        normal_idx = class_names.index('normal')
        f.write("Disease predicted as normal (False Negatives):\n")
        for i, class_name in enumerate(class_names):
            if i != normal_idx:
                disease_as_normal = np.sum((y_true == i) & (y_pred == normal_idx))
                f.write(f"  {class_name:20s} → normal: {disease_as_normal:3d} times\n")

print("✓ Evaluation report saved as 'evaluation_report.txt'")



def predict_single_eye(image_path, model, class_names):
    """
    Predict eye condition from a single fundus/eye image
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
    
    
    num_top = min(len(class_names), len(class_names))
    top_idx = np.argsort(predictions[0])[-num_top:][::-1]
    
    
    print(f"\n{'='*60}")
    print(f"EYE CONDITION PREDICTION: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    print(f"\nDiagnosis: {class_names[predicted_idx].upper()}")
    print(f"Confidence: {confidence:.2%}\n")
    
    
    if predicted_idx != class_names.index('normal') if 'normal' in class_names else -1:
        disease_name = class_names[predicted_idx]
        if disease_name == 'glaucoma':
            print("⚠️  GLAUCOMA DETECTED - Progressive vision loss, requires immediate treatment!")
        elif disease_name == 'diabetes':
            print("⚠️  DIABETIC RETINOPATHY DETECTED - Vision-threatening complication!")
        elif disease_name == 'cataract':
            print("⚠️  CATARACT DETECTED - Clouding of lens, may require surgery!")
        print("    → Ophthalmologist consultation recommended\n")
    
    print(f"All Predictions:")
    for idx in top_idx:
        prob = predictions[0][idx]
        bar = '█' * int(prob * 40)
        status = '✓' if idx == predicted_idx else ' '
        print(f"{status} {class_names[idx]:20s} {prob:6.2%} {bar}")
    
    return class_names[predicted_idx], confidence, predictions[0]

def predict_batch_eyes(image_paths, model, class_names):
    """
    Predict multiple eye images
    """
    results = []
    for img_path in image_paths:
        condition, conf, probs = predict_single_eye(img_path, model, class_names)
        results.append({
            'image': img_path,
            'predicted_condition': condition,
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
print("  ✓ eye_disease_classifier_final.keras  - Trained model")
print("  ✓ eye_disease_model.tflite            - Mobile model")
print("  ✓ best_eye_disease_model.keras        - Best checkpoint")
print("  ✓ class_names.txt                     - Condition labels")
print("  ✓ confusion_matrix.png                - Confusion matrices")
print("  ✓ training_history.png                - Training curves")
print("  ✓ training_history.json               - History data")
print("  ✓ training_log.csv                    - Training log")
print("  ✓ sample_eye_images.png               - Sample images")
print("  ✓ evaluation_report.txt               - Detailed metrics")

print("\n" + "="*60)
print("NEXT STEPS")
print("="*60)
print("1. Review confusion matrix - especially disease → normal")
print("2. Test on fundus camera images from real clinical settings")
print("3. Validate with ophthalmologist ground truth")
print("4. Consider ensemble models for critical screening")
print("5. Deploy as a screening tool (NOT diagnostic)")

print("\n" + "="*60)
print("SCREENING REMINDER")
print("="*60)
print("⚠️  This model is for SCREENING purposes only!")
print("   - Glaucoma: Early detection prevents blindness")
print("   - Diabetic Retinopathy: Leading cause of blindness in adults")
print("   - Cataract: Common but treatable with surgery")
print("   ALL positive screenings require ophthalmologist confirmation!")
print("="*60 + "\n")
