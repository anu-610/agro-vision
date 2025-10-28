import tensorflow as tf
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
import os

# --- 1. SETUP AND DATA LOADING ---

# The path to your dataset on Kaggle
data_dir = 'plantdisease/PlantVillage/'  

# Verify the path exists
if not os.path.exists(data_dir):
    print(f"Error: The directory '{data_dir}' does not exist. Please read the text file in this directory and do the steps in that and run this file")
else:
    print(f"Success: Found dataset at '{data_dir}'")

# Model parameters
IMG_SIZE = 224
BATCH_SIZE = 64 # Increased batch size for powerful Kaggle GPU

# Load the dataset
print("Loading datasets...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# Get class names
class_names = train_dataset.class_names
num_classes = len(class_names)
print(f"Found {num_classes} classes.")

# Configure for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)


# --- 2. DATA AUGMENTATION ---
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
])


# --- 3. MODEL BUILDING (TRANSFER LEARNING) ---
print("Building the model...")
base_model = tf.keras.applications.EfficientNetV2S(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# Build the final model
inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)


# --- 4. INITIAL TRAINING ---
print("\n--- Starting Initial Training (Training the Head) ---")
initial_epochs = 10
history = model.fit(
    train_dataset,
    epochs=initial_epochs,
    validation_data=validation_dataset,
    verbose=2 # Use verbose=2 for cleaner logs in background runs
)


# --- 5. FINE-TUNING FOR MAXIMUM ACCURACY ---
print("\n--- Starting Fine-Tuning ---")
base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # Lower learning rate
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_dataset,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1], # Start from where we left off
    validation_data=validation_dataset,
    verbose=2
)


# --- 6. SAVE THE FINAL MODEL ---
# The model will be saved in the output directory: /kaggle/working/
model.save('fasal_rakshak_model.h5')
print("\nTraining complete. Model saved to /kaggle/working/fasal_rakshak_model.h5")