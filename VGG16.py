# ============================================================
# 🧠 VGG16 Classification Brain Tumor Dataset sur CPU
# ============================================================

import os
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ============================================================
# 1️⃣ Configuration CPU
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force l'utilisation du CPU

# ============================================================
# 2️⃣ Extraction du ZIP
# ============================================================
zip_path = r"C:\Users\maram\PycharmProjects\Datamining\archive (2).zip"
extract_path = r"C:\Users\maram\PycharmProjects\Datamining\brain_dataset"

os.makedirs(extract_path, exist_ok=True)

if zipfile.is_zipfile(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"✅ Extraction terminée ! Contenu de {extract_path} :")
    print(os.listdir(extract_path))
else:
    print("❌ Le fichier ZIP est invalide ou corrompu")
    exit()

# ============================================================
# 3️⃣ Chemins vers les données
# ============================================================
train_dir = os.path.join(extract_path, "Training")
test_dir  = os.path.join(extract_path, "Testing")

# ============================================================
# 4️⃣ Préparation des données
# ============================================================
img_size = (224, 224)
batch_size = 16

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# ============================================================
# 5️⃣ Construction du modèle VGG16
# ============================================================
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ============================================================
# 6️⃣ Compilation du modèle
# ============================================================
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ============================================================
# 7️⃣ Callbacks
# ============================================================
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('vgg16_brain_tumor.h5', monitor='val_loss', save_best_only=True)

# ============================================================
# 8️⃣ Entraînement
# ============================================================
epochs = 50

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=epochs,
    callbacks=[early_stop, checkpoint]
)

# ============================================================
# 🔟 Sauvegarde des courbes Accuracy et Loss
# ============================================================
import matplotlib
matplotlib.use('Agg')  # backend non interactif
import matplotlib.pyplot as plt
import pickle

# Courbe Accuracy
plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Courbe Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("accuracy_curve.png", dpi=300)
plt.close()
print("✅ Courbe Accuracy sauvegardée sous 'accuracy_curve.png'")

# Courbe Loss
plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Courbe Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png", dpi=300)
plt.close()
print("✅ Courbe Loss sauvegardée sous 'loss_curve.png'")

# Sauvegarde du history pour usage futur
with open("history.pkl", "wb") as f:
    pickle.dump(history.history, f)
print("✅ History sauvegardé sous 'history.pkl'")

# ============================================================
# 9️⃣ Évaluation sur test set
# ============================================================
loss, acc = model.evaluate(test_generator)
print(f"\nTest Accuracy: {acc*100:.2f}%")
