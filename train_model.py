"""train_model.py
Trains a CNN on the prepared dataset.
Usage:
    python train_model.py --data_dir dataset --model_out models/fruits_veg_cnn.h5
"""
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks

def build_model(input_shape=(100,100,3), num_classes=20):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main(data_dir='dataset', model_out='models/fruits_veg_cnn.h5', epochs=20, batch_size=32):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found. Run dataset_script.py first.")

    img_size = (100,100)
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15,
                                       rotation_range=15, width_shift_range=0.1,
                                       height_shift_range=0.1, horizontal_flip=True)

    train_gen = train_datagen.flow_from_directory(
        data_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='training'
    )
    val_gen = train_datagen.flow_from_directory(
        data_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='validation'
    )

    num_classes = len(train_gen.class_indices)
    model = build_model(input_shape=img_size+(3,), num_classes=num_classes)
    model.summary()

    out = Path('outputs')
    out.mkdir(parents=True, exist_ok=True)
    model_dir = Path(model_out).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    cb = [
        callbacks.ModelCheckpoint(model_out, save_best_only=True, monitor='val_accuracy', mode='max'),
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]

    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=cb)

    # save training plots
    plt.figure(figsize=(8,4))
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(out / 'accuracy.png')
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(out / 'loss.png')
    plt.close()

    model.save(model_out)
    print('Model saved to', model_out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset')
    parser.add_argument('--model_out', type=str, default='models/fruits_veg_cnn.h5')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    main(data_dir=args.data_dir, model_out=args.model_out, epochs=args.epochs, batch_size=args.batch_size)
