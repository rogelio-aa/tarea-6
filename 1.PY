import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model

# Global configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
MODEL_PATH = 'animal_classifier_final.h5'
DATA_DIR = r"C:\Users\ESTUDIANTE\Documents\animales deteccion\animals\animals"

# Spanish translation dictionary
SPANISH_LABELS = {
    'antelope': 'antílope',
    'badger': 'tejón',
    'bat': 'murciélago',
    'bear': 'oso',
    'bee': 'abeja',
    'beetle': 'escarabajo',
    'bison': 'bisonte',
    'boar': 'jabalí',
    'butterfly': 'mariposa',
    'cat': 'gato',
    'caterpillar': 'oruga',
    'chimpanzee': 'chimpancé',
    'cockroach': 'cucaracha',
    'cow': 'vaca',
    'coyote': 'coyote',
    'crab': 'cangrejo',
    'crow': 'cuervo',
    'deer': 'ciervo',
    'dog': 'perro',
    'dolphin': 'delfín',
    'donkey': 'burro',
    'dragonfly': 'libélula',
    'duck': 'pato',
    'eagle': 'águila',
    'elephant': 'elefante',
    'flamingo': 'flamenco',
    'fly': 'mosca',
    'fox': 'zorro',
    'goat': 'cabra',
    'goldfish': 'pez dorado',
    'goose': 'ganso',
    'gorilla': 'gorila',
    'grasshopper': 'saltamontes',
    'hamster': 'hámster',
    'hare': 'liebre',
    'hedgehog': 'erizo',
    'hippopotamus': 'hipopótamo',
    'hornbill': 'cálao',
    'horse': 'caballo',
    'hummingbird': 'colibrí',
    'hyena': 'hiena',
    'jellyfish': 'medusa',
    'kangaroo': 'canguro',
    'koala': 'koala',
    'ladybugs': 'mariquita',
    'leopard': 'leopardo',
    'lion': 'león',
    'lizard': 'lagarto',
    'lobster': 'langosta',
    'mosquito': 'mosquito',
    'moth': 'polilla',
    'mouse': 'ratón',
    'octopus': 'pulpo',
    'okapi': 'okapi',
    'orangutan': 'orangután',
    'otter': 'nutria',
    'owl': 'búho',
    'ox': 'buey',
    'oyster': 'ostra',
    'panda': 'panda',
    'parrot': 'loro',
    'pelecaniformes': 'pelícano',
    'penguin': 'pingüino',
    'pig': 'cerdo',
    'pigeon': 'paloma',
    'porcupine': 'puercoespín',
    'possum': 'zarigüeya',
    'raccoon': 'mapache',
    'rat': 'rata',
    'reindeer': 'reno',
    'rhinoceros': 'rinoceronte',
    'sandpiper': 'andarríos',
    'seahorse': 'caballito de mar',
    'seal': 'foca',
    'shark': 'tiburón',
    'sheep': 'oveja',
    'snake': 'serpiente',
    'sparrow': 'gorrión',
    'squid': 'calamar',
    'squirrel': 'ardilla',
    'starfish': 'estrella de mar',
    'swan': 'cisne',
    'tiger': 'tigre',
    'turkey': 'pavo',
    'turtle': 'tortuga',
    'whale': 'ballena',
    'wolf': 'lobo',
    'wombat': 'wombat',
    'woodpecker': 'pájaro carpintero',
    'zebra': 'cebra'
}

def train_model():
    """Function to train the model from scratch"""
    # Verify available classes
    classes = os.listdir(DATA_DIR)
    NUM_CLASSES = len(classes)
    print(f"Detected classes: {classes}")

    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode='reflect',
        validation_split=0.2
    )

    # Data generators
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    val_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Save class indices for later use
    global class_indices
    class_indices = train_generator.class_indices
    np.save('class_indices.npy', class_indices)

    # Base model
    base_model = EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False

    # Model construction
    model = keras.Sequential([
        base_model,
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Compilation and training
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Training model...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Fine-tuning model...")
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        epochs=10,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    model.save(MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")
    
    return model, class_indices

def load_class_indices():
    """Load class indices from file"""
    if os.path.exists('class_indices.npy'):
        return np.load('class_indices.npy', allow_pickle=True).item()
    return None

def real_time_detection(model):
    """Function for real-time detection with webcam"""
    # Get class labels
    class_indices = load_class_indices()
    
    if class_indices is None:
        print("Warning: Class indices not found. Using default labels.")
        class_indices = {v: k for k, v in SPANISH_LABELS.items()}
    
    # Reverse the dictionary for lookup
    idx_to_class = {v: k for k, v in class_indices.items()}
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("\nInstructions:")
    print("1. Press 'q' to quit")
    print("2. Press 's' to save the current image")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture frame")
            break

        # Create a copy of the frame for display
        display_frame = frame.copy()
        
        # Preprocessing for the model
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMG_SIZE)
        img_normalized = img_resized / 255.0
        img_input = np.expand_dims(img_normalized, axis=0)

        # Prediction
        preds = model.predict(img_input, verbose=0)
        class_idx = np.argmax(preds[0])
        confidence = preds[0][class_idx] * 100
        
        # Get English and Spanish labels
        english_label = idx_to_class.get(class_idx, "unknown")
        spanish_label = SPANISH_LABELS.get(english_label, english_label)
        
        # Create a nice display box
        box_color = (0, 255, 0)  # Green
        text_color = (255, 255, 255)  # White
        
        # Draw a rectangle around the entire frame
        cv2.rectangle(display_frame, (10, 10), (630, 90), (0, 0, 0), -1)  # Black background
        cv2.rectangle(display_frame, (10, 10), (630, 90), box_color, 2)  # Border
        
        # Put the text with the prediction
        label = f"Animal: {spanish_label}"
        confidence_text = f"Confianza: {confidence:.1f}%"
        
        cv2.putText(display_frame, label, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        cv2.putText(display_frame, confidence_text, (20, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # Show result on frame
        cv2.imshow('Animal Detection - Presione Q para salir', display_frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_path = f"captura_{spanish_label}_{confidence:.0f}%.jpg"
            cv2.imwrite(save_path, display_frame)
            print(f"Imagen guardada como {save_path}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Check if model is already trained
    if os.path.exists(MODEL_PATH):
        print("Loading pre-trained model...")
        model = load_model(MODEL_PATH)
        class_indices = load_class_indices()
    else:
        print("Training new model...")
        model, class_indices = train_model()
    
    # Start real-time detection
    print("\nStarting real-time detection...")
    real_time_detection(model)