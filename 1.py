import cv2
import numpy as np
from ultralytics import YOLO

# Configuración
MODEL_PATH = 'yolov8n.pt'  # Modelo YOLOv8 pre-entrenado

# Clases de animales en el modelo COCO (YOLOv8)
ANIMAL_CLASSES = {
    16: 'pájaro', 17: 'gato', 18: 'perro', 19: 'caballo', 20: 'oveja',
    21: 'vaca', 22: 'elefante', 23: 'oso', 24: 'cebra', 25: 'jirafa',
    # Agrega más clases animales según sea necesario
}

# Traducciones personalizadas (extiende según tus necesidades)
SPANISH_LABELS = {
    'bird': 'pájaro', 'cat': 'gato', 'dog': 'perro', 'horse': 'caballo',
    'sheep': 'oveja', 'cow': 'vaca', 'elephant': 'elefante', 'bear': 'oso',
    'zebra': 'cebra', 'giraffe': 'jirafa', 'lion': 'león', 'tiger': 'tigre'
}

def cargar_modelo():
    """Carga el modelo YOLO y configura las clases animales"""
    model = YOLO(MODEL_PATH)
    
    # Filtrar solo clases de animales
    model.classes = list(ANIMAL_CLASSES.keys())
    
    return model

def dibujar_deteccion(frame, box, label, confianza):
    """Dibuja un cuadro de detección profesional"""
    x1, y1, x2, y2 = map(int, box)
    
    # Marco de detección
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Etiqueta con fondo
    texto = f"{label} {confianza:.1f}%"
    (w, h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
    cv2.putText(frame, texto, (x1, y1 - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

def detectar_animales():
    model = cargar_modelo()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("🔍 Sistema de Detección de Animales Profesional")
    print("Presiona 'q' para salir")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detección con YOLOv8 (solo animales)
        results = model(frame, verbose=False)
        
        for r in results:
            for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                if conf < 0.5:  # Filtro de confianza
                    continue
                
                # Obtener nombre de la clase
                class_name = model.names[int(cls)]
                label = SPANISH_LABELS.get(class_name, class_name)
                
                # Dibujar detección
                dibujar_deteccion(frame, box, label, conf * 100)
        
        cv2.imshow('Detección Profesional de Animales', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Verificar instalaciones
    try:
        detectar_animales()
    except Exception as e:
        print(f"Error: {e}")
        print("\n⚠️ Soluciona los problemas de instalación:")
        print("1. Ejecuta en Anaconda Prompt:")
        print("pip uninstall torch torchvision ultralytics -y")
        print("pip cache purge")
        print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("pip install ultralytics")