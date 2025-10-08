import cv2
import mediapipe as mp
import numpy as np
import os
import pytesseract
import re 
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

# =========================================================================
# --- CONFIGURACIÓN DE PARÁMETROS GLOBALES/CONSTANTES ---
# =========================================================================

# COORDENADAS OCR Y VALORES DE RESPALDO
ROI_TMAX_TEXTO = [568, 7, 628, 40]
ROI_TMIN_TEXTO = [561, 439, 634, 474] 
DEFAULT_T_MAX = 37.0
DEFAULT_T_MIN = 25.0

# MODELOS DNN (DEBEN ESTAR EN EL MISMO DIRECTORIO DEL ARCHIVO main_processor.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROTOTXT_PATH = os.path.join(SCRIPT_DIR, 'deploy.prototxt.txt')
CAFFEMODEL_PATH = os.path.join(SCRIPT_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
FACE_MESH = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.4
)

# Definir índices de los puntos para cada zona del rostro
ZONAS = {
    "1. Frente": [9, 336, 296, 334, 293, 301, 251, 284, 332, 297, 338, 10, 109, 67, 103, 54, 21, 71, 63, 105, 66, 107],
    "2. Mejilla Izquierda": [138, 215, 177, 137, 227, 111, 31, 228, 229, 230, 120, 47, 126, 209, 129, 203, 206, 216],
    "3. Mejilla Derecha": [367, 435, 401, 366, 447, 340, 261, 448, 449, 450, 349, 277, 355, 429, 358, 423, 426, 436],
    "4. Nariz": [2, 326, 328, 290, 392, 439, 278, 279, 420, 399, 419, 351, 168, 122, 196, 174, 198, 49, 48, 219, 64, 98, 97],
    "5. Mentón": [17, 314, 405, 321, 375, 287, 432, 434, 364, 394, 395, 369, 396, 175, 171, 140, 170, 169, 135, 214, 212, 57, 61, 146, 91, 181, 84],
}

# Cargar el modelo de detección de rostros de OpenCV DNN
try:
    FACE_DETECTOR = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
    if FACE_DETECTOR.empty():
        raise cv2.error("Modelo DNN cargado pero vacío.")
    print("✅ Módulo de lógica cargado: Modelos DNN listos.")
except cv2.error as e:
    print(f"❌ Error al cargar el modelo DNN: {e}")
    FACE_DETECTOR = None
except Exception:
     print(f"❌ Error: Archivos del modelo DNN no encontrados o corruptos.")
     FACE_DETECTOR = None


# =========================================================================
# --- FUNCIONES AUXILIARES (OCR, CALIBRACIÓN, ANÁLISIS, PDF) ---
# =========================================================================

def obtener_temperaturas_ocr(image_bgr):
    """Intenta leer T_MIN y T_MAX de la imagen usando OCR."""
    def procesar_roi(image, roi):
        x1, y1, x2, y2 = roi
        if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]): return None
        roi_region = image[y1:y2, x1:x2] 
        gray = cv2.cvtColor(roi_region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(thresh, config='--psm 6 -c tessedit_char_whitelist=0123456789.-')
        clean_value = re.sub(r'[^\d\.\-]', '', text).strip()
        try:
            return float(clean_value.replace(',', '.'))
        except ValueError:
            return None
            
    T_MAX_AUTO = procesar_roi(image_bgr, ROI_TMAX_TEXTO)
    T_MIN_AUTO = procesar_roi(image_bgr, ROI_TMIN_TEXTO)
    return T_MIN_AUTO, T_MAX_AUTO

def pixel_a_temperatura(pixel_val, m, b):
    """Convierte un valor de píxel a temperatura: T = m * P + b."""
    return m * pixel_val + b

def calcular_calibracion(T_MAX_AUTO, T_MIN_AUTO):
    """Calcula m y b para la calibración lineal 0-255."""
    T_MAX_val = T_MAX_AUTO if T_MAX_AUTO is not None and T_MAX_AUTO > DEFAULT_T_MIN else DEFAULT_T_MAX
    T_MIN_val = T_MIN_AUTO if T_MIN_AUTO is not None and T_MIN_AUTO < DEFAULT_T_MAX else DEFAULT_T_MIN

    if T_MAX_val <= T_MIN_val:
        T_MAX_val = DEFAULT_T_MAX
        T_MIN_val = DEFAULT_T_MIN
    
    P_HIGH = 255.0
    P_LOW = 0.0
    rango_temp = T_MAX_val - T_MIN_val
    rango_pixel = P_HIGH - P_LOW 
    m = rango_temp / rango_pixel
    b = T_MIN_val
    
    return T_MAX_val, T_MIN_val, m, b

def analizar_zona_termica(imagen_gris, puntos_poly, m, b):
    """Calcula T_max, T_min y T_promedio para una zona poligonal."""
    mascara = np.zeros_like(imagen_gris)
    cv2.fillPoly(mascara, [puntos_poly], 255)
    zona_pixeles_gris = imagen_gris[np.where(mascara == 255)]
    
    if len(zona_pixeles_gris) == 0: return None
    
    temperaturas = pixel_a_temperatura(zona_pixeles_gris, m, b)
    
    return {
        "max": np.max(temperaturas),
        "min": np.min(temperaturas),
        "promedio": np.mean(temperaturas)
    }

def generar_pdf_informe(pdf_filename, image_original_path, image_etiquetada_path, analisis_data, T_MAX_val, T_MIN_val, m, b, imagen_nombre_base):
    """Genera el informe PDF con imágenes y tabla de resultados."""
    doc = SimpleDocTemplate(pdf_filename, pagesize=A4, 
                            leftMargin=2.5*cm, rightMargin=2.5*cm, topMargin=2.5*cm, bottomMargin=2.5*cm)
    styles = getSampleStyleSheet()
    elementos = []

    # 1. Título
    elementos.append(Paragraph(f"<b>INFORME TÉRMICO DE ROSTRO: {imagen_nombre_base}</b>", styles['Title']))
    elementos.append(Spacer(1, 12))
    
    # 2. Calibración
    calibracion_info = f"<b>Calibración Usada:</b> T Máx={T_MAX_val:.1f}°C, T Mín={T_MIN_val:.1f}°C."
    elementos.append(Paragraph(calibracion_info, styles['Normal']))
    elementos.append(Spacer(1, 18))

    # 3. Imágenes
    elementos.append(Paragraph("<b>Imagen de Entrada y Zonas Analizadas</b>", styles['h2']))
    elementos.append(Spacer(1, 6))

    img_original = Image(image_original_path, width=7.5*cm, height=5.6*cm)
    img_etiquetada = Image(image_etiquetada_path, width=7.5*cm, height=5.6*cm)
    
    tabla_imagenes = Table([[img_original, img_etiquetada]], colWidths=[9*cm, 9*cm])
    elementos.append(tabla_imagenes)
    elementos.append(Spacer(1, 18))

    # 4. Tabla de Resultados 
    elementos.append(Paragraph("<b>Resultados Detallados por Zona</b>", styles['h2']))
    elementos.append(Spacer(1, 6))

    datos_tabla = [["Zona", "T. Máxima", "T. Mínima", "T. Promedio"]]
    for nombre, datos in analisis_data.items():
        datos_tabla.append([
            nombre,
            f"{datos['max']:.2f}°C",
            f"{datos['min']:.2f}°C",
            f"{datos['promedio']:.2f}°C"
        ])

    tabla = Table(datos_tabla, colWidths=[4*cm, 3*cm, 3*cm, 3*cm]) 
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2A6F8F')), 
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke), 
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black), 
    ])
    for i in range(len(datos_tabla)):
        if i > 0 and i % 2 == 1:
            style.add('BACKGROUND', (0, i), (-1, i), colors.HexColor('#F0F8FF')) 
        elif i > 0 and i % 2 == 0:
            style.add('BACKGROUND', (0, i), (-1, i), colors.HexColor('#FFFFFF'))
            
    tabla.setStyle(style)
    elementos.append(tabla)
    elementos.append(Spacer(1, 24))
    
    # 5. Construir el PDF
    try:
        doc.build(elementos)
        print(f"\n✅ Informe guardado en: {pdf_filename}")
    except Exception as e:
        print(f"\n❌ Error al generar el PDF: {e}")




# =========================================================================
# --- FUNCIÓN PRINCIPAL EXPORTABLE ---
# =========================================================================

def analisis_termico_rostro(image_path, pdf_filename, imagen_etiquetada_path):
    """
    Función principal que ejecuta el flujo completo de análisis térmico y reporte.
    Debe ser llamada desde el archivo principal.
    """
    if not os.path.exists(image_path):
        print(f"Error: La ruta de entrada {image_path} no existe.")
        return

    if FACE_DETECTOR is None:
        print("Error: No se puede continuar sin el modelo DNN cargado.")
        return

    imagen_bgr = cv2.imread(image_path)
    if imagen_bgr is None:
        print(f"Error: No se pudo cargar la imagen BGR de {image_path}.")
        return

    # 1. Calibración (OCR)
    T_MIN_AUTO, T_MAX_AUTO = obtener_temperaturas_ocr(imagen_bgr)
    T_MAX_val, T_MIN_val, m, b = calcular_calibracion(T_MAX_AUTO, T_MIN_AUTO)
    print(f"Mapeo final: T = {m:.4f} * Pixel + {b:.4f}")

    # 2. Inicialización de imágenes y datos
    original_image = imagen_bgr.copy()
    imagen_con_zonas = imagen_bgr.copy()
    imagen_gris = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    (h, w) = original_image.shape[:2]
    analisis_data = {}
    rostro_encontrado = False

    # 3. Pre-procesamiento y Detección DNN
    gray_image_clahe = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image_clahe = clahe.apply(gray_image_clahe)
    enhanced_image_clahe_rgb = cv2.cvtColor(enhanced_image_clahe, cv2.COLOR_GRAY2BGR)

    blob = cv2.dnn.blobFromImage(cv2.resize(enhanced_image_clahe_rgb, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    FACE_DETECTOR.setInput(blob)
    detections = FACE_DETECTOR.forward()
    
    # 4. Mapeo de Zonas y Análisis Térmico
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            face_roi = original_image[startY:endY, startX:endX]
            if face_roi.shape[0] < 50 or face_roi.shape[1] < 50: continue

            results = FACE_MESH.process(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
            
            if results.multi_face_landmarks:
                rostro_encontrado = True
                face_landmarks = results.multi_face_landmarks[0]
                roi_width = endX - startX
                roi_height = endY - startY
                
                for nombre_zona, indices in ZONAS.items():
                    puntos = [
                        (int(face_landmarks.landmark[i].x * roi_width + startX),
                         int(face_landmarks.landmark[i].y * roi_height + startY))
                        for i in indices
                    ]
                    puntos_np = np.array(puntos, np.int32)
                    
                    if np.array(puntos).shape[0] >= 3:
                        resultados = analizar_zona_termica(imagen_gris, puntos_np, m, b)
                        
                        if resultados:
                            analisis_data[nombre_zona] = resultados
                            print(f"Zona {nombre_zona}: T Promedio = {resultados['promedio']:.2f} C")
                            
                            # Dibujar y Etiquetar
                            cv2.polylines(imagen_con_zonas, [puntos_np], isClosed=True, color=(255, 0, 255), thickness=2)
                            texto_etiqueta = nombre_zona.split('. ')[0] 
                            x, y, w_zona, h_zona = cv2.boundingRect(puntos_np)
                            cx, cy = x + w_zona // 2, y + h_zona // 2
                            cv2.putText(imagen_con_zonas, texto_etiqueta, (cx, cy), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
                        
                    else:
                        cv2.polylines(imagen_con_zonas, [puntos_np], isClosed=True, color=(255, 0, 0), thickness=2)

                break 
                
    if not rostro_encontrado:
        print(f"❌ No se pudo encontrar y mapear el rostro en {os.path.basename(image_path)}.")
        return

    # 5. Generar Reporte
    cv2.imwrite(imagen_etiquetada_path, imagen_con_zonas)
    imagen_nombre_base = os.path.splitext(os.path.basename(image_path))[0]
    
    generar_pdf_informe(pdf_filename, image_path, imagen_etiquetada_path, analisis_data, T_MAX_val, T_MIN_val, m, b, imagen_nombre_base)