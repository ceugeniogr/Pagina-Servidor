import cv2
import mediapipe as mp
import os
import numpy as np
import pytesseract
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm

# =========================================================================
# === CONFIGURACIÓN GLOBAL DE TESSERACT Y ROIs ===
# =========================================================================

# --- Configuración OCR de Tesseract ---
# Descomentar si es necesario:
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract' 

# Coordenadas de las ROIs (Regiones de Interés) para el texto de la temperatura (OCR)
# Estas coordenadas son específicas de tu imagen FLIR3589.jpg
ROI_TMAX_TEXTO = [568, 7, 628, 40]
ROI_TMIN_TEXTO = [561, 439, 634, 474] 

# =========================================================================
# === 1. FUNCIONES DE PROCESAMIENTO MEDIAPIPE (DIBUJO DE CONTORNOS MAGENTA) ===
# =========================================================================

# --- Configuración de MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Configuración de CLAHE ---
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# --- Función para trazar una línea entre puntos de MediaPipe ---
def trazar_linea_entre_puntos(image, points, indices_to_connect, color, thickness, is_closed=False):
    if len(indices_to_connect) < 2: return
    points_to_draw = np.array([points[i] for i in indices_to_connect], dtype=np.int32)
    cv2.polylines(image, [points_to_draw], isClosed=is_closed, color=color, thickness=thickness)

# --- Traza y filtra una línea usando una máscara ---
def filtrar_y_trazar_linea_dentro_contorno(image, all_points, indices_to_connect, contour_mask, color, thickness):
    temp_line_image = np.zeros_like(image, dtype=np.uint8)
    
    # 1. Trazar la línea en una imagen temporal
    trazar_linea_entre_puntos(temp_line_image, all_points, indices_to_connect, color, thickness, is_closed=False)
    
    # 2. Convertir la línea temporal a escala de grises para el filtro
    temp_line_gray = cv2.cvtColor(temp_line_image, cv2.COLOR_BGR2GRAY)
    
    # 3. Aplicar la máscara de contorno a la línea (solo mantiene lo que está DENTRO)
    masked_line_gray = cv2.bitwise_and(temp_line_gray, temp_line_gray, mask=contour_mask)
    
    # 4. Obtener solo los píxeles de la línea que se conservaron
    line_pixels = cv2.bitwise_and(temp_line_image, temp_line_image, mask=masked_line_gray)
    
    # 5. Superponer la línea filtrada sobre la imagen original
    indices = masked_line_gray > 0
    image[indices] = line_pixels[indices]

# --- Función para proyectar una línea desde un punto hacia fuera del contorno del dedo (Meñique/Índice) ---
def proyectar_linea(image, adjusted_points, all_points, finger_name, finger_indices, base_idx, ref_idx, thickness_for_fingers, color):
    finger_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    indices = finger_indices[finger_name]
    for i in range(len(indices) - 1):
        pt1_idx = all_points[indices[i]]; pt2_idx = all_points[indices[i+1]]
        cv2.line(finger_mask, pt1_idx, pt2_idx, 255, thickness=thickness_for_fingers)
    finger_mask = cv2.morphologyEx(finger_mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    
    finger_contours, _ = cv2.findContours(finger_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finger_contour = finger_contours[0] if finger_contours else None

    if finger_contour is None: return None

    P_base = adjusted_points[base_idx].astype(float)
    P_ref = adjusted_points[ref_idx].astype(float)
    projection_vector = P_base - P_ref
    
    norm = np.linalg.norm(projection_vector)
    if norm != 0: projection_vector /= norm
        
    current_point = P_base.copy()
    step_size = 2
    line_points_projection = []
    line_points_projection.append(tuple(current_point.astype(int)))
    final_projection_point = None
    
    while True:
        current_point += projection_vector * step_size
        current_point_int = current_point.astype(int)
        
        if cv2.pointPolygonTest(finger_contour, tuple(current_point_int.tolist()), False) >= 0:
            line_points_projection.append(tuple(current_point_int.tolist()))
        else:
            final_projection_point = line_points_projection[-1]
            break
    
    if len(line_points_projection) > 1:
        cv2.polylines(image, [np.array(line_points_projection, dtype=np.int32)], isClosed=False, color=color, thickness=2)
    
    return final_projection_point

# --- Proyección de Cierre Lateral de Falange ---
def proyectar_linea_cierre_falange(image, all_points, P_start_idx, P_ref_idx, finger_contour, color):
    P_start = all_points[P_start_idx].astype(float)
    P_ref = all_points[P_ref_idx].astype(float)
    
    projection_vector = P_start - P_ref
    norm = np.linalg.norm(projection_vector)
    if norm != 0: projection_vector /= norm
        
    current_point = P_start.copy()
    step_size = 2
    line_points = [tuple(current_point.astype(int))]
    
    while True:
        current_point += projection_vector * step_size
        current_point_int = current_point.astype(int)
        
        # El límite es el contorno del dedo.
        if cv2.pointPolygonTest(finger_contour, tuple(current_point_int.tolist()), False) >= 0:
            line_points.append(tuple(current_point_int.tolist()))
        else:
            break
            
    if len(line_points) > 1:
        cv2.polylines(image, [np.array(line_points, dtype=np.int32)], isClosed=False, color=color, thickness=2)
        return line_points[-1]

    return None

# --- Proyección del Pulgar (3) con Inversión de Dirección ---
def proyectar_linea_pulgar(image, all_points, adjusted_points, finger_indices, thickness_for_fingers, color, P_start_idx, P_target_idx, invert_direction=False):
    P_start = all_points[P_start_idx].astype(float)
    P_target = adjusted_points[P_target_idx].astype(float)

    # Generar la máscara SÓLO del contorno del pulgar
    thumb_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    thumb_indices = finger_indices['thumb']
    for i in range(len(thumb_indices) - 1):
        pt1 = all_points[thumb_indices[i]]; pt2 = all_points[thumb_indices[i+1]]
        cv2.line(thumb_mask, pt1, pt2, 255, thickness=thickness_for_fingers)
    thumb_mask = cv2.morphologyEx(thumb_mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    thumb_contours, _ = cv2.findContours(thumb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thumb_contour = thumb_contours[0] if thumb_contours else None
    
    if thumb_contour is None: return None

    direction_vector = P_target - P_start
    
    # Aplicar la inversión de dirección solicitada y cambiar color a magenta si se invierte
    if invert_direction:
        direction_vector = -direction_vector
        draw_color = (255, 0, 255) # MAGENTA
    else:
        draw_color = color
        
    norm = np.linalg.norm(direction_vector)
    if norm != 0: direction_vector /= norm
        
    current_point = P_start.copy()
    step_size = 2
    line_points = []
    
    while True:
        current_point_int = current_point.astype(int)
        
        # Si el punto ya no está DENTRO del contorno del pulgar, se detiene
        if cv2.pointPolygonTest(thumb_contour, tuple(current_point_int.tolist()), False) < 0:
            break
        
        line_points.append(tuple(current_point_int.tolist()))
        current_point += direction_vector * step_size

    if len(line_points) > 1:
        cv2.polylines(image, [np.array(line_points, dtype=np.int32)], isClosed=False, color=draw_color, thickness=2)
        return line_points[-1]

    return None

# =========================================================================
# === 2. FUNCIONES DE ANÁLISIS TÉRMICO Y PDF ===
# =========================================================================

def obtener_temperaturas_ocr(image_bgr, roi_max, roi_min):
    """
    Intenta leer T_MIN y T_MAX de la imagen usando OCR.
    """
    def procesar_roi(image, roi):
        x1, y1, x2, y2 = roi
        if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]):
            return None
        roi_region = image[y1:y2, x1:x2] 
        gray = cv2.cvtColor(roi_region, cv2.COLOR_BGR2GRAY)
        # Configuración OCR mejorada para números
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(thresh, config='--psm 6 -c tessedit_char_whitelist=0123456789.-')
        clean_value = re.sub(r'[^\d\.\-]', '', text).strip()
        try:
            return float(clean_value.replace(',', '.'))
        except ValueError:
            return None
            
    T_MAX_AUTO = procesar_roi(image_bgr, roi_max)
    T_MIN_AUTO = procesar_roi(image_bgr, roi_min)
    return T_MIN_AUTO, T_MAX_AUTO

def pixel_a_temperatura(pixel_val, m, b):
    """Convierte un valor de píxel a temperatura usando la fórmula lineal: T = m * P + b."""
    return m * pixel_val + b

# =========================================================================
# === 3. FUNCIÓN PRINCIPAL DE PROCESAMIENTO DE MANOS ===
# =========================================================================

def analisis_termico_manos(image_path, pdf_filename, imagen_etiquetada_path):
    """
    Realiza la segmentación de manos con MediaPipe, el análisis térmico de las
    zonas magenta y genera un informe en PDF.

    Args:
        image_path (str): Ruta de la imagen térmica de entrada (FLIR*.jpg).
        pdf_filename (str): Ruta completa para guardar el PDF de salida.
        imagen_etiquetada_path (str): Ruta completa para guardar la imagen
                                      segmentada y etiquetada final.
    """

    print("--- INICIO DEL PROCESO DE ANÁLISIS DE MANOS ---")
    filename = os.path.basename(image_path)
    
    # 🚨 Definición de la ruta temporal (se guarda en el mismo directorio base)
    BASE_DIR = os.path.dirname(image_path) if os.path.dirname(image_path) else "."
    mp_output_path = os.path.join(BASE_DIR, "temp_segmentada_manos_" + filename)
    
    # 1. --- ETAPA DE PROCESAMIENTO MEDIAPIPE (Generación de Contornos) ---
    print(f"\n--- Procesando {filename} con MediaPipe (Manos) ---")

    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        print(f"ERROR: No se pudo leer la imagen original: {filename}")
        return

    image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    image_final_display = image_bgr.copy() # Usada para dibujar los contornos

    current_image_to_process = image_bgr.copy()
    current_image_to_process.flags.writeable = False
    
    # Intento inicial de detección
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        results = hands.process(current_image_to_process)
        
        # Si falla, aplica CLAHE y reintenta
        if not results.multi_hand_landmarks:
            print(f"No se detectaron manos en {filename}. Aplicando filtro CLAHE...")
            processed_image_gray = clahe.apply(image_gray)
            processed_image_bgr = cv2.cvtColor(processed_image_gray, cv2.COLOR_GRAY2BGR)
            processed_image_bgr.flags.writeable = False
            results = hands.process(processed_image_bgr)
            processed_image_bgr.flags.writeable = True
            current_image_to_process = processed_image_bgr.copy()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = image_final_display.shape 
                all_points = np.array([[int(lm.x * w), int(lm.y * h)] for lm in hand_landmarks.landmark], dtype=np.int32)

                # --- Configuración de dibujo ---
                finger_indices = {
                    'thumb': [1, 2, 3, 4], 'index': [5, 6, 7, 8],
                    'middle': [9, 10, 11, 12], 'ring': [13, 14, 15, 16],
                    'pinky': [17, 18, 19, 20]
                }
                
                # Preparar puntos ajustados (código de ajuste omitido por brevedad, pero necesario)
                points_to_adjust = {2: 3, 5: 6, 9: 10, 13: 14, 17: 18}
                displacement_factor = 0.5
                adjusted_points = all_points.copy()
                for base_idx, next_idx in points_to_adjust.items():
                    base_point_original = all_points[base_idx]
                    next_point_reference = all_points[next_idx]
                    direction_vector = next_point_reference - base_point_original
                    norm = np.linalg.norm(direction_vector)
                    if norm != 0:
                        direction_vector = direction_vector / norm
                    adjusted_points[base_idx] = base_point_original + direction_vector * (norm * displacement_factor)
                
                # --- DIBUJO DE CONTORNOS MAGENTA (Código original mantenido) ---
                thickness_for_fingers = 20
                MAGENTA_COLOR = (255, 0, 255) # BGR
                
                pinky_end_point_tuple = proyectar_linea(image_final_display, adjusted_points, all_points, finger_name='pinky', finger_indices=finger_indices, base_idx=17, ref_idx=13, thickness_for_fingers=thickness_for_fingers, color=MAGENTA_COLOR)
                index_end_point_tuple = proyectar_linea(image_final_display, adjusted_points, all_points, finger_name='index', finger_indices=finger_indices, base_idx=5, ref_idx=9, thickness_for_fingers=thickness_for_fingers, color=MAGENTA_COLOR)
                proyectar_linea_pulgar(image_final_display, all_points, adjusted_points, finger_indices=finger_indices, thickness_for_fingers=thickness_for_fingers, color=MAGENTA_COLOR, P_start_idx=3, P_target_idx=5, invert_direction=False)
                proyectar_linea_pulgar(image_final_display, all_points, adjusted_points, finger_indices=finger_indices, thickness_for_fingers=thickness_for_fingers, color=MAGENTA_COLOR, P_start_idx=3, P_target_idx=5, invert_direction=True)

                pinky_end_point = np.array(pinky_end_point_tuple, dtype=np.int32) if pinky_end_point_tuple is not None else adjusted_points[17]
                index_end_point = np.array(index_end_point_tuple, dtype=np.int32) if index_end_point_tuple is not None else adjusted_points[5]
                
                palm_hull_points = [adjusted_points[0], adjusted_points[1], index_end_point, adjusted_points[9], adjusted_points[13], pinky_end_point]
                palm_hull_points_np = np.array(palm_hull_points, dtype=np.int32)
                
                palm_mask = np.zeros(image_final_display.shape[:2], dtype=np.uint8)
                if len(palm_hull_points_np) > 2:
                    cv2.fillConvexPoly(palm_mask, palm_hull_points_np, 255) 
                    
                all_fingers_mask = np.zeros(image_final_display.shape[:2], dtype=np.uint8)
                index_contour = None
                pinky_contour = None

                for finger_name, indices in finger_indices.items():
                    finger_mask = np.zeros(image_final_display.shape[:2], dtype=np.uint8)
                    for i in range(len(indices) - 1):
                        pt1 = all_points[indices[i]]; pt2 = all_points[indices[i+1]]
                        cv2.line(finger_mask, pt1, pt2, 255, thickness=thickness_for_fingers)
                    
                    kernel_finger = np.ones((5,5),np.uint8)
                    finger_mask = cv2.morphologyEx(finger_mask, cv2.MORPH_CLOSE, kernel_finger)
                    all_fingers_mask = cv2.bitwise_or(all_fingers_mask, finger_mask)

                    filtered_finger_mask = cv2.bitwise_and(finger_mask, cv2.bitwise_not(palm_mask))
                    
                    finger_contours, _ = cv2.findContours(filtered_finger_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if finger_contours:
                        largest_finger_contour = max(finger_contours, key=cv2.contourArea)
                        cv2.drawContours(image_final_display, [largest_finger_contour], -1, MAGENTA_COLOR, 2)
                        
                        if finger_name == 'index': index_contour = largest_finger_contour
                        elif finger_name == 'pinky': pinky_contour = largest_finger_contour

                if len(palm_hull_points_np) > 2:
                    palm_hull_adjusted = cv2.convexHull(palm_hull_points_np)
                    cv2.drawContours(image_final_display, [palm_hull_adjusted], -1, MAGENTA_COLOR, 2)
                    
                separation_line_indices_final = [5, 9, 13, 17]
                trazar_linea_entre_puntos(image_final_display, adjusted_points, separation_line_indices_final, MAGENTA_COLOR, 2)
                
                falange_contour_mask = cv2.bitwise_and(all_fingers_mask, cv2.bitwise_not(palm_mask))
                
                for indices in [[6, 10, 14, 18], [7, 11, 15, 19]]:
                    filtrar_y_trazar_linea_dentro_contorno(image_final_display, all_points, indices, falange_contour_mask, color=MAGENTA_COLOR, thickness=2)
                
                if index_contour is not None:
                    proyectar_linea_cierre_falange(image_final_display, all_points, P_start_idx=7, P_ref_idx=11, finger_contour=index_contour, color=MAGENTA_COLOR)
                    proyectar_linea_cierre_falange(image_final_display, all_points, P_start_idx=6, P_ref_idx=10, finger_contour=index_contour, color=MAGENTA_COLOR)
                if pinky_contour is not None:
                    proyectar_linea_cierre_falange(image_final_display, all_points, P_start_idx=19, P_ref_idx=15, finger_contour=pinky_contour, color=MAGENTA_COLOR)
                    proyectar_linea_cierre_falange(image_final_display, all_points, P_start_idx=18, P_ref_idx=14, finger_contour=pinky_contour, color=MAGENTA_COLOR)


    # **PASO CLAVE:** GUARDAR la imagen segmentada temporalmente
    cv2.imwrite(mp_output_path, image_final_display)
    print(f"✅ Imagen temporal segmentada guardada en: {mp_output_path}")

    # 2. --- ETAPA DE ANÁLISIS TÉRMICO (Detección de Magenta y PDF) ---
    print(f"\n--- Analizando Temperaturas y generando PDF para {filename} ---")
    
    # Cargar la imagen generada por MediaPipe (LA TEMPORAL)
    imagen_mp = cv2.imread(mp_output_path)
    if imagen_mp is None:
        print("ERROR: No se pudo cargar la imagen procesada por MediaPipe.")
        return

    # Cargar la imagen original para la escala de grises (y OCR)
    imagen_original_bgr = cv2.imread(image_path)
    imagen_gris = cv2.cvtColor(imagen_original_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2.1. Calibración (OCR)
    T_MIN_AUTO, T_MAX_AUTO = obtener_temperaturas_ocr(imagen_original_bgr, ROI_TMAX_TEXTO, ROI_TMIN_TEXTO)
    T_MAX_val = T_MAX_AUTO if T_MAX_AUTO is not None and T_MAX_AUTO > 10 else 37.0
    T_MIN_val = T_MIN_AUTO if T_MIN_AUTO is not None and T_MIN_AUTO < 40 else 25.0

    if T_MAX_AUTO is None or T_MIN_AUTO is None or T_MAX_val <= T_MIN_val:
        print(f"ATENCIÓN: Falló OCR o rango inválido. Usando respaldo: T_MAX={T_MAX_val:.1f} C, T_MIN={T_MIN_val:.1f} C.")
    else:
        print(f"OCR DETECTADO: T_MAX={T_MAX_val:.1f} C, T_MIN={T_MIN_val:.1f} C.")

    P_HIGH = 255.0; P_LOW = 0.0
    rango_temp = T_MAX_val - T_MIN_val
    rango_pixel = P_HIGH - P_LOW 
    m = rango_temp / rango_pixel
    b = T_MIN_val

    # 2.2. Detección de Zonas (Magenta)
    hsv = cv2.cvtColor(imagen_mp, cv2.COLOR_BGR2HSV)
    lower_magenta = np.array([145, 170, 190]) 
    upper_magenta = np.array([155, 240, 255])
    mascara_magenta = cv2.inRange(hsv, lower_magenta, upper_magenta)
    kernel = np.ones((3, 3), np.uint8)
    mascara_magenta = cv2.morphologyEx(mascara_magenta, cv2.MORPH_CLOSE, kernel)
    contornos, _ = cv2.findContours(mascara_magenta, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    umbral_area = 28  
    contornos_filtrados = [c for c in contornos if cv2.contourArea(c) > umbral_area]
    contornos_ordenados = sorted(contornos_filtrados, key=cv2.contourArea, reverse=True)
    contornos_finales = contornos_ordenados[:32]
    
    print(f"Se analizarán las {len(contornos_finales)} zonas magenta detectadas.")

    datos_analisis = [] 
    imagen_etiquetar = imagen_mp.copy() # Usar la imagen MP para dibujar etiquetas

    for i, contorno in enumerate(contornos_finales):
        mascara_zona_actual = np.zeros_like(mascara_magenta)
        cv2.drawContours(mascara_zona_actual, [contorno], -1, 255, thickness=cv2.FILLED)

        # Los valores de píxel se extraen de la imagen ORIGINAL en GRISES
        zona_actual_pixeles = imagen_gris[np.where(mascara_zona_actual == 255)]
        
        if len(zona_actual_pixeles) == 0: continue

        pixeles_temperatura = pixel_a_temperatura(zona_actual_pixeles, m, b)
        
        temperatura_promedio = np.mean(pixeles_temperatura)
        temperatura_maxima = np.max(pixeles_temperatura)
        temperatura_minima = np.min(pixeles_temperatura)
        
        datos_analisis.append([
            f"{i+1}", f"{temperatura_maxima:.2f}°C",
            f"{temperatura_minima:.2f}°C", f"{temperatura_promedio:.2f}°C"
        ])

        # Etiquetar en la imagen
        M = cv2.moments(contorno)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            texto = f"{i+1}"
            font_scale = 0.35; font_thickness = 1
            text_color = (255, 255, 255) # Blanco para mejor contraste

            (ancho_texto, alto_texto), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            pos_x = cx - ancho_texto // 2
            pos_y = cy + alto_texto // 2

            cv2.putText(imagen_etiquetar, texto, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

    # 2.3. Guardado Final de Imagen Etiquetada (RUTA DE SALIDA)
    text_scale = f"T Max: {T_MAX_val:.1f} C, T Min: {T_MIN_val:.1f} C"
    cv2.putText(imagen_etiquetar, text_scale, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(imagen_etiquetada_path, imagen_etiquetar)
    print(f"🖼️ Imagen de análisis etiquetada guardada en: {imagen_etiquetada_path}")

    # 2.4. Exportar a PDF
    print("\n--- Generando PDF de Informe de Manos ---")
    doc = SimpleDocTemplate(pdf_filename, pagesize=A4, leftMargin=2.5*cm, rightMargin=2.5*cm, topMargin=2.5*cm, bottomMargin=2.5*cm)
    styles = getSampleStyleSheet()
    elementos = []

    # Título
    elementos.append(Paragraph("<b>INFORME DE ANÁLISIS TÉRMICO DE MANO</b>", styles['Title']))
    elementos.append(Spacer(1, 12))

    # Datos de Calibración
    calibracion_info = f"<b>Calibración:</b> T_MAX={T_MAX_val:.1f} C, T_MIN={T_MIN_val:.1f} C."
    elementos.append(Paragraph(calibracion_info, styles['Normal']))
    elementos.append(Spacer(1, 18))

    # Imágenes
    try:
        # IMAGEN 1: Original (usa la ruta original)
        elementos.append(Paragraph("<b>Imagen Térmica Original</b>", styles['Heading2']))
        elementos.append(Spacer(1, 6))
        img_original = Image(image_path, width=220, height=165) 
        elementos.append(img_original)
        elementos.append(Spacer(1, 12))

        # IMAGEN 2: Segmentada con Zonas Etiquetadas (usa la ruta final etiquetada)
        elementos.append(Paragraph("<b>Imagen con Zonas Etiquetadas para Análisis</b>", styles['Heading2']))
        elementos.append(Spacer(1, 6))
        img_etiquetada = Image(imagen_etiquetada_path, width=220, height=165)
        elementos.append(img_etiquetada)
        elementos.append(Spacer(1, 18))
    except Exception as e:
        elementos.append(Paragraph(f"<i>Error al cargar imágenes para PDF: {e}</i>", styles['Normal']))

    # Tabla de Resultados
    elementos.append(Paragraph("<b>Resultados Detallados por Zona</b>", styles['Heading2']))
    elementos.append(Spacer(1, 6))
    datos_tabla = [["Zona", "T. Máxima", "T. Mínima", "T. Promedio"]] + datos_analisis
    tabla = Table(datos_tabla, colWidths=[2.5*cm, 3.5*cm, 3.5*cm, 3.5*cm]) 
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2A6F8F')), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke), 
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('GRID', (0, 0), (-1, -1), 0.5, colors.black), 
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
    ])
    for i in range(len(datos_tabla)):
        if i > 0 and i % 2 == 1: style.add('BACKGROUND', (0, i), (-1, i), colors.HexColor('#F0F8FF')) 
    tabla.setStyle(style)
    elementos.append(tabla)
    elementos.append(Spacer(1, 24))

    try:
        doc.build(elementos)
        print(f"✅ --- PDF DE MANOS CREADO EXITOSAMENTE ---")
        print(f"Informe guardado en: {pdf_filename}")
        
        # **PASO CLAVE:** Eliminar imagen TEMPORAL de MediaPipe para limpiar
        if os.path.exists(mp_output_path):
            os.remove(mp_output_path)
            print("🧹 Imagen temporal de MediaPipe eliminada.")

    except Exception as e:
        print(f"❌ Error al generar el PDF de manos: {e}")