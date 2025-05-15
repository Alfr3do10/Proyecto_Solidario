import cv2
import mediapipe as mp
import numpy as np
import math
import time

# Inicializar MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Ojos
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Parpadeo
EAR_THRESHOLD_LEFT = 0.21  # Valor inicial temporal
EAR_THRESHOLD_RIGHT = 0.21
CONSEC_FRAMES = 3
current_time = 0
blink_count_left = 0
blink_count_right = 0
last_blink_time_left = 0
last_blink_time_right = 0
blink_delay = 0.5  # Tiempo de espera entre parpadeos en segundos
frame_counter_left = 0
frame_counter_right = 0
# Calibración de EAR personalizada
calibrating = True
ear_baseline_left = []
ear_baseline_right = []
calibration_frames = 200

# EAR
def compute_ear(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    ear = (A + B) / (2.0 * C)
     
    
    return ear

# Yaw (rotación horizontal de cabeza)
def get_head_yaw(landmarks, w, h):
    left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    nose_tip = np.array([landmarks[1].x * w, landmarks[1].y * h])

    # Dirección horizontal de la cara
    eye_vector = right_eye - left_eye
    face_direction = nose_tip - (left_eye + eye_vector / 2)

    # Ángulo en grados (yaw) — izquierda negativa, derecha positiva
    yaw_rad = np.arctan2(face_direction[0], face_direction[1])
    yaw_deg = np.degrees(yaw_rad)

    # Mapear a rango -90 a 90 (suavizado y limitado)
    yaw_deg = np.clip(yaw_deg, -90, 90)

    # Convertir a rango tipo potenciómetro -255 a 255
    pot_value = int((yaw_deg / 90) * 255)
    return yaw_deg, pot_value

# Cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            lms = landmarks.landmark

            # EARs
            left_ear = compute_ear(lms, LEFT_EYE, w, h)
            right_ear = compute_ear(lms, RIGHT_EYE, w, h)
            if calibrating:
                ear_baseline_left.append(left_ear)
                ear_baseline_right.append(right_ear)

                cv2.putText(frame, f"Calibrando... {len(ear_baseline_left)}/{calibration_frames}",
                            (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if len(ear_baseline_left) >= calibration_frames:
                    EAR_THRESHOLD_LEFT = np.mean(ear_baseline_left) * 0.75
                    EAR_THRESHOLD_RIGHT = np.mean(ear_baseline_right) * 0.75
                    calibrating = False
                    print(f"Calibración completada.")
                    print(f"EAR_THRESHOLD_LEFT = {EAR_THRESHOLD_LEFT:.3f}")
                    print(f"EAR_THRESHOLD_RIGHT = {EAR_THRESHOLD_RIGHT:.3f}")
                
                # Evitar que continúe con conteo hasta terminar calibración
                cv2.imshow("Control con Parpadeo y Giro de Cabeza", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue


            # Conteo parpadeo izquierdo
            # Parpadeo izquierdo
            if left_ear < EAR_THRESHOLD_LEFT:
                frame_counter_left += 1
            else:
                if frame_counter_left >= CONSEC_FRAMES:
                    current_time = time.time()
                    if current_time - last_blink_time_left >= blink_delay:
                        blink_count_left += 1
                        last_blink_time_left = current_time
                frame_counter_left = 0

            # Parpadeo derecho
            if right_ear < EAR_THRESHOLD_RIGHT:
                frame_counter_right += 1
            else:
                if frame_counter_right >= CONSEC_FRAMES:
                    current_time = time.time()
                    if current_time - last_blink_time_right >= blink_delay:
                        blink_count_right += 1
                        last_blink_time_right = current_time
                frame_counter_right = 0

            # Inclinación de cabeza (yaw)
            yaw_deg, pot_value = get_head_yaw(lms, w, h)

            # Mostrar malla facial
            mp_drawing.draw_landmarks(
                frame, landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
            )

            # Mostrar valores
            cv2.putText(frame, f"Parpadeos IZQ: {blink_count_left}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f"Parpadeos DER: {blink_count_right}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Yaw (°): {yaw_deg:.1f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Potenciómetro: {pot_value}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("Control con Parpadeo y Giro de Cabeza", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()