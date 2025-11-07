# app_facemesh.py — Streamlit + MediaPipe FaceMesh (kamera idx=1, CAP_DSHOW)
import time
import cv2
import numpy as np
import mediapipe as mp
import streamlit as st

st.set_page_config(page_title="FaceMesh Live", layout="centered")
st.title("🎥 FaceMesh Live — MediaPipe + Streamlit")

# --- Ustawienia / sterowanie ---
if "run" not in st.session_state:
    st.session_state.run = False

col1, col2, col3 = st.columns(3)
with col1:
    mirror = st.checkbox("Lustrzane", value=True)
with col2:
    refine = st.checkbox("Refine landmarks", value=False)
with col3:
    width = st.slider("Szerokość podglądu", 640, 1280, 960, 10)

start = st.button("▶️ Start")
stop  = st.button("⏹ Stop")

if start:
    st.session_state.run = True
if stop:
    st.session_state.run = False

placeholder = st.empty()  # tu będziemy wstawiać klatki

# --- Nic nie rób, dopóki użytkownik nie kliknie Start ---
if not st.session_state.run:
    st.info("Kliknij **Start**, aby rozpocząć podgląd.")
    st.stop()

# --- Inicjalizacja MediaPipe ---
mp_face  = mp.solutions.face_mesh
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles
TESSEL   = mp_style.get_default_face_mesh_tesselation_style()
CONTOUR  = mp_style.get_default_face_mesh_contours_style()

# --- Kamera (Iriun zwykle idx=1) ---
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    st.error("❌ Nie udało się otworzyć kamery (idx=1). Upewnij się, że Iriun Desktop pokazuje obraz.")
    st.session_state.run = False
    st.stop()

# parametry strumienia
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# --- Pętla streamu ---
try:
    with mp_face.FaceMesh(
        static_image_mode=True,        # jak w Twojej działającej wersji OpenCV
        max_num_faces=1,
        refine_landmarks=bool(refine),
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as mesh:
        t0, n = time.time(), 0
        while st.session_state.run and cap.isOpened():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            if mirror:
                frame = cv2.flip(frame, 1)

            # MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mesh.process(rgb)

            lmk = 0
            if res.multi_face_landmarks:
                fl = res.multi_face_landmarks[0]
                lmk = len(fl.landmark)
                mp_draw.draw_landmarks(frame, fl, mp_face.FACEMESH_TESSELATION,
                                       landmark_drawing_spec=None, connection_drawing_spec=TESSEL)
                mp_draw.draw_landmarks(frame, fl, mp_face.FACEMESH_CONTOURS,
                                       landmark_drawing_spec=None, connection_drawing_spec=CONTOUR)

            # overlay + skalowanie do podglądu
            n += 1
            fps = n / max(1e-6, time.time() - t0)
            cv2.putText(frame, f"LMK={lmk}  FPS={fps:.1f}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            H, W = frame.shape[:2]
            new_h = max(1, int(H * (int(width) / max(1, W))))
            frame_resized = cv2.resize(frame, (int(width), new_h), interpolation=cv2.INTER_LINEAR)

            # wyświetlenie w placeholderze (bez wstępnego st.image([]))
            placeholder.image(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB), channels="RGB")
            time.sleep(0.001)
finally:
    cap.release()
