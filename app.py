import streamlit as st
import cv2
import numpy as np
from mtcnn import MTCNN
from io import BytesIO

def detect_faces(image):
    mtcnn = MTCNN(min_face_size=20) 

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Equalize histogram
    equalized_image = cv2.equalizeHist(gray_image)
    
    # Convert image back to BGR
    enhanced_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)
    
    # Perform face detection
    faces = mtcnn.detect_faces(enhanced_image)
    
    # Draw rectangles around detected faces
    num_faces = len(faces)
    for face in faces:
        x, y, width, height = face['box']
        cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)
    
    return image, num_faces


def main():
    st.title("Face Detection Web App")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Count Faces"):
            with st.spinner("Detecting faces..."):
                result_image, num_faces = detect_faces(image)
            st.image(result_image, caption=f"Detected {num_faces} face(s)", use_column_width=True)
            
if __name__ == "__main__":
    main()
