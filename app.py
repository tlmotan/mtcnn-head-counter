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
    st.title("Head counter for lazy folks :moyai:")

    st.markdown("""
        ###  :rainbow[How to use?]
            1. Browse and upload an image (must be jpg, jpeg or png).
            2. Click on the "Start Count" button and get results.
                
        ###  :rainbow[How it works?]
            This app uses a face detection algorithm called MTCNN to identify faces
            in the image. Once the faces are detected, bounding boxes are overlayed 
            around each detected face, which provides us with the total head count.

    """)

    uploaded_file = st.file_uploader('Choose an image:',type=["jpg", "jpeg", "png"])

    
    if uploaded_file is not None:
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        if st.button("Start Count"):
            with st.spinner("Detecting faces..."):
                result_image, num_faces = detect_faces(image)
            st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_column_width=True)
            st.subheader(f'There are {num_faces} people.')
            
if __name__ == "__main__":
    main()
