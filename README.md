# Machine Learning Head Counter App
![Screenshot](https://github.com/rachels-archive/mtcnn-head-counter/assets/79963756/9dae323e-231f-4d99-af48-e39797a89fbe)

This is a web application built with Python and Streamlit, that automates the task of head counting by leveraging the MTCNN (Multi-Task Cascaded Convolutional Neural Network) algorithm. [Try it out here!](https://ml-headcounter.streamlit.app/)


## Why I Built This
I was assigned the weekly task of conducting headcounts for a school club, which was time-consuming and tedious. I decided to build this application to simplify my life and learn more about ML.

## How to Use
1. **Upload an Image**: Select and upload an image file (must be jpg, jpeg, or png format).
2. **Start Count**: Click on the "Start Count" button to begin face detection.
3. **View Results**: The app will display the uploaded image with bounding boxes around detected faces along with the total number of people detected.

## How It Works
- **Face Detection Algorithm**: The application utilizes the MTCNN, a deep learning model specifically designed for face detection tasks. Before feeding the image into the model, the image is preprocessed.
- **Preprocessing**: This involves converting the image to grayscale to simplify processing, performing histogram equalization to enhance contrast and improve overall image quality.
- **Face Detection**: MTCNN detects faces in the image by analyzing the candidate regions and refining them iteratively. 
- **Visualization**: After detecting faces, bounding boxes are overlaid onto the original image around each detected face, allowing users to visually verify the accuracy of the results.
- **Head Counting**: The total number of detected faces corresponds to the head count. 

---
If you have any suggestions, feedback, or issues, feel free to open an [issue](https://github.com/rachels-archive/mtcnn-head-counter/issues).

