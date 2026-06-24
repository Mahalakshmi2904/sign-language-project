# Sign Language to Text Conversion Using Computer Vision

#### Overview

Sign Language to Text Conversion is a Computer Vision and Machine Learning-based application that recognizes hand gestures and converts them into readable text in real time. The system is designed to improve communication between hearing-impaired individuals and people unfamiliar with sign language by providing an accessible and efficient translation solution. In addition to text conversion, the application includes Text-to-Speech functionality, enabling recognized text to be spoken aloud for enhanced accessibility and user interaction.

#### Features

- Real-time hand gesture detection using a webcam
- Hand landmark tracking with MediaPipe
- Sign language recognition using Machine Learning models
- Conversion of recognized gestures into text
- Live prediction and sentence formation
- User-friendly web interface
- Text-to-Speech support for enhanced accessibility

## Problem Statement

Communication barriers often exist between individuals who use sign language and those who do not understand it. This project addresses that challenge by automatically translating sign language gestures into text, enabling smoother and more effective communication.

## Technology Stack

***Tech Stack:** Python, OpenCV, MediaPipe, Flask, Machine Learning, HTML, CSS, JavaScript*

## How It Works

1. Captures live video from the webcam.
2. Detects hand landmarks using MediaPipe.
3. Extracts gesture features from the detected hand.
4. Uses trained Machine Learning models to classify gestures.
5. Converts recognized gestures into text and displays the output in real time.
6. Converts the generated text into speech using Text-to-Speech technology.
7. Supports sentence formation and accessibility features.

## Key Contributions

- Developed a real-time sign language recognition system that converts hand gestures into text using Computer Vision and Machine Learning techniques.
- Implemented hand landmark detection and gesture recognition using OpenCV and MediaPipe for accurate sign interpretation.
- Built a Flask-based web application to display recognized text in real time.
- Collected and processed gesture datasets for model training and evaluation.
- Integrated machine learning models for live gesture prediction and recognition.
- Improved accessibility by enabling automated sign-to-text conversion for hearing and speech-impaired individuals.

## Project Structure

- Dataset Collection
- Data Preprocessing
- Feature Extraction
- Model Training
- Real-Time Prediction
- Web Interface

## Results

- Accurate real-time gesture recognition
- Fast prediction and response time
- Improved communication support through automated text generation
- Practical application of Computer Vision and Machine Learning techniques

## Project Demo

### Application Interface


### Gesture Recognition

![Gesture Detection](images/gesture.png)

### Output Prediction

![Output](images/output.png)

## Future Enhancements

- Support for complete sentence recognition
- Multi-language translation
- Mobile application integration
- Improved accuracy using deep learning models
- Cloud deployment for wider accessibility

## Authors

- Mahalakshmi S
- Harini K.S

## License

This project was developed as an academic project for educational and research purposes.
