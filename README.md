# Privacy-Preserving Face Detection and Anonymization

## Project Overview
This project implements a robust face detection and anonymization system using the **CenterFace** model. It is designed to automatically detect faces in images, videos, and real-time webcam feeds and apply an adaptive blur (Gaussian or Mosaic) to anonymize them. This ensures privacy while maintaining the overall context of the visual data.

The system uses a lightweight ONNX model for detection, making it suitable for various environments, and includes advanced features like temporal smoothing for video stability.

## Features
- **High-Performance Face Detection**: Utilizes the CenterFace ONNX model for accurate and efficient face detection.
- **Adaptive Anonymization**: Automatically selects between Gaussian blur and Mosaic blur based on the face size and region resolution.
- **Multiple Processing Modes**:
    - **Image Processing**: Anonymize faces in static images.
    - **Video Processing**: Process entire video files frame-by-frame.
    - **Webcam Stream**: Real-time anonymization of live webcam feed.
- **Advanced Tracking (Video)**: Integrates ORB feature matching and RANSAC for temporal smoothing, reducing jitter and ensuring consistent anonymization across frames.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- ONNX Runtime
- NumPy
- Pillow (PIL)
- Matplotlib
- Tqdm

## Installation
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install opencv-python onnx onnxruntime numpy pillow matplotlib tqdm
   ```

## Usage
The core implementation is provided in the Jupyter Notebook `face_detection_3.ipynb`.

### Running the Notebook
1. Open `face_detection_3.ipynb` in Jupyter Notebook or Google Colab.
2. Run the cells to initialize the `CenterFace` detector.
3. Use the provided functions to process your data:
   - `blur_faces_in_image(image_path)`
   - `blur_faces_in_video(input_path, output_path)`
   - `blur_faces_in_webcam_and_record(output_path)`

### Example Output
Below is a demonstration of the face anonymization capability:

| Original Image | Anonymized Output |
| :---: | :---: |
| ![Before Anonymization](22_Picnic_Picnic_22_183.jpg) | ![After Anonymization](output_22_Picnic.jpg) |

## Project Report
For a detailed explanation of the methodology, algorithms used (CenterFace, ORB, RANSAC), and performance analysis, please refer to the project report:
[Project Report (PDF)](116847153_Manjunaatt_116061887_Saketh_Project_Report.pdf)

## Credits
- **CenterFace**: A practical anchor-free face detection method.
- **OpenCV**: Open Source Computer Vision Library.
