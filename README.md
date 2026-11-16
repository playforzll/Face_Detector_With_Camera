
# Face Detector Camera

A Python-based real-time face detection project using OpenCV and a pre-trained Caffe deep learning model.

## Features

- Real-time face detection using a camera.
- Uses OpenCV's Deep Neural Network (DNN) module.
- Uses a pre-trained SSD face detection model.
- Detects faces through a webcam.

## Project Structure

```text
Face-Detector-Camera/
├── face_dnn/
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── main.py
└── README.md
```

## Requirements

- Python 3
- OpenCV
- A working webcam

## Installation

1. Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
```

2. Enter the project directory:

```bash
cd Face-Detector-Camera
```

3. Install OpenCV:

```bash
pip install opencv-python
```

## Usage

Run the following command:

```bash
python3 main.py
```

Make sure your webcam is connected and accessible.

## Technologies Used

- Python
- OpenCV
- Caffe
- SSD (Single Shot Detector)

## License

This project is licensed under the GNU General Public License v3.0, if you choose GPL-3.0. See the LICENSE file for details.
