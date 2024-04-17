# Automatic Vehicle Identification Number Tracker

This project implements Automatic License Plate Recognition (ALPR) using the YOLOv8 object detection algorithm in Python. It provides functionality to detect license plates in images and videos, along with utilities for processing and testing.

## Features

- License plate detection using YOLOv8.
- Utilities for image and video processing.
- Test script for evaluating ALPR performance.

## Project Structure
```
├── Data 
│   ├──Train
│   ├── Test
├── Dockerfile
├── README.md
├── model
│   ├── saved_model.keras
├── vid2array.py
├── training.npy
├── train.py
├── test.py
├── requirements.txt
├── results
    ├── live.png
    ├── test2_output.gif
    ├── abornal_frames.json
```

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/jaigane6387/Automatic-VIN-Tracker
    ```

2. Navigate to the project directory:

    ```bash
    cd Automatic-VIN-Tracker
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the main script `main.py` to perform car and license plate detection on a video.

    ```bash
    python main.py --input test_video.mp4
    ```

    Replace `test_video.mp4` with the path to your input video.

2. Utilize the provided utilities in `utils.py` for additional processing or customization.

3. Evaluate the performance of the Automatic Vehicle Identification Number system using the test script `test.py`.

    ```bash
    python test.py
    
## Output    
<center></center><img src="results/test2_output.gif" alt="Test output"></center>


## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.


## Contact

For any inquiries or support, please contact [your@email.com](mailto:your@email.com).
