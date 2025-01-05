# WatchStream

## Description

[`watchstream.py`](src/watchstream.py) is a Python script that captures video streams from an RTSP source, processes each frame using a YOLO model for object detection, and performs annotations and logging based on detected objects. It utilizes background subtraction to identify moving objects and logs contour areas to monitor activity.

## Features

- **RTSP Stream Capture:** Connects to an RTSP stream using credentials provided in the `.env` file.
- **Object Detection:** Uses a pretrained YOLO model (`yolo11n.pt`) to detect objects in each frame.
- **Background Subtraction:** Applies background subtraction to identify moving objects.
- **Frame Annotation:** Draws annotations on frames based on detection results.
- **Logging:** Records contour areas and elapsed time to log files.
- **Frame Saving:** Saves frames when specific conditions, such as local maxima of normalized area, are met.
- **Flask server** Streams the annotated frames to a web page accessible on your LAN

## Usage

1. **Clone the Repository:**
    ```sh
    git clone https://github.com/your-repo/watchstream.git
    cd watchstream
    ```

2. **Install Dependencies:**
    Ensure you have the required Python packages installed.
    ```sh
    pip install -r requirements.txt
    ```

3. **Configure Environment Variables:**
    Create a `.env` file in the root directory with the necessary configuration. See the [Example `.env`](#example-env) section below.

4. **Run the Script:**
    ```sh
    python src/watchstream.py
    ```

## Configuration

### .env

The `.env` file contains essential configuration variables for connecting to the RTSP stream and setting up proxies. Below is an example of how to structure your `.env` file:

```env
RTSP_URL=10.0.0.1/feed_url
RTSP_USERNAME=???
RTSP_PASSWORD=???

# if a proxy is needed
https_proxy=???
http_proxy=???
```

- **RTSP_URL:** The URL of the RTSP stream.
- **RTSP_USERNAME:** Username for RTSP authentication.
- **RTSP_PASSWORD:** Password for RTSP authentication.
- **https_proxy / http_proxy:** Proxy settings for network configurations.

Ensure that your `.env` file is correctly configured with your specific RTSP stream details and proxy settings.

## License

This project is licensed under the [GNU General Public License](datasets/coco8/LICENSE).

