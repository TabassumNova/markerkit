# Marker-detection
Aruco and barcode markers


Aruco creation
---------------------------

```python
from marker_detection import generate_arucos

generate_arucos(
	output="location/to/output/path/my_board.svg",
	columns=12,
	rows=9,
	p_type="charuco_board",
	square_size=13,
	aruco_marker_size=8,
	dict_name="DICT_6x6_1000.json",
)
```

This writes the SVG to the path you pass in `output`.

Aruco inner and outer border detection
---------------------------

<img src="images/Aruco_detection.png" alt="Aruco Detection" width="600" height="400" />

```python
import cv2
from marker_detection import getAruco

image = cv2.imread("path/to/image.png")
markers = getAruco(image, cv2.aruco.DICT_4X4_1000, visualisation=False)
```

# References
- https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html#autotoc_md255
- https://www.geeksforgeeks.org/python/how-to-generate-barcode-in-python/
- https://www.geeksforgeeks.org/python/detect-and-read-barcodes-with-opencv-in-python/
