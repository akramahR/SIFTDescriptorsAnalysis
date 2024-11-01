# Image Processing with SIFT Keypoints Analysis

This project processes and analyzes images using Scale-Invariant Feature Transform (SIFT) to detect and compare keypoints across images. The script can be run on single or multiple images, extracting SIFT descriptors, resizing images, and generating histograms for visual word occurrences. It also computes similarity between images using a dissimilarity matrix.

## Features

- **SIFT Keypoints Extraction**: Detects keypoints and descriptors using the luminance (Y) component of the image.
- **Image Resizing**: Scales images to a target VGA resolution (480x600) while maintaining the aspect ratio.
- **Histogram Construction**: Creates histograms of visual word occurrences using k-means clustering on SIFT descriptors.
- **Dissimilarity Matrix**: Compares images using a dissimilarity matrix to highlight visual similarities or differences.
- **Keypoint Visualization**: Shows the original image alongside an image annotated with keypoints and orientations.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Matplotlib

Install the dependencies with:

```bash
pip install opencv-python-headless numpy matplotlib
```

## Usage

Run the script with one or more images as command-line arguments:

```bash
python script.py image1.jpg [image2.jpg ...]
```

- For a single image: The script detects keypoints and displays the annotated image.
- For multiple images: It performs keypoint detection, histogram construction, and displays a dissimilarity matrix.

## Code Structure

### Classes

- **SiftImage**: Represents an image with its SIFT descriptors, keypoints, and histogram.

### Functions

- **main()**: Parses command-line arguments and runs tasks based on the number of images.
- **get_file_name(file_path)**: Extracts the filename from the path.
- **extract_sift_luminance(image)**: Converts the image to grayscale and extracts SIFT keypoints and descriptors.
- **construct_histogram(siftImg, kmeans_centers, K)**: Constructs a histogram of visual word occurrences for a SIFT image.
- **visualize_keypoints(image)**: Annotates and displays keypoints on an image.
- **resize_image(img, target_width, target_height)**: Resizes an image while maintaining aspect ratio.
- **TotalKeypointsDescriptors(siftImagesList)**: Calculates the total keypoints and descriptors for a list of SIFT images.
- **run_task1()**: Processes a single image, resizes it, and visualizes keypoints.
- **run_task2(display)**: Processes multiple images, extracts descriptors, clusters them, constructs histograms, and creates a dissimilarity matrix.

## Example Output

1. **Single Image**: Displays the original and annotated images with keypoints.
2. **Multiple Images**:
   - Displays histograms of visual word frequencies.
   - Prints a dissimilarity matrix comparing images.

