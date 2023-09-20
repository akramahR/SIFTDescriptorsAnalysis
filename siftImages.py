import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

def main():
    """
    The main entry point of the script.
    Parses command-line arguments and performs tasks based on the provided inputs.
    """
    if len(sys.argv) < 2:
        print("Usage: python script.py image1.jpg [image2.jpg ...]")
        sys.exit(1)
    if len(sys.argv) == 2:
        run_task1()
    else:
        # pass true to display if we want to see images and histograms
        run_task2(display= False)

class SiftImage:
    name = ""
    image = None
    descriptors = None
    keyPoints = None
    histogram = None


def get_file_name(file_path):
    file_path_components = file_path.split('/')
    file_name_and_extension = file_path_components[-1]
    return file_name_and_extension

# Function to extract SIFT keypoints and descriptors from the luminance Y component of an image
def extract_sift_luminance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

#function to construct a histogram of visual word occurrences for a SIFT image.
def construct_histogram(siftImg, kmeans_centers, K):    
    labels = np.argmin(np.linalg.norm(siftImg.descriptors[:, None] - kmeans_centers, axis=2), axis=1)
    hist, _ = np.histogram(labels, bins=range(K + 1))   
    return hist

# function to visualize SIFT keypoints on an image.
def visualize_keypoints(image):
    # Detect keypoints and compute descriptors
    keypoints, descriptors = extract_sift_luminance(image)

    # Draw the keypoints on the original image
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

    # Iterate through each keypoint and draw a cross and circle
    for kp in keypoints:
        x, y = map(int, kp.pt)
        scale = int(kp.size)
        angle = kp.angle

        # Draw a cross ("+")
        cv2.line(image_with_keypoints, (x - scale, y), (x + scale, y), (0, 0, 255), 2)
        cv2.line(image_with_keypoints, (x, y - scale), (x, y + scale), (0, 0, 255), 2)

        # Draw a circle around the keypoint
        cv2.circle(image_with_keypoints, (x, y), scale, (0, 255, 0), 2)

        # Draw a line indicating the orientation
        angle_rad = np.deg2rad(angle)
        line_length = int(scale * 0.8)
        line_x = int(x + line_length * np.cos(angle_rad))
        line_y = int(y + line_length * np.sin(angle_rad))
        cv2.line(image_with_keypoints, (x, y), (line_x, line_y), (255, 0, 0), 2)

    # Concatenate the original image and the image with keypoints horizontally
    result_image = np.hstack((image, image_with_keypoints))

    # Display the concatenated image
    cv2.imshow('Original vs. Keypoints', result_image) 

    # Access the keypoints and descriptors for further processing
    print(f"Number of keypoints detected: {len(keypoints)}")
    print(f"Descriptor shape: {descriptors.shape}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to resize an image while maintaining aspect ratio
def resize_image(img, target_width, target_height):
            # Calculate the aspect ratio
            aspect_ratio = img.shape[1] / img.shape[0]

            # Determine the new size while preserving the aspect ratio
            if aspect_ratio > (target_width / target_height):
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            else:
                new_height = target_height
                new_width = int(target_height * aspect_ratio)

            # Resize the image to the new size
            resized_img = cv2.resize(img, (new_width, new_height))
            
            return resized_img

#Calculate total keypoints and descriptors and print them
def TotalKeypointsDescriptors(siftImagesList):
    total_keypoints = 0
    total_descriptors = []
    for sift_image in siftImagesList:
        num_keypoints = len(sift_image.keyPoints)
        print(f"# of keypoints in {sift_image.name} is {num_keypoints}")
        total_keypoints += num_keypoints
        total_descriptors.extend(sift_image.descriptors)
    
    return total_keypoints, total_descriptors

    print(f"Total number of keypoints of all images is {total_keypoints}")
    print("\n")
def run_task1():
    image_path = sys.argv[1]
    original_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Define the target VGA resolution
    vga_rows = 480
    vga_columns = 600
    # Resize the image to the new dimensions using OpenCV
    rescaled_image = resize_image(original_image, vga_rows, vga_columns)

    visualize_keypoints(rescaled_image)

def run_task2(display):
    siftImages = []

    for i in range(1, len(sys.argv)):
        image_path = sys.argv[i]
        image = cv2.imread(image_path)
        siftImg = SiftImage()
        siftImg.name = get_file_name(image_path)
        siftImg.image = resize_image(image, 480, 600)
        siftImg.keyPoints, siftImg.descriptors = extract_sift_luminance(siftImg.image)

        siftImages.append(siftImg)
    
    total_keypoints, total_descriptors = TotalKeypointsDescriptors(siftImages)
    #display- can be ignored
    if(display):
        for s in siftImages:
            # Draw the keypoints on the original image
            image_with_keypoints = cv2.drawKeypoints(s.image, s.keyPoints, None)
            cv2.imshow(f"resized {s.name} with keypoints", image_with_keypoints) 

    # cluster SIFT descriptors
    cluster_percentages = [0.05, 0.10, 0.20]
    
    
    for cluster_percentage in cluster_percentages:
        num_clusters = int(cluster_percentage * total_keypoints)
        #define clustering criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        #perform clustering
        compactness,labels,centers = cv2.kmeans(np.array(total_descriptors), num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
        #histogram of the occurrence of the visual words
        
        histograms = []
        num_images = len(siftImages)
        comparison_matrix = np.zeros((num_images, num_images))
        for siftImg in siftImages:
            histogram = construct_histogram(siftImg, centers, num_clusters)
            siftImg.histogram = histogram
            histograms.append(histogram)
        # Visualize histograms- can be ignored
        if(display):
            for i, hist in enumerate(np.array(histograms)):
                plt.bar(range(num_clusters), hist)
                plt.xlabel("Visual Word")
                plt.ylabel("Frequency")
                plt.title(f"Histogram for Image {i+1} and cluster percentage {cluster_percentage}- count {num_clusters}")
                plt.show()
        
        #calculate X^2 distance between the normalized histograms
        for i in range(len(siftImages)):
            for j in range(i + 1, len(siftImages)):
                hist1 = siftImages[i].histogram.astype(np.float32)
                hist2 = siftImages[j].histogram.astype(np.float32)

                # Normalize histograms (ensure they sum to 1)
                hist1 /= hist1.sum()
                hist2 /= hist2.sum()

                result = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR_ALT)

                comparison_matrix[i, j] = result
                comparison_matrix[j, i] = result 

        print(f"K={cluster_percentage*100}%* (total number of keypoionts)={num_clusters}")
        print("Dissimilarity Matrix")

        image_names = [sift_image.name.split('.')[0] + " "*3 for sift_image in siftImages]

        # Join the items with a fixed width of 8 characters each
        formatted_names = "{:<8}".format("") + "".join(["{:<8}".format(name) for name in image_names])
        print(formatted_names)

        # Display the comparison matrix with names
        for i, sift_image in enumerate(siftImages):
            row_data = "{:<8}".format( f"{sift_image.name.split('.')[0]}")
            for j in range(len(siftImages)):
                if(i>j):
                    row_data += "{:<8}".format("")  # Leave empty space for diagonal
                else:
                    result = comparison_matrix[i, j]
                    row_data += "{:<8}".format("{:.2f}".format(result))
            print(row_data)
        print("\n")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()





 