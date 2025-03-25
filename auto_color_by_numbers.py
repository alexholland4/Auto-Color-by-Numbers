import cv2
import numpy as np

def auto_color_by_numbers(image, threshold_option='normal'):
    """
    Process an image to generate a white canvas with black outlines and numbered regions.
    
    Parameters:
      image (np.ndarray): Input image in BGR format.
      threshold_option (str): One of 'basic', 'normal', or 'detailed' that controls merge sensitivity.
      
    Returns:
      output_img (np.ndarray): White image with black region outlines and overlaid numbers.
      master_list (dict): Mapping from region label (as string) to the average RGB color of that region.
    """
    
    # Convert image from BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w, c = hsv.shape
    data = hsv.reshape((-1, 3)).astype(np.float32)
    
    # Set number of initial clusters (we use a higher number and then merge similar clusters)
    K = 16
    # Define criteria and run k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.flatten()  # shape (num_pixels,)
    
    # Define thresholds (Euclidean distance in HSV space) based on user option
    thresholds = {
        'basic': 30.0,    # merge clusters with a larger difference => fewer regions
        'normal': 20.0,
        'detailed': 10.0  # lower threshold => more detailed segmentation
    }
    threshold_value = thresholds.get(threshold_option, 20.0)
    
    # Post-process clusters: merge clusters with centers closer than the threshold.
    # We'll use a simple union-find algorithm.
    parent = list(range(K))
    
    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    
    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x
    
    # Compare each pair of cluster centers in HSV space
    for i in range(K):
        for j in range(i + 1, K):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < threshold_value:
                union(i, j)
    
    # Map original cluster labels to new merged labels
    merged_mapping = {}
    new_label = 0
    for i in range(K):
        root = find(i)
        if root not in merged_mapping:
            merged_mapping[root] = new_label
            new_label += 1
        merged_mapping[i] = merged_mapping[root]
    
    # Create the merged segmentation mask from the clustering result
    merged_labels = np.array([merged_mapping[label] for label in labels])
    segmentation_mask = merged_labels.reshape((h, w))
    
    # Build the master list: for each unique label, compute the average color (convert to RGB for display)
    master_list = {}
    for label in np.unique(segmentation_mask):
        mask = segmentation_mask == label
        avg_hsv = np.mean(hsv[mask], axis=0)
        # Convert average HSV color to BGR then to RGB
        avg_hsv_uint8 = np.array(avg_hsv, dtype=np.uint8).reshape((1,1,3))
        avg_bgr = cv2.cvtColor(avg_hsv_uint8, cv2.COLOR_HSV2BGR)[0][0]
        avg_rgb = (int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0]))
        # Number regions starting at 1 for display purposes
        master_list[str(label + 1)] = avg_rgb

    # Create a blank white output image (same size as input, 3 channels)
    output_img = np.full((h, w, 3), 255, dtype=np.uint8)
    
    # For each region, find contours and draw them with black outlines.
    # Also compute centroids to overlay the region number.
    for label in np.unique(segmentation_mask):
        # Create a binary mask for the current label
        region_mask = np.uint8(segmentation_mask == label) * 255
        # Find external contours
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours in black (thickness 2)
        cv2.drawContours(output_img, contours, -1, (0, 0, 0), 2)
        
        # For each contour, compute the centroid and put the region number.
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(output_img, str(label + 1), (cX, cY), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    
    return output_img, master_list

# Example usage:
if __name__ == "__main__":
    # Replace 'input.jpg' with the path to your test image.
    input_path = "test_input.JPEG"
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Image not found. Please check the file path.")

    # Process the image with the 'normal' threshold option
    output_img, master_list = auto_color_by_numbers(image, threshold_option='normal')
    
    # Save or display the result
    cv2.imwrite("output_numbered.png", output_img)
    print("Master List (Region Label : Average RGB):")
    for label, color in master_list.items():
        print(f"  {label}: {color}")
    
    # Optionally display the image (uncomment the lines below if running locally)
    # cv2.imshow("Auto Color by Numbers", output_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
