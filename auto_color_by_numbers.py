import cv2
import numpy as np
from collections import Counter

def remove_small_components(segmentation_mask, min_size=500):
    """
    Merge small connected components (fewer than min_size pixels) into
    their most frequently adjacent label.
    
    segmentation_mask: 2D array of shape (H, W) with integer labels.
    min_size: minimum component size in pixels to keep as-is.
    
    Returns: A new mask with small regions merged.
    """
    new_mask = segmentation_mask.copy()
    unique_labels = np.unique(new_mask)

    for lab in unique_labels:
        # Binary mask for the current label
        region_mask = (new_mask == lab).astype(np.uint8)
        
        # Identify connected components within this label
        num_comps, comp_labels = cv2.connectedComponents(region_mask)
        # comp_labels has values in [0..num_comps-1]; 0 is background

        for comp_id in range(1, num_comps):
            # Pixels belonging to this connected component
            comp_pixels = np.where(comp_labels == comp_id)
            comp_size = len(comp_pixels[0])
            
            # If component is too small, merge it into a neighbor
            if comp_size < min_size:
                neighbor_labels = []
                for i, j in zip(comp_pixels[0], comp_pixels[1]):
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < new_mask.shape[0] and 0 <= nj < new_mask.shape[1]:
                                neighbor_lab = new_mask[ni, nj]
                                if neighbor_lab != lab:
                                    neighbor_labels.append(neighbor_lab)
                
                if neighbor_labels:
                    most_common_label = Counter(neighbor_labels).most_common(1)[0][0]
                    new_mask[comp_pixels] = most_common_label

    return new_mask

def auto_color_by_numbers(image, threshold_option='normal', min_region_size=500):
    """
    Process an image to generate a white canvas with black outlines and
    numbered regions. Combines blurring, k-means clustering, color-distance
    thresholding, and a post-processing step to merge small regions.
    
    Parameters:
      image (np.ndarray): Input image in BGR format.
      threshold_option (str): One of 'basic', 'normal', or 'detailed' that
                              controls merge sensitivity.
      min_region_size (int): Minimum connected component size (in pixels).
    
    Returns:
      output_img (np.ndarray): White image with black region outlines and overlaid numbers.
      master_list (dict): Mapping from region label (as string) to the average RGB color of that region.
    """

    # 1. Blur to unify small color differences
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # 2. Convert to HSV for clustering
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    h, w, _ = hsv.shape
    data = hsv.reshape((-1, 3)).astype(np.float32)
    
    # 3. Run k-means with fewer clusters (K=8)
    K = 8
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.flatten()
    
    # 4. Post-cluster merging (Union-Find) based on HSV distance
    # Adjust thresholds for "basic", "normal", "detailed"
    thresholds = {
        'basic': 50.0,     # merges more aggressively
        'normal': 30.0,
        'detailed': 15.0
    }
    threshold_value = thresholds.get(threshold_option, 30.0)
    
    parent = list(range(K))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx
    
    for i in range(K):
        for j in range(i + 1, K):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < threshold_value:
                union(i, j)
    
    # Map original cluster labels to merged labels
    merged_mapping = {}
    new_label = 0
    for i in range(K):
        root = find(i)
        if root not in merged_mapping:
            merged_mapping[root] = new_label
            new_label += 1
        merged_mapping[i] = merged_mapping[root]
    
    merged_labels = np.array([merged_mapping[label] for label in labels])
    segmentation_mask = merged_labels.reshape((h, w))
    
    # 5. Remove or merge small connected components
    segmentation_mask = remove_small_components(segmentation_mask, min_region_size)
    
    # After merging small components, re-map labels to a compact range [0..n-1]
    unique_final = np.unique(segmentation_mask)
    remap_dict = {}
    for idx, val in enumerate(unique_final):
        remap_dict[val] = idx
    final_mask = np.vectorize(remap_dict.get)(segmentation_mask)
    
    # 6. Build master list (label -> average color in RGB)
    master_list = {}
    for label_val in np.unique(final_mask):
        mask = final_mask == label_val
        avg_hsv = np.mean(hsv[mask], axis=0)
        avg_hsv_uint8 = np.array(avg_hsv, dtype=np.uint8).reshape((1, 1, 3))
        avg_bgr = cv2.cvtColor(avg_hsv_uint8, cv2.COLOR_HSV2BGR)[0][0]
        avg_rgb = (int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0]))
        # Number regions starting at 1 for display
        master_list[str(label_val + 1)] = avg_rgb
    
    # 7. Create a blank white output image
    output_img = np.full((h, w, 3), 255, dtype=np.uint8)
    
    # 8. For each region, find contours and draw black outlines + region number
    for label_val in np.unique(final_mask):
        region_mask = (final_mask == label_val).astype(np.uint8) * 255
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours
        cv2.drawContours(output_img, contours, -1, (0, 0, 0), 2)
        
        # Overlay label number at each contour's centroid
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(output_img, str(label_val + 1), (cX, cY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    
    return output_img, master_list

# Example usage:
if __name__ == "__main__":
    input_path = "test_input.JPEG"  # Replace with your image path
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Image not found. Please check the file path.")

    output_img, master_list = auto_color_by_numbers(
        image,
        threshold_option='basic',     # 'basic', 'normal', or 'detailed'
        min_region_size=1000         # Adjust to control how big regions must be
    )

    cv2.imwrite("output_numbered.png", output_img)
    print("Master List (Region Label -> Average RGB):")
    for label, color in master_list.items():
        print(f"  {label}: {color}")
    
    # Uncomment for local display
    # cv2.imshow("Auto Color by Numbers", output_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
