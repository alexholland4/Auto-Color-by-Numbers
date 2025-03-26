import cv2
import numpy as np
from collections import Counter

def remove_small_components(segmentation_mask, min_size=500):
    """
    Merge small connected components (fewer than min_size pixels) into
    their most frequently adjacent label.
    """
    new_mask = segmentation_mask.copy()
    unique_labels = np.unique(new_mask)

    for lab in unique_labels:
        region_mask = (new_mask == lab).astype(np.uint8)
        num_comps, comp_labels = cv2.connectedComponents(region_mask)
        for comp_id in range(1, num_comps):
            comp_pixels = np.where(comp_labels == comp_id)
            comp_size = len(comp_pixels[0])
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

def auto_color_by_numbers(
    image,
    threshold_option='normal',
    min_region_size=500,
    downsample_factor=1,
    apply_morph_open=False
):
    """
    Process an image to generate a white canvas with black outlines and numbered regions.
    Incorporates:
      - Downsampling for speed.
      - Gaussian blurring.
      - K-means clustering (with K=8).
      - Post-processing to merge similar clusters.
      - Removal of small regions.
      - Improved label placement using a distance transform.
    
    Returns:
      output_img: White image with outlines and labels.
      master_list: Dictionary mapping region label (as string) to average RGB color.
      final_mask: Final segmentation mask (for further processing).
    """
    
    print("[INFO] Starting 'auto_color_by_numbers'...")

    # Optional downsampling
    if downsample_factor > 1:
        print(f"[INFO] Downsampling image by factor of {downsample_factor}...")
        h, w, _ = image.shape
        new_h, new_w = h // downsample_factor, w // downsample_factor
        image_small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        image_small = image.copy()

    print("[INFO] Applying Gaussian blur...")
    blurred = cv2.GaussianBlur(image_small, (7, 7), 0)
    
    print("[INFO] Converting to HSV color space...")
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    h, w, _ = hsv.shape
    data = hsv.reshape((-1, 3)).astype(np.float32)

    # K-Means Clustering with K=8
    K = 12
    print(f"[INFO] Running k-means with K={K}...")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.flatten()

    print("[INFO] Merging similar cluster centers...")
    thresholds = {'basic': 50.0, 'normal': 30.0, 'detailed': 15.0}
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
    
    if apply_morph_open:
        print("[INFO] Applying morphological opening to reduce specks...")
        kernel = np.ones((3,3), np.uint8)
        unique_labels = np.unique(segmentation_mask)
        morph_mask = segmentation_mask.copy()
        for lab in unique_labels:
            lab_mask = (morph_mask == lab).astype(np.uint8)*255
            lab_mask_opened = cv2.morphologyEx(lab_mask, cv2.MORPH_OPEN, kernel)
            difference = cv2.subtract(lab_mask, lab_mask_opened)
            removed_pixels = np.where(difference > 0)
            morph_mask[removed_pixels] = -1
        removed_coords = np.where(morph_mask == -1)
        for i, j in zip(removed_coords[0], removed_coords[1]):
            neighbors = []
            for di in [-1,0,1]:
                for dj in [-1,0,1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i+di, j+dj
                    if 0 <= ni < h and 0 <= nj < w:
                        if morph_mask[ni, nj] != -1:
                            neighbors.append(morph_mask[ni, nj])
            if neighbors:
                morph_mask[i, j] = Counter(neighbors).most_common(1)[0][0]
            else:
                morph_mask[i, j] = 0
        segmentation_mask = morph_mask

    print("[INFO] Removing/merging small connected components...")
    segmentation_mask = remove_small_components(segmentation_mask, min_region_size)
    unique_final = np.unique(segmentation_mask)
    remap_dict = {val: idx for idx, val in enumerate(unique_final)}
    final_mask = np.vectorize(remap_dict.get)(segmentation_mask)
    
    print("[INFO] Building master color list...")
    master_list = {}
    for label_val in np.unique(final_mask):
        mask = final_mask == label_val
        avg_hsv = np.mean(hsv[mask], axis=0)
        avg_hsv_uint8 = np.array(avg_hsv, dtype=np.uint8).reshape((1, 1, 3))
        avg_bgr = cv2.cvtColor(avg_hsv_uint8, cv2.COLOR_HSV2BGR)[0][0]
        avg_rgb = (int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0]))
        master_list[str(label_val + 1)] = avg_rgb

    print("[INFO] Generating final output image with contours and labels...")
    output_img = np.full((h, w, 3), 255, dtype=np.uint8)
    
    for label_val in np.unique(final_mask):
        region_mask = (final_mask == label_val).astype(np.uint8) * 255
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output_img, contours, -1, (0, 0, 0), 2)
        
        # For each contour, use distance transform for optimal label placement.
        for cnt in contours:
            # Create a filled mask for the contour.
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
            dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            # Find the point with maximum distance (i.e. furthest from the edge)
            _, _, _, maxLoc = cv2.minMaxLoc(dist_transform)
            cv2.putText(output_img, str(label_val + 1), maxLoc,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    
    if downsample_factor > 1:
        original_h, original_w, _ = image.shape
        print("[INFO] Upscaling result back to original size...")
        output_img = cv2.resize(output_img, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    
    print("[INFO] Finished processing.")
    return output_img, master_list, final_mask

def generate_test_colored_image(final_mask, master_list):
    """
    Generate a test image by filling each region in the segmentation mask
    with its corresponding color from the master list.
    """
    h, w = final_mask.shape
    colored_img = np.zeros((h, w, 3), dtype=np.uint8)
    for label_val in np.unique(final_mask):
        color = master_list[str(label_val + 1)]
        colored_img[final_mask == label_val] = color
    return colored_img

# Example usage:
if __name__ == "__main__":
    input_path = "test_input.JPEG"  # Replace with your image path
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Image not found. Please check the file path.")
    
    print("[INFO] Running 'auto_color_by_numbers' on:", input_path)
    output_img, master_list, final_mask = auto_color_by_numbers(
        image,
        threshold_option='normal',     # Options: 'basic', 'normal', 'detailed'
        min_region_size=1000,           # Adjust to control minimal region size
        downsample_factor=3,            # Increase to speed up processing
        apply_morph_open=False          # Set True to apply morphological opening
    )

    output_path = "output_numbered.png"
    cv2.imwrite(output_path, output_img)
    print(f"[INFO] Labeled output saved to {output_path}")

    test_colored_img = generate_test_colored_image(final_mask, master_list)
    test_colored_path = "output_colored.png"
    cv2.imwrite(test_colored_path, test_colored_img)
    print(f"[INFO] Test colored image saved to {test_colored_path}")

    print("Master List (Region Label -> Average RGB):")
    for label, color in master_list.items():
        print(f"  {label}: {color}")
    
    # Uncomment for local display:
    # cv2.imshow("Output with Labels", output_img)
    # cv2.imshow("Test Colored Image", test_colored_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
