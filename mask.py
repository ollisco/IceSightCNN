import cv2
import numpy as np
from PIL import Image

def read_mask(path):
    """Read an image file as a mask."""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    thresh = 127 
    _, mask = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    return mask

def calculate_masked_percentage(base_mask_path, changing_mask_path) -> float:
    """Calculate the percentage of the base mask that is covered by the changing mask."""
    base_mask = read_mask(base_mask_path)
    changing_mask = read_mask(changing_mask_path)
    
    intersection = cv2.bitwise_and(base_mask, changing_mask)
    
    base_area = np.count_nonzero(base_mask)
    intersection_area = np.count_nonzero(intersection)
    
    if base_area == 0:
        return 0.0 
    
    # Calculate the percentage of the base mask that is covered by the changing mask
    percentage_covered = (intersection_area / base_area) * 100
    return percentage_covered


if __name__ == '__main__':
    base_mask_path = 'path_to_base_mask.png'
    changing_mask_path = 'path_to_changing_mask.png'
    percentage = calculate_masked_percentage(base_mask_path, changing_mask_path)
    print(f"Percentage of base mask covered by changing mask: {percentage:.2f}%")

