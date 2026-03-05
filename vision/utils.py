"""Utility functions for computer vision operations."""

import cv2
import numpy as np
from schemas import Detection
from PIL import Image, ImageDraw
from typing import Tuple, List, Optional, Dict

def load_image(image_path: str) -> Optional[np.ndarray]:
    """Load image from file.

    Args:
        image_path: Path to the image file

    Returns:
        Image as numpy array, or None if loading fails
    """
    try:
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image to target dimensions.

    Args:
        image: Input image
        target_size: Target (width, height)

    Returns:
        Resized image
    """
    return cv2.resize(image, target_size)

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image pixel values to [0, 1].

    Args:
        image: Input image

    Returns:
        Normalized image
    """
    return image.astype(np.float32) / 255.0

def draw_detections(image: np.ndarray, detections: List[Detection],
                   class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None) -> np.ndarray:
    """Draw bounding boxes and labels on image.

    Args:
        image: Input image
        detections: List of detection objects
        class_colors: Dictionary mapping class IDs to colors

    Returns:
        Image with drawn detections
    """
    if class_colors is None:
        class_colors = {
            0: (255, 0, 0),    # Red
            1: (0, 255, 0),    # Green
            2: (0, 0, 255),    # Blue
            3: (255, 255, 0), # Yellow
            4: (255, 0, 255), # Magenta
        }

    # Convert to PIL Image for easier drawing
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    for detection in detections:
        bbox = detection.bbox
        color = class_colors.get(detection.class_id, (128, 128, 128))

        # Draw bounding box
        draw.rectangle(bbox, outline=color, width=3)

        # Draw label
        label = f"{detection.label}: {detection.confidence:.2f}"
        draw.text((bbox[0], bbox[1] - 10), label, fill=color)

    return np.array(pil_image)

def calculate_iou(box1: Tuple[float, float, float, float],
                 box2: Tuple[float, float, float, float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: (x_min, y_min, x_max, y_max)
        box2: (x_min, y_min, x_max, y_max)

    Returns:
        IoU value
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Check if boxes intersect
    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection_area = (x2 - x1) * (y2 - y1)

    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0

def calculate_center_distance(box1: Tuple[float, float, float, float],
                            box2: Tuple[float, float, float, float]) -> float:
    """Calculate Euclidean distance between centers of two bounding boxes.

    Args:
        box1: (x_min, y_min, x_max, y_max)
        box2: (x_min, y_min, x_max, y_max)

    Returns:
        Distance between centers
    """
    center1_x = (box1[0] + box1[2]) / 2
    center1_y = (box1[1] + box1[3]) / 2
    center2_x = (box2[0] + box2[2]) / 2
    center2_y = (box2[1] + box2[3]) / 2

    return ((center2_x - center1_x) ** 2 + (center2_y - center1_y) ** 2) ** 0.5

def save_image_with_detections(image_path: str, detections: List[Detection],
                             output_path: str):
    """Load image, draw detections, and save result.

    Args:
        image_path: Path to input image
        detections: List of detection objects
        output_path: Path to save annotated image
    """
    image = load_image(image_path)
    if image is not None:
        annotated_image = draw_detections(image, detections)
        # Convert back to BGR for OpenCV saving
        annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, annotated_image_bgr)

def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """Get width and height of an image.

    Args:
        image_path: Path to image file

    Returns:
        (width, height) tuple
    """
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        print(f"Error getting dimensions for {image_path}: {e}")
        return 0, 0

def create_video_from_detections(image_paths: List[str], detections_list: List[List[Detection]],
                               output_path: str, fps: int = 30):
    """Create video from images with detections.

    Args:
        image_paths: List of paths to input images
        detections_list: List of detection lists for each image
        output_path: Path to save video
        fps: Frames per second
    """
    if not image_paths:
        return

    # Get dimensions from first image
    width, height = get_image_dimensions(image_paths[0])

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i, image_path in enumerate(image_paths):
        image = load_image(image_path)
        if image is not None:
            # Draw detections
            detections = detections_list[i] if i < len(detections_list) else []
            annotated_image = draw_detections(image, detections)

            # Convert to BGR and write to video
            annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            video_writer.write(annotated_image_bgr)

    video_writer.release()