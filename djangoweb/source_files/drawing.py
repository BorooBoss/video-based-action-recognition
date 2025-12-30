import re
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


# --- 1. PaliGemma Output Parsing Function ---

def parse_paligemma_output(model_output: str, image_width: int, image_height: int) -> list:
    """
    Parses PaliGemma's text output to extract bounding boxes and labels.
    """

    # Regex pattern to capture four <locXXXX> tokens followed by a label (word)
    loc_pattern = r"(?:detect\s+[^:]*:\s*)?(<loc\d{4}><loc\d{4}><loc\d{4}><loc\d{4}>\s+[^<]+)"

    # Find all sequences of four <locXXXX> tokens and the following label
    detections = re.findall(loc_pattern, model_output, re.DOTALL)

    parsed_results = []

    # Normalization factor for PaliGemma's coordinates
    NORM_FACTOR = 1024

    for detection_str in detections:
        # Extract the four-digit numbers from the <loc> tokens
        coords_str = re.findall(r"<loc(\d{4})>", detection_str)

        if len(coords_str) != 4:
            continue  # Skip malformed output

        # Extract the label (the text following the last <loc> token)
        label_match = re.search(r">\s*([^<]+)$", detection_str)
        label = label_match.group(1).strip() if label_match else "unknown"

        # Convert to integer coordinates and reorder to (y_min, x_min, y_max, x_max)
        y_min_norm, x_min_norm, y_max_norm, x_max_norm = map(int, coords_str)

        # Scale to original image dimensions (convert normalized to pixel values)
        y_min = int(y_min_norm * image_height / NORM_FACTOR)
        x_min = int(x_min_norm * image_width / NORM_FACTOR)
        y_max = int(y_max_norm * image_height / NORM_FACTOR)
        x_max = int(x_max_norm * image_width / NORM_FACTOR)

        # Ensure the box coordinates are in the standard PIL/Matplotlib format (x_min, y_min, x_max, y_max)
        pixel_box = (x_min, y_min, x_max, y_max)

        parsed_results.append({
            'box': pixel_box,
            'label': label
        })

    return parsed_results


# --- 2. Visualization Function (Modified to include Black Border) ---

def visualize_detections(image_path: str, detections: list):
    """
    Displays the image and draws bounding boxes on it, with a black border
    around the entire displayed image.
    """
    try:
        # Open the image using PIL
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return

    # Use Matplotlib to display the image and drawing patches
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)
    ax.axis('off')  # Hide axes ticks and labels

    # Get image dimensions for the border
    image_width = img.width
    image_height = img.height

    # **ADD BLACK BORDER:** Draw a Rectangle patch around the entire image area.
    border_rect = plt.Rectangle(
        (0, 0),  # Top-left corner (x, y)
        image_width,
        image_height,
        linewidth=4,
        edgecolor='black',
        facecolor='none',
        zorder=100  # Ensures the border is visible on top
    )
    ax.add_patch(border_rect)

    for det in detections:
        # Coordinates are (x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = det['box']
        label = det['label']

        # Calculate width and height for the matplotlib Rectangle patch
        width = x_max - x_min
        height = y_max - y_min

        # Create a Rectangle patch for the detection
        rect = plt.Rectangle(
            (x_min, y_min),  # Anchor point (x, y) - Top-left corner
            width,
            height,
            linewidth=2,
            edgecolor='red',  # Color of the box
            facecolor='none'
        )

        # Add the patch to the Axes
        ax.add_patch(rect)

        # Add the label above the bounding box
        ax.text(
            x_min,
            y_min - 10,  # Position the text slightly above the box
            label,
            color='red',
            fontsize=12,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
        )

    plt.title("PaliGemma Object Detections")
    plt.show()


# --- 3. Example Usage ---

if __name__ == "__main__":
    # --- STEP 1: Define Image and Mock PaliGemma Output ---

    # ⚠️ REPLACE 'path/to/your/image.jpg' with an actual image path
    IMAGE_FILE = 'temp_dummy_image.jpg'  # Changed to a local file name for better compatibility

    # Create a dummy image file for demonstration
    try:
        # Create an 800x600 white image
        img = Image.new('RGB', (800, 600), color='white')

        # Add a simple shape so it's not totally blank
        draw = ImageDraw.Draw(img)
        # Draw a grey rectangle in the middle
        draw.rectangle([200, 150, 600, 450], fill='lightgrey', outline='darkgrey', width=2)

        img.save(IMAGE_FILE)
        print(f"Created dummy image at {IMAGE_FILE}")
    except Exception as e:
        print(f"Could not create dummy image: {e}")
        exit()  # Exit if we can't create the file for the demonstration

    image_width, image_height = img.size

    # Example PaliGemma output for detecting a 'giraffe'.
    # Coordinates are y_min, x_min, y_max, x_max (normalized to 1024)
    # <loc0269> (y_min) <loc0353> (x_min) <loc0685> (y_max) <loc0475> (x_max)
    MODEL_OUTPUT = (
        "detect giraffe: <loc0269><loc0353><loc0685><loc0475> giraffe"
    )

    print(f"\n--- PaliGemma Output ---\n{MODEL_OUTPUT}")

    # --- STEP 2: Parse the Output ---
    detections = parse_paligemma_output(MODEL_OUTPUT, image_width, image_height)

    print("\n--- Parsed Pixel Coordinates ---")
    for det in detections:
        print(f"Label: {det['label']:<10} | Box (x_min, y_min, x_max, y_max): {det['box']}")

    # --- STEP 3: Visualize the Detections (Now with Black Border) ---
    visualize_detections(IMAGE_FILE, detections)