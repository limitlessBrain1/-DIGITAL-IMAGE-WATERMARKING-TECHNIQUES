import numpy as np
import cv2
import os

def create_test_images(output_dir):
    """
    Creates test images suitable for digital watermarking:
    1. A grayscale host image with good contrast and detail
    2. A simple watermark with clear patterns
    """
    
    if not os.path.exists(r"C:\Users\raman\Desktop\cvfa2"):
        os.makedirs(r"C:\Users\raman\Desktop\cvfa2")
    
    # Create host image (512x512 for optimal DWT processing)
    host_image = np.zeros((512, 512), dtype=np.uint8)
    
    # Add gradient background
    for i in range(512):
        for j in range(512):
            host_image[i, j] = (i + j) // 4 % 256
    
    # Add some geometric patterns for visual interest
    cv2.circle(host_image, (256, 256), 100, 200, -1)
    cv2.rectangle(host_image, (100, 100), (400, 400), 150, 3)
    cv2.line(host_image, (0, 0), (512, 512), 250, 2)
    
    # Create watermark (256x256, smaller than host image)
    watermark = np.zeros((256, 256), dtype=np.uint8)
    
    # Add text as watermark
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(watermark, 'TEST', (50, 128), font, 4, 255, 4)
    cv2.putText(watermark, 'WATERMARK', (20, 200), font, 2, 255, 4)
    
    # Add border to watermark
    cv2.rectangle(watermark, (10, 10), (245, 245), 255, 2)
    
    # Save images
    host_path = os.path.join(output_dir, 'test_host_image.png')
    watermark_path = os.path.join(output_dir, 'test_watermark.png')
    
    cv2.imwrite(host_path, host_image)
    cv2.imwrite(watermark_path, watermark)
    
    print(f"Host image saved to: {host_path}")
    print(f"Watermark image saved to: {watermark_path}")
    print(f"Host image shape: {host_image.shape}")
    print(f"Watermark shape: {watermark.shape}")
    
    return host_path, watermark_path

def test_watermarking_process(host_path, watermark_path):
    """
    Test the watermarking process with the generated images
    """
    # Import the watermarking functions
    from your_watermark_module import embed_watermark, extract_watermark
    
    output_dir = os.path.dirname(host_path)
    watermarked_path = os.path.join(output_dir, 'watermarked_output.png')
    extracted_path = os.path.join(output_dir, 'extracted_watermark.png')
    
    try:
        # Embed watermark
        S_base = embed_watermark(host_path, watermark_path, watermarked_path)
        
        # Extract watermark
        extract_watermark(watermarked_path, S_base, extracted_path)
        
        print("Watermarking process completed successfully!")
        
    except Exception as e:
        print(f"Error during watermarking process: {str(e)}")

# Create test directory and generate images
output_directory = "watermark_test_images"
host_image_path, watermark_path = create_test_images(output_directory)