import cv2
import numpy as np

def create_watermark_text(text="WATERMARK", size=(100, 100), font_scale=0.5, color=255):
    # Create a blank image
    watermark = np.zeros(size, dtype=np.uint8)
    
    # Define font, scale, thickness, and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (watermark.shape[1] - text_size[0]) // 2
    text_y = (watermark.shape[0] + text_size[1]) // 2
    
    # Put the text on the image
    cv2.putText(watermark, text, (text_x, text_y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    
    return watermark

# Generate and save the watermark
watermark_text = create_watermark_text("WATERMARK", size=(100, 100), font_scale=0.6)
cv2.imwrite("watermark.png", watermark_text)
