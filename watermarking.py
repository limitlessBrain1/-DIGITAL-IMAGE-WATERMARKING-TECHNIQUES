import numpy as np
import pywt
import cv2
from scipy.linalg import svd
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WatermarkEmbedder:
    def __init__(self, config):
        self.config = config

    def embed_watermark(self, image: np.ndarray, watermark: np.ndarray) -> (np.ndarray, np.ndarray): # type: ignore
        """Embed watermark into image using DWT-SVD and return both grayscale and color watermarked images."""
        logging.info("Starting watermark embedding process...")

        # Resize watermark to match image size if necessary
        if image.shape[:2] != watermark.shape[:2]:
            watermark = cv2.resize(watermark, (image.shape[1], image.shape[0]))

        # Pad image and watermark to be multiples of the block size
        padded_image, padded_watermark = self._pad_to_block_size(image, watermark)

        # Split the padded image and watermark into blocks
        blocks = self._split_into_blocks(padded_image, padded_watermark)

        # Process each block
        embedded_blocks = []
        for block in tqdm(blocks, desc="Processing blocks"):
            embedded_blocks.append(self._process_block(block))

        # Reconstruct image from blocks
        watermarked_image_padded = self._reconstruct_from_blocks(embedded_blocks, padded_image.shape)

        # Unpad the watermarked image to the original size
        watermarked_image = watermarked_image_padded[:image.shape[0], :image.shape[1]]
        
        # Add yellow tint to watermark regions for a color version
        watermarked_image_colored = self._add_yellow_tint(image, watermarked_image, watermark)

        logging.info("Watermark embedding completed.")
        return watermarked_image, watermarked_image_colored

    def _pad_to_block_size(self, image: np.ndarray, watermark: np.ndarray) -> tuple:
        """Pad image and watermark to be multiples of the block size."""
        block_size = self.config['block_size']
        pad_h = (block_size - image.shape[0] % block_size) % block_size
        pad_w = (block_size - image.shape[1] % block_size) % block_size
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant')
        padded_watermark = np.pad(watermark, ((0, pad_h), (0, pad_w)), mode='constant')
        return padded_image, padded_watermark

    def _split_into_blocks(self, image: np.ndarray, watermark: np.ndarray) -> list:
        """Split the image and watermark into blocks."""
        block_size = self.config['block_size']
        blocks = []

        for i in range(0, image.shape[0], block_size):
            for j in range(0, image.shape[1], block_size):
                image_block = image[i:i + block_size, j:j + block_size]
                watermark_block = watermark[i:i + block_size, j:j + block_size]
                blocks.append((image_block, watermark_block, self.config['alpha']))

        return blocks

    def _process_block(self, block_data: tuple) -> np.ndarray:
        """Process individual image block with DWT-SVD."""
        block, watermark_block, alpha = block_data

        # Resize the watermark block to match the image block if necessary
        if block.shape != watermark_block.shape:
            watermark_block = cv2.resize(watermark_block, (block.shape[1], block.shape[0]))

        # Apply DWT to block
        coeffs = pywt.wavedec2(block, self.config['wavelet'], level=self.config['level'])
        LL, (LH, HL, HH) = coeffs

        # Apply SVD to both image block and watermark block
        U_i, S_i, V_i = svd(LL, full_matrices=False)
        U_w, S_w, V_w = svd(watermark_block, full_matrices=False)

        # Ensure the shapes of the singular values are the same
        min_len = min(S_i.shape[0], S_w.shape[0])
        S_new = S_i[:min_len] + alpha * S_w[:min_len]

        # Reconstruct block
        LL_modified = np.dot(U_i[:, :min_len], np.dot(np.diag(S_new), V_i[:min_len, :]))
        coeffs_modified = (LL_modified, (LH, HL, HH))

        return pywt.waverec2(coeffs_modified, self.config['wavelet'])

    def _reconstruct_from_blocks(self, blocks: list, image_shape: tuple) -> np.ndarray:
        """Reconstruct the image from blocks."""
        block_size = self.config['block_size']
        reconstructed_image = np.zeros(image_shape, dtype=np.float64)

        block_idx = 0
        for i in range(0, image_shape[0], block_size):
            for j in range(0, image_shape[1], block_size):
                reconstructed_image[i:i + block_size, j:j + block_size] = blocks[block_idx]
                block_idx += 1

        return reconstructed_image

    def _add_yellow_tint(self, original_image: np.ndarray, watermarked_image: np.ndarray, watermark: np.ndarray) -> np.ndarray:
        """Convert grayscale watermarked image to RGB and add yellow tint to watermark regions."""
        # Convert the grayscale watermarked image to RGB
        watermarked_image_colored = cv2.cvtColor(np.uint8(watermarked_image), cv2.COLOR_GRAY2BGR)
        
        # Resize watermark to match watermarked image size
        watermark_resized = cv2.resize(watermark, (watermarked_image.shape[1], watermarked_image.shape[0]))
        
        # Create a mask where the watermark exists
        _, watermark_mask = cv2.threshold(watermark_resized, 1, 255, cv2.THRESH_BINARY)
        
        # Apply yellow color where the watermark mask is present
        watermarked_image_colored[watermark_mask > 0] = [0, 255, 255]  # RGB for yellow
        
        return watermarked_image_colored

# Configuration for watermark embedding
config = {
    'block_size': 8,     # Size of each block (e.g., 8x8)
    'alpha': 0.1,        # Strength of watermark embedding
    'wavelet': 'haar',   # Wavelet to use for DWT
    'level': 1           # Level of DWT decomposition
}

# Example usage
if __name__ == "__main__":
    # Load image and watermark
    image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
    watermark = cv2.imread('watermark.png', cv2.IMREAD_GRAYSCALE)

    if image is None or watermark is None:
        logging.error("Image or watermark not found.")
        exit()

    # Instantiate WatermarkEmbedder
    embedder = WatermarkEmbedder(config)

    # Embed watermark and get both versions (grayscale and yellow-tinted)
    watermarked_image_grayscale, watermarked_image_colored = embedder.embed_watermark(image, watermark)

    # Save the watermarked images
    cv2.imwrite('watermarked_image_grayscale.png', np.uint8(watermarked_image_grayscale))
    cv2.imwrite('watermarked_image_colored.png', watermarked_image_colored)
    logging.info("Watermarked images saved as 'watermarked_image_grayscale.png' and 'watermarked_image_colored.png'.")
