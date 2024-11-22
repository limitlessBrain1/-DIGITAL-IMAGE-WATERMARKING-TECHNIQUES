from logging import config
from watermarking import WatermarkProcessor


processor = WatermarkProcessor(config)
metadata = processor.embed_watermark('input.png', 'watermark.png', 'output.png')
processor.extract_watermark('output.png', 'extracted.png', 'output.png.json')