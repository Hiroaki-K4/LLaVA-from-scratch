"""
LLaVA-Instruct-150K Image Data Acquisition

Images in this dataset primarily come from the COCO dataset.
The following methods can be used to obtain image data.
"""

import io
from typing import Optional

import requests
from PIL import Image


class LLaVAImageDownloader:
    def __init__(self):
        """
        Image downloader for LLaVA-Instruct dataset
        """
        # COCO dataset configuration
        self.coco_urls = {
            "train2017": "http://images.cocodataset.org/train2017/",
            "val2017": "http://images.cocodataset.org/val2017/",
            "test2017": "http://images.cocodataset.org/test2017/",
        }

    def download_image_by_filename(
        self, filename: str, max_retries: int = 3
    ) -> Optional[Image.Image]:
        """
        Download image by filename (without saving locally)

        Args:
            filename: Image filename (e.g., "000000033471.jpg")
            max_retries: Maximum retry attempts

        Returns:
            PIL Image or None (if failed)
        """
        # Try downloading directly from COCO dataset
        for split in self.coco_urls:
            url = self.coco_urls[split] + filename

            for attempt in range(max_retries):
                try:
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        image = Image.open(io.BytesIO(response.content))
                        return image

                except Exception as e:
                    print(f"Download error (attempt {attempt+1}): {e}")

        print(f"Failed to download image: {filename}")
        return None
