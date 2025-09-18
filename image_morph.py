import cv2
import numpy as np
import time
import os
from typing import Tuple, List, Optional, Union
import colorsys
import imageio
from PIL import Image, ImageDraw, ImageFont

class ColoredOpenCVPixelGenerator:
    """
    Generates colored pixel art from real images using OpenCV.
    The output is a grid of solid color pixels, preserving the image's colors
    and general shape in a pixelated format. It supports morphing between two
    images and saving the result as a GIF.
    """
    
    def __init__(self, width: int = 120, height: int = 40):
        """
        Initialize the colored pixel generator.
        
        Args:
            width: Output pixel grid width.
            height: Output pixel grid height.
        """
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive integers.")
        
        self.width = width
        self.height = height
        
        # Grid to store pixel colors
        self.color_grid = [[(255, 255, 255) for _ in range(width)] for _ in range(height)]
        
    def get_proper_dimensions(self, original_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculate proper dimensions for resizing while maintaining aspect ratio.
        
        Args:
            original_shape: (height, width) of original image.
            
        Returns:
            (target_height, target_width) for resizing.
        """
        orig_h, orig_w = original_shape
        
        if orig_h <= 0 or orig_w <= 0:
            return min(10, self.height), min(20, self.width)
        
        image_aspect = orig_w / orig_h
        target_aspect = self.width / self.height
        
        if image_aspect > target_aspect:
            # Image is wider - fit to width
            target_w = self.width
            target_h = max(1, int(self.width / image_aspect))
            target_h = min(target_h, self.height)
        else:
            # Image is taller - fit to height
            target_h = self.height
            target_w = max(1, int(self.height * image_aspect))
            target_w = min(target_w, self.width)
            
        target_h = max(1, min(target_h, self.height))
        target_w = max(1, min(target_w, self.width))
            
        return target_h, target_w

    def load_and_resize_image(self, image_path: str) -> np.ndarray:
        """
        Load image and resize with proper aspect ratio handling.
        
        Args:
            image_path: Path to image file.
            
        Returns:
            Properly resized image.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        target_h, target_w = self.get_proper_dimensions(img.shape[:2])
        resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        return resized
        
    def enhance_color_contrast(self, img: np.ndarray, contrast: float = 1.2) -> np.ndarray:
        """
        Enhance color contrast for a more vibrant pixel art look.
        """
        # Convert to LAB color space for better contrast adjustment
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply contrast to luminance channel
        l = np.clip(l * contrast, 0, 255).astype(np.uint8)
        
        # Merge and convert back
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
    def apply_color_effects(self, img: np.ndarray, effect: str = "none") -> np.ndarray:
        """
        Apply color effects to image.
        """
        if effect == "sepia":
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                     [0.393, 0.769, 0.189]])
            return cv2.transform(img, sepia_filter.T)
        elif effect == "cool":
            img = img.astype(np.float32)
            img[:, :, 0] *= 0.8
            img[:, :, 2] *= 1.2
            return np.clip(img, 0, 255).astype(np.uint8)
        elif effect == "warm":
            img = img.astype(np.float32)
            img[:, :, 0] *= 1.2
            img[:, :, 1] *= 1.1
            img[:, :, 2] *= 0.8
            return np.clip(img, 0, 255).astype(np.uint8)
        elif effect == "vibrant":
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= 1.5
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        elif effect == "vintage":
            img = img.astype(np.float32)
            img[:, :, 0] *= 0.9
            img[:, :, 1] *= 1.0
            img[:, :, 2] *= 0.7
            img *= 0.9
            return np.clip(img, 0, 255).astype(np.uint8)
            
        return img
    
    def image_to_pixel_grid(self, img: np.ndarray,
                            brightness_boost: float = 1.0, contrast_boost: float = 1.0,
                            color_saturation: float = 1.0) -> None:
        """
        Convert image to a grid of colors and store them.
        """
        # Get dimensions of the input image
        h, w, _ = img.shape
        
        if contrast_boost != 1.0:
            img = self.enhance_color_contrast(img, contrast_boost)
            
        if brightness_boost != 1.0:
            img = np.clip(img * brightness_boost, 0, 255).astype(np.uint8)
            
        if color_saturation != 1.0:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= color_saturation
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply slight blur for smoother pixelated look
        if h > 2 and w > 2:
            kernel_size = max(3, min(5, min(h//10, w//10)))
            if kernel_size % 2 == 0:
                kernel_size += 1
            img_rgb = cv2.GaussianBlur(img_rgb, (kernel_size, kernel_size), 0)

        self.color_grid = [[(255, 255, 255) for _ in range(self.width)] for _ in range(self.height)]
        
        # Calculate centering offsets
        offset_y = (self.height - h) // 2
        offset_x = (self.width - w) // 2
        
        # Convert each pixel to a color in our grid
        for y in range(h):
            for x in range(w):
                target_y = y + offset_y
                target_x = x + offset_x
                
                if 0 <= target_y < self.height and 0 <= target_x < self.width:
                    r, g, b = img_rgb[y, x]
                    self.color_grid[target_y][target_x] = (int(r), int(g), int(b))
    
    def morph_colored_images(self, img1_path: str, img2_path: str, progress: float) -> None:
        """
        Morph between two colored images.
        """
        img1 = self.load_and_resize_image(img1_path)
        img2 = self.load_and_resize_image(img2_path)
        
        # Get target dimensions based on current settings and the first image
        target_h, target_w = self.get_proper_dimensions(img1.shape[:2])
        
        # Resize both images to same dimensions
        img1_resized = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_AREA)
        img2_resized = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        # Blend images
        morphed = cv2.addWeighted(img1_resized, 1 - progress, img2_resized, progress, 0)
        
        # Convert to pixel grid
        self.image_to_pixel_grid(morphed)

    def render_to_image(self, bg_color=(0, 0, 0), pixel_size=5) -> Image.Image:
        """
        Render the color grid to a PIL image (for GIF export).

        Args:
            bg_color: RGB tuple for background color.
            pixel_size: Size of each pixel in the output image.

        Returns:
            PIL Image.
        """
        img_w = self.width * pixel_size
        img_h = self.height * pixel_size
        
        img = Image.new("RGB", (img_w, img_h), bg_color)
        draw = ImageDraw.Draw(img)
        
        # Draw each pixel
        for y in range(self.height):
            for x in range(self.width):
                r, g, b = self.color_grid[y][x]
                draw.rectangle([x * pixel_size, y * pixel_size,
                                (x + 1) * pixel_size, (y + 1) * pixel_size],
                                fill=(r, g, b))
                                
        return img
    
    def save_colored_gif(self, img1_path: str, img2_path: str, steps: int = 30,
                         delay: float = 0.1, output_path: str = "pixel_morph.gif",
                         pixel_size: int = 5) -> None:
        """
        Save a morph animation between two images as a GIF.
        
        Args:
            img1_path, img2_path: Input image paths.
            steps: Number of frames.
            delay: Delay per frame (seconds).
            output_path: Path to save GIF.
            pixel_size: Size of each pixel in the output image.
        """
        frames = []
        
        for i in range(steps):
            progress = i / (steps - 1)
            self.morph_colored_images(img1_path, img2_path, progress)
            
            frame_img = self.render_to_image(pixel_size=pixel_size)
            frames.append(frame_img)
        
        if frames:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=int(delay * 1000),
                loop=0
            )
            print(f"Saved pixel art GIF to {output_path}")

    def save_as_html_grid(self, filename: str) -> None:
        """
        Save the pixel grid as an HTML file.
        """
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ background-color: black; margin: 0; display: flex; justify-content: center; align-items: center; height: 100vh; }}
        .pixel-grid-container {{ line-height: 0; white-space: nowrap; border: 1px solid white; }}
        .pixel {{
            display: inline-block;
            width: 5px;
            height: 5px;
            margin: 0;
            padding: 0;
            border: none;
        }}
    </style>
</head>
<body>
    <div class="pixel-grid-container">"""
    
        for y in range(len(self.color_grid)):
            for x in range(len(self.color_grid[y])):
                r, g, b = self.color_grid[y][x]
                html_content += f'<span class="pixel" style="background-color: rgb({r},{g},{b})"></span>'
            html_content += '\n'
    
        html_content += """    </div>
</body>
</html>"""
    
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Saved pixel art HTML grid to {filename}")

if __name__ == "__main__":
    try:
        # NOTE: You must have two images named 'image1.jpg' and 'image2.jpg'
        # in the same directory as this script, or change the paths below.
        
        # Create dummy images for demonstration if they don't exist
        img1_path = "image1.jpg"
        img2_path = "image2.jpg"
        if not os.path.exists(img1_path):
            img1 = Image.new('RGB', (200, 200), 'blue')
            draw = ImageDraw.Draw(img1)
            draw.rectangle([50, 50, 150, 150], fill='yellow')
            img1.save(img1_path)
            
        if not os.path.exists(img2_path):
            img2 = Image.new('RGB', (200, 200), 'red')
            draw = ImageDraw.Draw(img2)
            draw.ellipse([50, 50, 150, 150], fill='green')
            img2.save(img2_path)

        print("Colored OpenCV Pixel Art Generator")
        print("=" * 50)
        
        # Example 2: Create and save a morphing GIF
        print("2. Generating a morphing GIF between two images...")
        pixel_gen_gif = ColoredOpenCVPixelGenerator(width=100, height=50)
        pixel_gen_gif.save_colored_gif(img1_path, img2_path, steps=50, delay=0.1,
                                       output_path="pixel_morph.gif", pixel_size=1)

    except Exception as e:
        print(f"An error occurred: {e}")
