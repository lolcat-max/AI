import cv2
import numpy as np
import time
import os
from typing import Tuple, List, Optional, Union
import colorsys
import imageio
from PIL import Image, ImageDraw, ImageFont
class ColoredOpenCVASCIIGenerator:
    """
    Generates colored ASCII art from real images using OpenCV with proper dimension handling.
    Supports full RGB color output using ANSI escape codes.
    """
    
    # ASCII characters ordered from darkest to lightest with better density distribution
    ASCII_CHARS = " .:-=+*#%@"
    ASCII_CHARS_DETAILED = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
    
    def __init__(self, width: int = 120, height: int = 40, char_set: str = "simple", use_color: bool = True):
        """
        Initialize the colored ASCII generator.
        
        Args:
            width: Output ASCII width in characters
            height: Output ASCII height in characters
            char_set: "simple" or "detailed" character set
            use_color: Whether to use ANSI color codes
        """
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive integers.")
            
        self.width = width
        self.height = height
        self.use_color = use_color
        self.ascii_chars = self.ASCII_CHARS if char_set == "simple" else self.ASCII_CHARS_DETAILED
        
        # Character aspect ratio compensation (most terminal fonts are ~2:1 height:width)
        self.char_aspect_ratio = 2.0
        
        # Grid to store ASCII characters and colors
        self.grid = [[' ' for _ in range(width)] for _ in range(height)]
        self.color_grid = [[(255, 255, 255) for _ in range(width)] for _ in range(height)]




    def render_to_image(self, bg_color=(0, 0, 0), font_size=12) -> Image.Image:
        """
        Render the ASCII grid to a PIL image (for GIF export).

        Args:
            bg_color: RGB tuple for background color
            font_size: Font size for rendering

        Returns:
            PIL Image
        """
        # Pick a monospaced font (Courier New or DejaVu Sans Mono)
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # --- Get character size (works across Pillow versions) ---
        test_char = "A"
        try:
            # Preferred in Pillow >=10
            bbox = font.getbbox(test_char)
            char_w, char_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            # Older Pillow fallback
            char_w, char_h = font.getsize(test_char)

        img_w = char_w * self.width
        img_h = char_h * self.height

        # Create image
        img = Image.new("RGB", (img_w, img_h), bg_color)
        draw = ImageDraw.Draw(img)

        # Draw each char
        for y in range(self.height):
            for x in range(self.width):
                char = self.grid[y][x]
                r, g, b = self.color_grid[y][x]
                draw.text((x * char_w, y * char_h), char, font=font, fill=(r, g, b))

        return img


    def save_colored_gif(self, img1_path: str, img2_path: str, steps: int = 30,
                         delay: float = 0.1, use_edge_detection: bool = False,
                         output_path: str = "ascii_morph.gif", font_size: int = 12) -> None:
        """
        Save a morph animation between two images as a GIF.
        
        Args:
            img1_path, img2_path: Input image paths
            steps: Number of frames
            delay: Delay per frame (seconds)
            use_edge_detection: Whether to apply edge detection
            output_path: Path to save GIF
            font_size: Font size for ASCII rendering
        """
        frames = []

        for i in range(steps):
            progress = i / (steps - 1)
            self.morph_colored_images(img1_path, img2_path, progress, use_edge_detection)

            frame_img = self.render_to_image(font_size=font_size)
            frames.append(frame_img)

        # Save as GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(delay * 1000),
            loop=0
        )
        print(f"Saved ASCII GIF to {output_path}")
        
    def get_proper_dimensions(self, original_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculate proper dimensions maintaining aspect ratio and accounting for character dimensions.
        
        Args:
            original_shape: (height, width) of original image
            
        Returns:
            (target_height, target_width) for resizing
        """
        orig_h, orig_w = original_shape
        
        # Ensure we have valid dimensions
        if orig_h <= 0 or orig_w <= 0:
            return min(10, self.height), min(20, self.width)
        
        # Calculate aspect ratios
        image_aspect = orig_w / orig_h
        terminal_aspect = (self.width * self.char_aspect_ratio) / self.height
        
        if image_aspect > terminal_aspect:
            # Image is wider - fit to width
            target_w = self.width
            target_h = max(1, int(self.width / (image_aspect * self.char_aspect_ratio)))
            target_h = min(target_h, self.height)
        else:
            # Image is taller - fit to height  
            target_h = self.height
            target_w = max(1, int(self.height * image_aspect * self.char_aspect_ratio))
            target_w = min(target_w, self.width)
            
        # Ensure minimum dimensions
        target_h = max(1, min(target_h, self.height))
        target_w = max(1, min(target_w, self.width))
            
        return target_h, target_w
    
    def load_and_resize_image(self, image_path: str) -> np.ndarray:
        """
        Load image and resize with proper aspect ratio handling.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Properly resized image
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Get proper dimensions
        target_h, target_w = self.get_proper_dimensions(img.shape[:2])
        
        # Resize image
        resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        return resized
        
    def rgb_to_ansi(self, r: int, g: int, b: int, background: bool = False) -> str:
        """
        Convert RGB values to ANSI escape code.
        
        Args:
            r, g, b: RGB values (0-255)
            background: Whether this is for background color
            
        Returns:
            ANSI escape sequence
        """
        if not self.use_color:
            return ""
            
        code = 48 if background else 38  # 48 for background, 38 for foreground
        return f"\033[{code};2;{r};{g};{b}m"
    
    def get_color_intensity(self, r: int, g: int, b: int) -> float:
        """
        Calculate perceived brightness/intensity of RGB color.
        Uses luminance formula for better perception accuracy.
        """
        return 0.299 * r + 0.587 * g + 0.114 * b
    
    def enhance_color_contrast(self, img: np.ndarray, contrast: float = 1.2) -> np.ndarray:
        """
        Enhance color contrast for better ASCII representation.
        """
        # Convert to LAB color space for better contrast adjustment
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply contrast to luminance channel
        l = np.clip(l * contrast, 0, 255).astype(np.uint8)
        
        # Merge and convert back
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def image_to_colored_ascii(self, img: np.ndarray, use_edge_detection: bool = False,
                             brightness_boost: float = 1.0, contrast_boost: float = 1.0,
                             color_saturation: float = 1.0) -> None:
        """
        Convert image to colored ASCII and store in grids.
        
        Args:
            img: Input image (BGR format)
            use_edge_detection: Apply edge detection
            brightness_boost: Brightness multiplier
            contrast_boost: Contrast multiplier  
            color_saturation: Color saturation multiplier
        """
        # Apply contrast enhancement
        if contrast_boost != 1.0:
            img = self.enhance_color_contrast(img, contrast_boost)
            
        # Apply brightness boost
        if brightness_boost != 1.0:
            img = np.clip(img * brightness_boost, 0, 255).astype(np.uint8)
            
        # Adjust color saturation
        if color_saturation != 1.0:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= color_saturation  # Saturation channel
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Convert to RGB for proper color handling
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get grayscale for character selection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection if requested
        if use_edge_detection:
            if h > 50 and w > 50:  # Only apply edge detection for reasonable sized images
                edges = cv2.Canny(gray, 50, 150)
                # Use edges to modify character selection
                gray = np.maximum(gray, edges)
        
        # Apply slight blur for smoother ASCII (kernel size must be odd and positive)
        h, w = gray.shape
        if h > 2 and w > 2:
            kernel_size = max(3, min(5, min(h//10, w//10)))
            if kernel_size % 2 == 0:
                kernel_size += 1
            gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
            img_rgb = cv2.GaussianBlur(img_rgb, (kernel_size, kernel_size), 0)
        
        # Initialize grids
        self.grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        self.color_grid = [[(255, 255, 255) for _ in range(self.width)] for _ in range(self.height)]
        
        # Calculate centering offsets
        offset_y = (self.height - h) // 2
        offset_x = (self.width - w) // 2
        
        # Convert each pixel to colored ASCII
        for y in range(h):
            for x in range(w):
                target_y = y + offset_y
                target_x = x + offset_x
                
                if 0 <= target_y < self.height and 0 <= target_x < self.width:
                    # Get intensity for character selection
                    intensity = gray[y, x] / 255.0
                    char_index = int(intensity * (len(self.ascii_chars) - 1))
                    char_index = max(0, min(len(self.ascii_chars) - 1, char_index))
                    
                    # Get RGB color
                    r, g, b = img_rgb[y, x]
                    
                    # Store character and color
                    self.grid[target_y][target_x] = self.ascii_chars[char_index]
                    self.color_grid[target_y][target_x] = (int(r), int(g), int(b))
    
    def apply_color_effects(self, img: np.ndarray, effect: str = "none") -> np.ndarray:
        """
        Apply color effects to image.
        """
        if effect == "sepia":
            # Sepia tone effect
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168], 
                                     [0.393, 0.769, 0.189]])
            return cv2.transform(img, sepia_filter.T)
            
        elif effect == "cool":
            # Cool color temperature
            img = img.astype(np.float32)
            img[:, :, 0] *= 0.8  # Reduce red
            img[:, :, 2] *= 1.2  # Increase blue
            return np.clip(img, 0, 255).astype(np.uint8)
            
        elif effect == "warm":
            # Warm color temperature  
            img = img.astype(np.float32)
            img[:, :, 0] *= 1.2  # Increase red
            img[:, :, 1] *= 1.1  # Slight increase green
            img[:, :, 2] *= 0.8  # Reduce blue
            return np.clip(img, 0, 255).astype(np.uint8)
            
        elif effect == "vibrant":
            # Increase saturation
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= 1.5  # Increase saturation
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
        elif effect == "vintage":
            # Vintage film look
            img = img.astype(np.float32)
            # Add slight brown/yellow tint
            img[:, :, 0] *= 0.9    # Slight reduce red
            img[:, :, 1] *= 1.0    # Keep green
            img[:, :, 2] *= 0.7    # Reduce blue significantly
            # Reduce overall brightness slightly
            img *= 0.9
            return np.clip(img, 0, 255).astype(np.uint8)
            
        return img
    
    def morph_colored_images(self, img1_path: str, img2_path: str, progress: float,
                            use_edge_detection: bool = False) -> None:
        """
        Morph between two colored images.
        """
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            raise ValueError("Could not load one or both images")
            
        # Get target dimensions based on current settings
        target_h, target_w = self.get_proper_dimensions(img1.shape[:2])
        
        # Resize both images to same dimensions
        img1_resized = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_AREA)
        img2_resized = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        # Blend images
        morphed = cv2.addWeighted(img1_resized, 1 - progress, img2_resized, progress, 0)
        
        # Convert to ASCII
        self.image_to_colored_ascii(morphed, use_edge_detection)
    
    def display_colored(self, use_background_color: bool = False) -> None:
        """
        Display colored ASCII art using ANSI escape codes.
        
        Args:
            use_background_color: Whether to also color the background
        """
        reset_code = "\033[0m" if self.use_color else ""
        
        for y in range(len(self.grid)):
            line = ""
            for x in range(len(self.grid[y])):
                char = self.grid[y][x]
                r, g, b = self.color_grid[y][x]
                
                if self.use_color:
                    if use_background_color and char != ' ':
                        # Use darker version for background
                        bg_r, bg_g, bg_b = max(0, r-50), max(0, g-50), max(0, b-50)
                        color_code = self.rgb_to_ansi(r, g, b) + self.rgb_to_ansi(bg_r, bg_g, bg_b, True)
                    else:
                        color_code = self.rgb_to_ansi(r, g, b)
                    line += color_code + char
                else:
                    line += char
                    
            line += reset_code
            print(line)
    
    def save_colored_html(self, filename: str) -> None:
        """
        Save colored ASCII art as HTML file.
        """
        html_content = """<!DOCTYPE html>
<html>
<head>
    <style>
        body { background-color: black; font-family: 'Courier New', monospace; }
        .ascii-art { white-space: pre; font-size: 8px; line-height: 1.0; }
    </style>
</head>
<body>
    <div class="ascii-art">"""
    
        for y in range(len(self.grid)):
            for x in range(len(self.grid[y])):
                char = self.grid[y][x]
                r, g, b = self.color_grid[y][x]
                
                # Escape HTML characters
                if char == '<':
                    char = '&lt;'
                elif char == '>':
                    char = '&gt;'
                elif char == '&':
                    char = '&amp;'
                elif char == ' ':
                    char = '&nbsp;'
                
                html_content += f'<span style="color: rgb({r},{g},{b})">{char}</span>'
            html_content += '\n'
    
        html_content += """    </div>
</body>
</html>"""
    
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def generate_from_image(self, image_path: str, effect: str = "none",
                          use_edge_detection: bool = False,
                          brightness_boost: float = 1.0,
                          contrast_boost: float = 1.0,
                          color_saturation: float = 1.0) -> None:
        """
        Generate colored ASCII art from image.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Apply color effects
        img = self.apply_color_effects(img, effect)
        
        # Resize with proper dimensions
        target_h, target_w = self.get_proper_dimensions(img.shape[:2])
        img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        # Convert to colored ASCII
        self.image_to_colored_ascii(img_resized, use_edge_detection, 
                                 brightness_boost, contrast_boost, color_saturation)
    
    def animate_colored_morph(self, img1_path: str, img2_path: str, steps: int = 30,
                              delay: float = 0.1, use_edge_detection: bool = False) -> None:
        """
        Animate morphing between two colored images.
        """
        print(f"Morphing from {os.path.basename(img1_path)} to {os.path.basename(img2_path)}...")
        
        for i in range(steps):
            progress = i / (steps - 1)
            
            # Clear screen
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"Morph Progress: {progress:.1%}")
            print("=" * min(60, self.width))
            
            self.morph_colored_images(img1_path, img2_path, progress, use_edge_detection)
            self.display_colored()
            
            time.sleep(delay)
    
    def create_color_palette_demo(self) -> None:
        """
        Create a demo showing color capabilities.
        """
        # Create a gradient image with proper dimensions
        demo_height = min(self.height, 20)  # Limit height for demo
        demo_width = min(self.width, 60)    # Limit width for demo
        
        # Initialize grids with proper dimensions
        self.grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        self.color_grid = [[(255, 255, 255) for _ in range(self.width)] for _ in range(self.height)]
        
        # Calculate centering offsets
        offset_y = (self.height - demo_height) // 2
        offset_x = (self.width - demo_width) // 2
        
        for y in range(demo_height):
            for x in range(demo_width):
                target_y = y + offset_y
                target_x = x + offset_x
                
                if 0 <= target_y < self.height and 0 <= target_x < self.width:
                    # Create HSV gradient
                    h = (x / demo_width) * 360
                    s = 1.0
                    v = max(0.3, 1 - y / demo_height)  # Avoid too dark colors
                    
                    # Convert HSV to RGB
                    rgb = colorsys.hsv_to_rgb(h/360, s, v)
                    r, g, b = int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)
                    
                    # Select character based on brightness
                    intensity = (r * 0.299 + g * 0.587 + b * 0.114) / 255
                    char_index = int(intensity * (len(self.ascii_chars) - 1))
                    char_index = max(0, min(len(self.ascii_chars) - 1, char_index))
                    
                    # Store character and color
                    self.grid[target_y][target_x] = self.ascii_chars[char_index]
                    self.color_grid[target_y][target_x] = (r, g, b)

# Example usage and demonstrations
if __name__ == "__main__":
    try:
        print("Colored OpenCV ASCII Art Generator")
        print("=" * 50)

        print("Creating a basic pattern...")
        
        # Create a simple test pattern that doesn't require external images
        test_generator = ColoredOpenCVASCIIGenerator(width=80, height=50, char_set="detailed", use_color=True)
        test_generator.save_colored_gif("image1.jpg", "image2.jpg", steps=100, delay=0.08, output_path="morph.gif")
        
        # Create a simple geometric pattern
        for y in range(test_generator.height):
            for x in range(test_generator.width):
                # Create a circular pattern
                center_x, center_y = test_generator.width // 2, test_generator.height // 2
                distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                max_distance = min(test_generator.width, test_generator.height) // 2
                
                if distance <= max_distance:
                    intensity = 1 - (distance / max_distance)
                    char_index = int(intensity * (len(test_generator.ascii_chars) - 1))
                    char_index = max(0, min(len(test_generator.ascii_chars) - 1, char_index))
                    
                    # Create color based on position
                    hue = (x / test_generator.width) * 360
                    rgb = colorsys.hsv_to_rgb(hue/360, 0.8, intensity)
                    r, g, b = int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)
                    
                    test_generator.grid[y][x] = test_generator.ascii_chars[char_index]
                    test_generator.color_grid[y][x] = (r, g, b)
        
        print("\nCircular Pattern Demo:")
        test_generator.display_colored()
        
        print("\n" + "="*50)

      
        # Animate colored morph
        test_generator.animate_colored_morph('image1.jpg', 'image2.jpg', 
                                     steps=100, delay=0.05)
        
        
        print("\n3. Features:")
        print("    ✓ Proper aspect ratio handling")
        print("    ✓ Full RGB color support (ANSI codes)")
        print("    ✓ HTML export with colors") 
        print("    ✓ Multiple color effects")
        print("    ✓ Colored morphing animations")
        print("    ✓ Edge detection with color")
        print("    ✓ Brightness/contrast/saturation controls")
        
    except Exception as e:
        print(f"Demo error: {e}")
