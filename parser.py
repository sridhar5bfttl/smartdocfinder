import os
from pdf2image import convert_from_path
from PIL import Image

class PDFParser:
    """PDF processing utility to convert pages into images for LLM extraction."""
    
    def __init__(self, output_dir="data/processed"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def convert_pdf_to_images(self, pdf_path):
        """
        Converts each page of a PDF into a JPEG image.
        
        Args:
            pdf_path (str): Absolute or relative path to the PDF file.
            
        Returns:
            list: Paths to the generated JPEG images.
        """
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            return []
            
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        try:
            # Convert PDF pages to PIL images (DPI 200 is usually good for OCR/Vision)
            pages = convert_from_path(pdf_path, dpi=200)
            
            image_paths = []
            for i, page in enumerate(pages):
                image_name = f"{base_name}_page_{i+1}.jpg"
                image_path = os.path.join(self.output_dir, image_name)
                
                # Save as JPEG with quality 85 to balance size and clarity
                page.save(image_path, "JPEG", quality=85)
                image_paths.append(image_path)
            
            return image_paths
        except Exception as e:
            error_msg = str(e)
            if "poppler" in error_msg.lower() or "pdfinfo" in error_msg.lower():
                raise RuntimeError("System dependency 'poppler' is missing or not in PATH. Please run 'brew install poppler'.")
            raise RuntimeError(f"Failed to convert PDF: {error_msg}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        parser = PDFParser()
        images = parser.convert_pdf_to_images(sys.argv[1])
        print(f"Converted into {len(images)} images: {images}")
    else:
        print("Usage: python parser.py <pdf_path>")
