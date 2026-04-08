import ollama
import json
import os

class AIExtractor:
    """Handles metadata extraction from document images using local Vision-LLMs via Ollama."""
    
    def __init__(self, model="llama3.2-vision"):
        self.model = model

    def extract_metadata(self, image_path):
        """
        Processes an image and returns a dictionary of extracted metadata.
        
        Args:
            image_path (str): Path to the JPEG/PNG image of the document page.
            
        Returns:
            dict: The extracted fields (date, amount, vendor, purpose, doc_type).
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        prompt = """
        You are a highly accurate document analysis assistant. 
        Analyze the provided image and extract key details. 
        Return ONLY a JSON object with the following schema:

        {
            "date": "YYYY-MM-DD",
            "amount": float,
            "vendor": "string (The name of the company, organization, or sender)",
            "purpose": "string (Choose ONE from: Vehicle, Tuition, Medical, Business, Personal, Other)",
            "doc_type": "string (e.g. Invoice, Receipt, Application, Letter, ID Document)",
            "summary": "string (A one-sentence summary of the page)"
        }

        Important:
        - If the amount is not found, use 0.0.
        - If the date is not found or unclear, use null.
        - Ensure 'purpose' strictly matches one of the provided categories.
        - Strictly return ONLY JSON, no conversational text.
        """
        
        try:
            print(f"\n--- [AI Heartbeat] Processing: {os.path.basename(image_path)} ---")
            print(f"   [Step 1/3] Preparing image for {self.model}...")
            
            # Call Ollama with the vision model
            print(f"   [Step 2/3] Waiting for Vision LLM response (this can take 30-90s depending on hardware)...")
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                images=[image_path],
                format='json',
                options={
                    'temperature': 0.1,
                    'num_predict': 500
                }
            )
            
            print(f"   [Step 3/3] Parsing and validating AI response...")
            metadata = json.loads(response['response'])
            print(f"✅ Extraction Success for {os.path.basename(image_path)}\n")
            
            # Basic validation/cleanup
            return {
                "date": metadata.get("date"),
                "amount": float(metadata.get("amount", 0.0)),
                "vendor": metadata.get("vendor", "Unknown"),
                "purpose": metadata.get("purpose", "Other"),
                "doc_type": metadata.get("doc_type", "Unknown"),
                "summary": metadata.get("summary", "")
            }
            
        except Exception as e:
            # Return a graceful fallback if extraction fails
            print(f"Warning: AI Extraction failed for {image_path}. Error: {e}")
            return {
                "date": None,
                "amount": 0.0,
                "vendor": "Unknown",
                "purpose": "Other",
                "doc_type": "Unknown",
                "summary": f"Extraction error: {str(e)}"
            }

if __name__ == "__main__":
    import sys
    # Quick manual test if run directly
    if len(sys.argv) > 1:
        ex = AIExtractor()
        result = ex.extract_metadata(sys.argv[1])
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python extractor.py <image_path>")
