from google import genai
from PIL import Image

def gemini_req(image,api_key):
    
    if not api_key:
        return "API key is Missing",0.0;
    
    try:
        client = genai.Client(api_key=api_key)
        
        prompt = """Look at this handwritten curved Japanese character.
        Identify the character (Hiragana). 
        Return ONLY the single character"""
        
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents = [prompt, image]
        )
        
        if response.text:
            return response.text.strip(), 100.0
        else:
            return "No character detected", 0.0
    
    except Exception as e:
        error_msg = str(e).lower()
        
        if "429" in error_msg or "quota" in error_msg:
            return "Quota limit exceeded. Please try again later.", 0.0
        
        return f"Connection Error : {e}",0.0