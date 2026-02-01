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
        return f"Error: {str(e)}", 0.0