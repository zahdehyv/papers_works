import os
import re
import time
from PyPDF2 import PdfReader
import google.generativeai as genai
import shutil

# Configure Gemini
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

def get_existing_categories(output_base):
    """Retrieve existing category structure from output directory"""
    existing = []
    if not os.path.exists(output_base):
        return existing
    for category in os.listdir(output_base):
        category_path = os.path.join(output_base, category)
        if os.path.isdir(category_path):
            existing.append(category)
    return existing

def extract_text_from_pdf(pdf_path, max_pages=13):
    """Extract text from first n pages of a PDF"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            for i in range(min(len(reader.pages), max_pages)):
                page = reader.pages[i]
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text.strip()

def get_category_from_gemini(content, existing_categories):
    """Get category and explanation from Gemini API with exponential backoff"""
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 65536,
        "response_mime_type": "text/plain",
        }

    model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-thinking-exp-01-21",
    generation_config=generation_config,
    )
    existing_list = "\n- ".join(existing_categories) or "None"
    
    prompt = f"""Read Carefuly this document looking for the most restrictive main object or area of study.
Propose a category for it.
Existing categories:
- {existing_list}

Guidelines:
1. Use existing categories if VEEEEERY similar.
2. If creating new, make it DISTINCT from existing
3. Use a SINGLE category name
4. Consider the document's primary focus
5. Keep explanations brief (10-20 words)

tip: use keywords if available

Example Categories:
- Consciousness (prioritize this)
- Reasoning (prioritize this)
- Psychology
- NLP
- Neuroscience

Document content (first 3 pages):
{content[:15000]}

Respond EXACTLY in this format:

Long explanation of categorization reason, with a step by step reasoning about it
(Example of long explanation:
- The document discusses the nature of consciousness, including its definition, types, and implications. It also explores the relationship between consciousness and the brain, and the philosophical debates surrounding it. Therefore, it falls under the category of Consciousness.)
- It uses LLM to explain details about some cognitive archutectures
- Some Computational models are described so it can be Computer Science
After all the considerations, the model mainly falls in Consciousness category
)
<answer>CategoryName</answer>"""

    max_retries = 5
    base_delay = 1  # Initial delay in seconds
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract explanation and category using regex
            match = re.search(
                r'(.*?)<answer>(.*?)</answer>',
                response_text,
                re.DOTALL
            )
            
            if not match:
                raise ValueError(f"Invalid response format: {response_text}")
                
            explanation = match.group(1).strip()
            category = match.group(2).strip()
            
            # Validate category format
            if '/' in category or '\n' in category:
                raise ValueError(f"Invalid category format: {category}")
                
            return category, explanation
            
        except Exception as e:
            if '429' in str(e):  # Rate limit error detection
                delay = base_delay * (2 ** attempt)
                print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                print(f"due to {e}")
                time.sleep(delay)
            else:
                print(f"Gemini API error: {e}")
                raise
                
    raise Exception(f"Failed after {max_retries} retries due to rate limiting")

def process_pdfs(input_folder, output_base):
    """Main processing function"""
    existing_categories = get_existing_categories(output_base)
    
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith('.pdf'):
            continue
            
        pdf_path = os.path.join(input_folder, filename)
        text = extract_text_from_pdf(pdf_path)
        
        if not text:
            print(f"Skipping {filename} - no text extracted")
            continue
            
        try:
            category, explanation = get_category_from_gemini(text, existing_categories)
            
            # Create target directory
            target_dir = os.path.join(output_base, category)
            os.makedirs(target_dir, exist_ok=True)
            
            # Copy PDF with original name
            shutil.move(pdf_path, os.path.join(target_dir, filename))
            
            # Update existing categories list
            if category not in existing_categories:
                existing_categories.append(category)
            print("="*53)
            print(f"PROCESSED: {filename} -> {target_dir}")
            print(f"EXPLANATION:\n{explanation}")
            print(f"CATEGORY: {category}")
            print("="*53,"\n")
            
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

def main():

    input_folder = os.path.join(os.getcwd(), 'newbies')
    output_folder = os.path.join(os.getcwd(), 'papers')
    print(f"Processing PDFs from {input_folder}")
    print(f"Output directory: {output_folder}")
    
    process_pdfs(input_folder, output_folder)

if __name__ == '__main__':
    main()