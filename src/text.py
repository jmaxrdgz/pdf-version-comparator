import PyPDF2
import pytesseract
import re
from pdf2image import convert_from_path

def clean_sentence(s: str) -> str:
    s = s.replace('\n', ' ')
    s = ' '.join(s.split())
    return s

def segment_sentences(text1: str, text2: str):
    sentences1 = re.split(r'(?<=[.!?])\s+', text1.strip())
    sentences2 = re.split(r'(?<=[.!?])\s+', text2.strip())
    sentences1 = [clean_sentence(s) for s in sentences1 if s.strip()]
    sentences2 = [clean_sentence(s) for s in sentences2 if s.strip()]
    return sentences1, sentences2

def extract_text_from_pdf(pdf_path:str) -> dict:
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            metadata = {}

            if reader.metadata:
                metadata = {key: value for key, value in reader.metadata.items()}

            for page in reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"

            if not text.strip():
                images = convert_from_path(pdf_path)
                for image in images:
                    text += pytesseract.image_to_string(image) + "\n"

            return {"text": text, "metadata": metadata}

    except FileNotFoundError:
        return {"error": "The file was not found."}
    except PyPDF2.errors.PdfReadError:
        return {"error": "Could not read the PDF file."}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}