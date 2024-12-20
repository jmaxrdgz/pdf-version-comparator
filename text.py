import PyPDF2
import pytesseract
from pdf2image import convert_from_path

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