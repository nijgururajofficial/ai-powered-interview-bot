import PyPDF2
import re

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from an uploaded PDF file using PyPDF2.
    """
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def extract_score_from_response(response_text):
    """
    Extracts the numeric score from the ChatGPT response.
    The response is expected to include a line like "Score: <score>".
    """
    match = re.search(r"Score:\s*([\d\.]+)", response_text)
    if match:
        try:
            score = float(match.group(1))
            return score
        except ValueError:
            return None
    return None