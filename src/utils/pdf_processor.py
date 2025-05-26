import PyPDF2
import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_file) -> Optional[str]:
    """
    Extract text content from a PDF file
    
    Args:
        pdf_file: File object (from Flask request.files)
        
    Returns:
        str: Extracted text from PDF, or None if extraction fails
    """
    try:
        # Read the PDF file
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        
        # Extract text from all pages
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
            
        # Clean up the text
        text = text.strip()
        
        if not text:
            logger.warning("No text content extracted from PDF")
            return None
            
        return text
        
    except Exception as e:
        logger.error(f"Error processing PDF file: {str(e)}")
        return None 