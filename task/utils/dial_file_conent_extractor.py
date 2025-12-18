import io
from pathlib import Path

import pdfplumber
import pandas as pd
from aidial_client import Dial
from bs4 import BeautifulSoup


class DialFileContentExtractor:

    def __init__(self, endpoint: str, api_key: str):
        # Set Dial client with endpoint as base_url and api_key
        self.client = Dial(base_url=endpoint, api_key=api_key)

    def extract_text(self, file_url: str) -> str:
        # 1. Download file by file_url
        downloaded_file = self.client.files.download(file_url)
        
        # 2. Get downloaded file name and content
        filename = downloaded_file.filename
        file_content = downloaded_file.get_content()
        
        # 3. Get file extension
        file_extension = Path(filename).suffix.lower()
        
        # 4. Call __extract_text and return its result
        return self.__extract_text(file_content, file_extension, filename)

    def __extract_text(self, file_content: bytes, file_extension: str, filename: str) -> str:
        """Extract text content based on file type."""
        try:
            # 1. Handle .txt files
            if file_extension == '.txt':
                return file_content.decode('utf-8', errors='ignore')
            
            # 2. Handle .pdf files
            if file_extension == '.pdf':
                pdf_bytes = io.BytesIO(file_content)
                with pdfplumber.open(pdf_bytes) as pdf:
                    pages_text = [page.extract_text() or '' for page in pdf.pages]
                return '\n'.join(pages_text)
            
            # 3. Handle .csv files
            if file_extension == '.csv':
                decoded_text_content = file_content.decode('utf-8', errors='ignore')
                csv_buffer = io.StringIO(decoded_text_content)
                dataframe = pd.read_csv(csv_buffer)
                return dataframe.to_markdown(index=False)
            
            # 4. Handle .html and .htm files
            if file_extension in ['.html', '.htm']:
                decoded_html_content = file_content.decode('utf-8', errors='ignore')
                soup = BeautifulSoup(decoded_html_content, features='html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                return soup.get_text(separator='\n', strip=True)
            
            # 5. Default: return decoded content
            return file_content.decode('utf-8', errors='ignore')
            
        except Exception as e:
            print(f"Error extracting text from {filename}: {e}")
            return ""
