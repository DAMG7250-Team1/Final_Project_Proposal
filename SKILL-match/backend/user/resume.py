import os
import boto3
from datetime import datetime
from typing import Optional, Dict, Tuple, List
import PyPDF2
from io import BytesIO
from dotenv import load_dotenv
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class ResumeProcessor:
    def __init__(self):
        # Load and validate AWS credentials
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_bucket_name = os.getenv('AWS_BUCKET_NAME')
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
        self.s3_base_path = 'resume/raw_files/'
        self.markdown_base_path = 'resume/markdown/'

        # Validate required credentials
        if not all([self.aws_access_key, self.aws_secret_key, self.aws_bucket_name]):
            logger.error("Missing AWS credentials. Please check your .env file.")
            raise ValueError("Missing AWS credentials. Check .env file for AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_BUCKET_NAME")

        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.aws_region
            )
            # Test S3 connection
            self.s3_client.list_buckets()
            logger.info(f"Successfully connected to AWS S3 in region {self.aws_region}")
        except Exception as e:
            logger.error(f"Failed to connect to AWS S3: {str(e)}")
            raise Exception(f"AWS S3 connection failed: {str(e)}")

    def validate_pdf(self, pdf_content: bytes) -> Tuple[bool, str]:
        """Validate if the content is a valid PDF file."""
        try:
            # Try to read PDF
            pdf_file = BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Check if PDF has pages
            if len(pdf_reader.pages) < 1:
                return False, "PDF file is empty (no pages found)"
                
            # Try to extract some text to verify PDF is readable
            try:
                _ = pdf_reader.pages[0].extract_text()
            except Exception as e:
                return False, f"PDF appears to be corrupted or unreadable: {str(e)}"
                
            return True, "Valid PDF"
        except Exception as e:
            return False, f"Invalid PDF file: {str(e)}"

    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text content from PDF bytes."""
        try:
            # First validate PDF
            is_valid, message = self.validate_pdf(pdf_content)
            if not is_valid:
                raise Exception(message)

            pdf_file = BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_content = ""
            
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
            
            if not text_content.strip():
                logger.warning("Extracted text is empty")
                return "No text content could be extracted from the PDF"
            
            logger.info("Successfully extracted text from PDF")
            return text_content
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def convert_to_markdown(self, text: str) -> str:
        """Convert extracted text to markdown format with better structure."""
        # Split text into lines
        lines = text.split('\n')
        markdown_lines = []
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to identify main sections (all caps or ending with :)
            if line.isupper() or (len(line) < 60 and line.endswith(':')):
                current_section = line.strip(':')
                markdown_lines.append(f"\n# {current_section}\n")
            
            # Identify subsections (ending with : but not all caps)
            elif len(line) < 60 and line.endswith(':'):
                markdown_lines.append(f"\n## {line.strip(':')}\n")
            
            # Identify list items
            elif line.startswith('•') or line.startswith('-') or line.startswith('*'):
                markdown_lines.append(f"* {line[1:].strip()}")
            
            # Handle dates and locations (often in parentheses)
            elif '(' in line and ')' in line:
                markdown_lines.append(f"*{line}*\n")
            
            # Regular text
            else:
                markdown_lines.append(line)
        
        return '\n'.join(markdown_lines)

    def upload_markdown_to_s3(self, markdown_text: str, original_filename: str) -> str:
        """Upload single markdown file to S3 and return its URL."""
        try:
            base_name = os.path.splitext(original_filename)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            markdown_filename = f"{self.markdown_base_path}{timestamp}_{base_name}.md"
            
            # Add metadata as YAML front matter
            markdown_content = f"""---
original_file: {original_filename}
conversion_date: {datetime.now().isoformat()}
---

{markdown_text}
"""
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.aws_bucket_name,
                Key=markdown_filename,
                Body=markdown_content.encode('utf-8'),
                ContentType='text/markdown'
            )
            
            markdown_url = f"s3://{self.aws_bucket_name}/{markdown_filename}"
            logger.info(f"Successfully uploaded markdown file to {markdown_url}")
            
            return markdown_url
            
        except Exception as e:
            logger.error(f"Error uploading markdown file: {str(e)}")
            raise Exception(f"Failed to upload markdown file")

    def upload_to_s3(self, file_content: bytes, original_filename: str) -> Dict[str, str]:
        """Upload original PDF file to S3 and return the URL."""
        try:
            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_filename = ''.join(e for e in original_filename if e.isalnum() or e == '.')
            filename = f"{self.s3_base_path}{timestamp}_{clean_filename}"
            
            logger.info(f"Attempting to upload file to S3: {filename}")
            
            # Upload to S3
            response = self.s3_client.put_object(
                Bucket=self.aws_bucket_name,
                Key=filename,
                Body=file_content,
                ContentType='application/pdf'
            )
            
            # Check if upload was successful
            if response['ResponseMetadata']['HTTPStatusCode'] == 200:
                logger.info(f"Successfully uploaded file to S3: {filename}")
                s3_url = f"s3://{self.aws_bucket_name}/{filename}"
                
                # Verify file exists
                try:
                    self.s3_client.head_object(Bucket=self.aws_bucket_name, Key=filename)
                    logger.info("Verified file exists in S3")
                except Exception as e:
                    logger.error(f"File upload succeeded but verification failed: {str(e)}")
                    raise Exception("File upload verification failed")
                
                return {
                    "filename": filename,
                    "s3_url": s3_url
                }
            else:
                logger.error(f"S3 upload failed with response: {response}")
                raise Exception("S3 upload failed")
                
        except Exception as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            raise Exception(f"Error uploading to S3: {str(e)}")

    def process_resume(self, file_content: bytes, original_filename: str) -> Dict[str, str]:
        """Process resume: extract text, convert to markdown, and upload files."""
        try:
            logger.info(f"Starting to process resume: {original_filename}")
            
            # Validate PDF first
            is_valid, message = self.validate_pdf(file_content)
            if not is_valid:
                raise Exception(message)
            
            # Extract text from PDF
            extracted_text = self.extract_text_from_pdf(file_content)
            
            # Convert to markdown with better structure
            markdown_text = self.convert_to_markdown(extracted_text)
            
            # Upload markdown file
            markdown_url = self.upload_markdown_to_s3(markdown_text, original_filename)
            
            # Upload original PDF
            s3_info = self.upload_to_s3(file_content, original_filename)
            
            logger.info("Successfully processed resume")
            return {
                "extracted_text": extracted_text,
                "s3_url": s3_info["s3_url"],
                "filename": s3_info["filename"],
                "markdown_url": markdown_url
            }
        except Exception as e:
            logger.error(f"Error processing resume: {str(e)}")
            raise Exception(f"Error processing resume: {str(e)}")
