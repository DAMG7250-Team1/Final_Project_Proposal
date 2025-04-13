import os
import boto3
import logging
import json
import re
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
from dotenv import load_dotenv
import openai
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserEmbeddingProcessor:
    def __init__(self):
        # Initialize OpenAI API
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            logger.error("OpenAI API key not found in environment variables")
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in your environment.")
        
        # Set OpenAI client
        self.client = openai.OpenAI(api_key=self.openai_api_key)
        
        # Initialize AWS credentials
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_bucket_name = os.getenv('AWS_BUCKET_NAME', 'skillmatchai')
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
        
        # Validate AWS credentials
        if not all([self.aws_access_key, self.aws_secret_key, self.aws_bucket_name]):
            logger.error("Missing AWS credentials. Please check your .env file.")
            raise ValueError("Missing AWS credentials. Check environment variables.")
        
        # Initialize S3 client
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
        
        # Embedding model and dimension
        self.embedding_model = "text-embedding-3-large"
        self.embedding_dimensions = 3072  # Dimensions for text-embedding-3-large
        
        # Weights for different document types
        self.weights = {
            "resume": 0.7,
            "github": 0.3
        }

    def parse_s3_url(self, s3_url: str) -> Tuple[str, str]:
        """Parse S3 URL into bucket and key."""
        parsed = urlparse(s3_url)
        if parsed.scheme != 's3':
            raise ValueError(f"Not a valid S3 URL: {s3_url}")
        
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        return bucket, key

    def fetch_markdown_from_s3(self, s3_url: str) -> str:
        """Fetch markdown content from S3."""
        try:
            bucket, key = self.parse_s3_url(s3_url)
            
            logger.info(f"Fetching markdown from S3: {bucket}/{key}")
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            
            logger.info(f"Successfully fetched markdown ({len(content)} bytes)")
            return content
        except Exception as e:
            logger.error(f"Error fetching markdown from S3: {str(e)}")
            raise Exception(f"Failed to fetch markdown from S3: {str(e)}")

    def process_markdown_text(self, markdown_text: str) -> str:
        """Process markdown text to extract meaningful content."""
        # Remove YAML front matter
        markdown_text = re.sub(r'^---\n.*?\n---\n', '', markdown_text, flags=re.DOTALL)
        
        # Remove markdown formatting symbols
        markdown_text = re.sub(r'#{1,6}\s+', '', markdown_text)  # Headers
        markdown_text = re.sub(r'\*\*|\*|__|_', '', markdown_text)  # Bold and italic
        markdown_text = re.sub(r'```.*?```', '', markdown_text, flags=re.DOTALL)  # Code blocks
        markdown_text = re.sub(r'`.*?`', '', markdown_text)  # Inline code
        markdown_text = re.sub(r'\[.*?\]\(.*?\)', '', markdown_text)  # Links
        
        # Remove HTML tags
        markdown_text = re.sub(r'<.*?>', '', markdown_text)
        
        # Clean up whitespace
        markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)
        markdown_text = markdown_text.strip()
        
        return markdown_text

    def chunk_text(self, text: str, chunk_size: int = 8000) -> List[str]:
        """Split text into chunks of specified size."""
        if not text:
            return []
            
        # Split by paragraphs first to maintain context
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph exceeds chunk size, save current chunk and start new one
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        logger.info(f"Text chunked into {len(chunks)} segments")
        return chunks

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using OpenAI API."""
        if not texts:
            return []
            
        try:
            logger.info(f"Generating embeddings for {len(texts)} text chunks")
            
            embeddings = []
            for i, text in enumerate(texts):
                response = self.client.embeddings.create(
                    input=text,
                    model=self.embedding_model
                )
                embedding = response.data[0].embedding
                embeddings.append(embedding)
                logger.info(f"Generated embedding {i+1}/{len(texts)}")
            
            return embeddings
                
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise Exception(f"Failed to generate embeddings: {str(e)}")

    def combine_embeddings(self, embeddings: List[List[float]], weights: Optional[List[float]] = None) -> List[float]:
        """Combine multiple embeddings using weighted average."""
        if not embeddings:
            return [0.0] * self.embedding_dimensions
            
        # Use equal weights if none provided
        if not weights:
            weights = [1.0/len(embeddings)] * len(embeddings)
        
        # Ensure weights sum to 1
        weights = [w/sum(weights) for w in weights]
        
        # Convert to numpy arrays for easier computation
        embedding_arrays = np.array(embeddings)
        weight_array = np.array(weights).reshape(-1, 1)
        
        # Compute weighted average
        combined = np.sum(embedding_arrays * weight_array, axis=0)
        
        # Normalize the combined embedding to unit length
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        
        return combined.tolist()

    def process_user_data(self, resume_url: str, github_url: Optional[str] = None) -> Dict[str, Any]:
        """Process user's resume and GitHub data to create combined embeddings."""
        try:
            logger.info("Starting user data processing")
            result = {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "sources": []
            }
            
            # Process resume markdown
            resume_data = self.process_single_source(resume_url, "resume")
            result["sources"].append(resume_data)
            
            # Process GitHub markdown if available
            if github_url:
                github_data = self.process_single_source(github_url, "github")
                result["sources"].append(github_data)
            
            # Combine all chunk embeddings with weights based on source type
            all_embeddings = []
            all_weights = []
            
            for source in result["sources"]:
                for i, embedding in enumerate(source["embeddings"]):
                    all_embeddings.append(embedding)
                    # Apply source-specific weight
                    all_weights.append(self.weights.get(source["type"], 0.5))
            
            # Generate combined embedding
            if all_embeddings:
                result["combined_embedding"] = self.combine_embeddings(all_embeddings, all_weights)
                result["embedding_dimensions"] = len(result["combined_embedding"])
            else:
                result["combined_embedding"] = [0.0] * self.embedding_dimensions
                result["embedding_dimensions"] = self.embedding_dimensions
                result["status"] = "warning"
                result["message"] = "No embeddings were generated"
            
            # Extract and combine all skills
            all_skills = set()
            for source in result["sources"]:
                all_skills.update(source.get("extracted_skills", []))
            
            result["all_skills"] = list(all_skills)
            result["total_skills"] = len(all_skills)
            
            logger.info(f"User data processing completed with {len(all_skills)} skills extracted")
            return result
            
        except Exception as e:
            logger.error(f"Error processing user data: {str(e)}")
            return {
                "status": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }

    def process_single_source(self, s3_url: str, source_type: str) -> Dict[str, Any]:
        """Process a single markdown source (resume or GitHub profile)."""
        try:
            # Fetch markdown content
            markdown_content = self.fetch_markdown_from_s3(s3_url)
            
            # Process markdown text
            processed_text = self.process_markdown_text(markdown_content)
            
            # Chunk the text
            chunks = self.chunk_text(processed_text)
            
            # Generate embeddings for each chunk
            embeddings = self.generate_embeddings(chunks)
            
            # Extract skills (simplified, would be more complex in production)
            extracted_skills = self.extract_skills_from_text(processed_text)
            
            return {
                "type": source_type,
                "s3_url": s3_url,
                "text_length": len(processed_text),
                "chunk_count": len(chunks),
                "embeddings": embeddings,
                "extracted_skills": extracted_skills
            }
            
        except Exception as e:
            logger.error(f"Error processing {source_type} from {s3_url}: {str(e)}")
            return {
                "type": source_type,
                "s3_url": s3_url,
                "error": str(e),
                "embeddings": [],
                "extracted_skills": []
            }

    def extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from text using OpenAI API."""
        try:
            prompt = f"""
            Extract ALL technical skills, tools, frameworks, programming languages, and relevant job skills from the following text.
            Include:
            - Programming languages (e.g., Python, Java, C++)
            - Databases (e.g., SQL, MySQL, PostgreSQL)
            - Cloud platforms (e.g., AWS, Azure, GCP)
            - Frameworks and libraries (e.g., TensorFlow, PyTorch, React)
            - Tools and technologies (e.g., Docker, Kubernetes, Git)
            - Data science and ML skills (e.g., Machine Learning, Data Analysis)
            - Business and soft skills (e.g., Project Management, Communication)
            
            Return ONLY a JSON array of strings with the skills, nothing else.
            Example: ["Python", "SQL", "AWS", "Machine Learning", "Project Management"]
            
            TEXT:
            {text[:8000]}  # Limit to avoid token limits
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting technical and professional skills from resumes and job descriptions. Return skills in a JSON array."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            # Extract the JSON string from the response
            skills_json = response.choices[0].message.content
            
            # Parse the JSON
            try:
                skills_data = json.loads(skills_json)
                skills = skills_data.get("skills", [])
                if not isinstance(skills, list):
                    skills = []
            except:
                # Fallback: try to extract array directly using regex
                match = re.search(r'\[(.*?)\]', skills_json)
                if match:
                    skills_str = match.group(1)
                    skills = [s.strip().strip('"\'') for s in skills_str.split(',')]
                else:
                    skills = []
            
            # Remove duplicates while preserving case
            seen = set()
            unique_skills = []
            for skill in skills:
                lower_skill = skill.lower()
                if lower_skill not in seen:
                    seen.add(lower_skill)
                    unique_skills.append(skill)
            
            logger.info(f"Extracted {len(unique_skills)} skills from text")
            return unique_skills
            
        except Exception as e:
            logger.error(f"Error extracting skills: {str(e)}")
            return []

    def save_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> str:
        """Save user profile data to S3."""
        try:
            # Format key with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            key = f"user_profiles/{user_id}/{timestamp}_profile.json"
            
            # Convert to JSON
            json_data = json.dumps(profile_data, ensure_ascii=False)
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.aws_bucket_name,
                Key=key,
                Body=json_data.encode('utf-8'),
                ContentType='application/json'
            )
            
            profile_url = f"s3://{self.aws_bucket_name}/{key}"
            logger.info(f"User profile saved to {profile_url}")
            
            return profile_url
            
        except Exception as e:
            logger.error(f"Error saving user profile: {str(e)}")
            raise Exception(f"Failed to save user profile: {str(e)}")

# Example usage
if __name__ == "__main__":
    processor = UserEmbeddingProcessor()
    
    # Example S3 URLs (placeholders)
    resume_url = "s3://skillmatchai/resume/markdown/20250411_200621_AUM-JS-python-Doc.md"
    github_url = "s3://skillmatchai/github/markdown/20250411_200617_Aumpatelarjun_github_profile.md"
    
    # Process user data
    result = processor.process_user_data(resume_url, github_url)
    
    # Save user profile
    user_id = "test_user_123"
    profile_url = processor.save_user_profile(user_id, result)
    
    print(f"User profile processed and saved to {profile_url}")
    print(f"Total skills extracted: {result['total_skills']}")
