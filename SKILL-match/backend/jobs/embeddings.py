import os
import sys
import json
import logging
import boto3
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from datetime import datetime
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class JobEmbeddingsProcessor:
    def __init__(self):
        # OpenAI configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")

        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.embedding_model = "text-embedding-3-large"
        self.embedding_dimensions = 3072

        # S3 configuration
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_bucket_name = os.getenv('AWS_BUCKET_NAME', 'skillmatchai')
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
        
        # S3 paths
        self.s3_jobs_raw_path = 'jobs/raw_files/'
        self.s3_jobs_embeddings_path = 'jobs/embeddings/'
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
            region_name=self.aws_region
        )

        # Pinecone configuration
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pinecone_environment = os.getenv('PINECONE_ENVIRONMENT', 'gcp-starter')
        self.pinecone_index_name = os.getenv('PINECONE_INDEX_NAME', 'skillmatch-index')
        self.pinecone_namespace = os.getenv('PINECONE_NAMESPACE', 'jobs')
        
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key not found in environment variables")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Create index if it doesn't exist
        if self.pinecone_index_name not in [index.name for index in self.pc.list_indexes()]:
            self.pc.create_index(
                name=self.pinecone_index_name,
                dimension=self.embedding_dimensions,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-west-2")
            )
            logger.info(f"Created new Pinecone index: {self.pinecone_index_name}")
        
        # Connect to the index
        self.index = self.pc.Index(self.pinecone_index_name)
        logger.info(f"Connected to Pinecone index: {self.pinecone_index_name}")

        # Similarity thresholds
        self.similarity_thresholds = {
            "high": 0.60,    # Currently set to 0.60
            "medium": 0.40,  # Currently set to 0.40
            "low": 0.20      # Currently set to 0.20
        }

    def list_s3_job_files(self) -> List[str]:
        """List all job JSON files in the S3 raw files directory"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.aws_bucket_name,
                Prefix=self.s3_jobs_raw_path
            )
            
            if 'Contents' not in response:
                logger.warning(f"No files found in s3://{self.aws_bucket_name}/{self.s3_jobs_raw_path}")
                return []
            
            files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.json')]
            logger.info(f"Found {len(files)} job files in S3")
            return files
        
        except Exception as e:
            logger.error(f"Error listing S3 job files: {str(e)}")
            raise

    def load_job_file_from_s3(self, file_key: str) -> List[Dict[str, Any]]:
        """Load a job JSON file from S3"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.aws_bucket_name,
                Key=file_key
            )
            
            job_data = json.loads(response['Body'].read().decode('utf-8'))
            logger.info(f"Loaded {len(job_data)} jobs from {file_key}")
            return job_data
        
        except Exception as e:
            logger.error(f"Error loading job file {file_key}: {str(e)}")
            raise

    def save_embeddings_to_s3(self, file_key: str, embeddings: Dict[str, Any]) -> str:
        """Save job embeddings to S3"""
        try:
            # Generate a filename based on the original file
            base_name = file_key.split('/')[-1].replace('.json', '')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_key = f"{self.s3_jobs_embeddings_path}{base_name}_embeddings_{timestamp}.json"
            
            # Save to S3
            self.s3_client.put_object(
                Bucket=self.aws_bucket_name,
                Key=output_key,
                Body=json.dumps(embeddings, ensure_ascii=False),
                ContentType='application/json'
            )
            
            s3_url = f"s3://{self.aws_bucket_name}/{output_key}"
            logger.info(f"Saved embeddings to {s3_url}")
            return s3_url
        
        except Exception as e:
            logger.error(f"Error saving embeddings to S3: {str(e)}")
            raise

    def create_job_text(self, job: Dict[str, Any]) -> str:
        """Create a text representation of a job for embedding"""
        return f"""
            `Job Title: {job.get('Job Title', '')}
            Company: {job.get('Company', '')}
            Location: {job.get('Location', '')}
            Job Type: {job.get('Job Type', '')}
            Work Mode: {job.get('Work Mode', '')}
            Seniority: {job.get('Seniority', '')}
            Salary: {job.get('Salary', '')}
            Experience: {job.get('Experience', '')}
            Responsibilities: {job.get('Responsibilities', '')}
            Qualifications: {job.get('Qualifications', '')}
            Skills: {job.get('Skills', '')}
        """.strip()

    def extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from job text using OpenAI"""
        try:
            prompt = f"""
                Extract all technical skills, tools, frameworks, programming languages, and relevant job skills from the text below.
                Return ONLY a comma-separated list of skills with no other text or explanation.
                Example: Python, JavaScript, React, AWS, Project Management, Agile

                Text:
                {text}
            """
                
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You extract skills from job descriptions accurately."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=200
            )
            
            skills_text = response.choices[0].message.content.strip()
            skills_list = [skill.strip() for skill in skills_text.split(',') if skill.strip()]
            logger.info(f"Extracted {len(skills_list)} skills")
            return skills_list
        
        except Exception as e:
            logger.error(f"Error extracting skills: {str(e)}")
            return []

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text using OpenAI API"""
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            embedding = response.data[0].embedding
            return embedding
        
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise

    def process_job_data(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single job and generate its embedding"""
        try:
            # Create a unique job identifier with ASCII-only characters
            job_title = job.get('Job Title', '').lower()
            company = job.get('Company', '').lower()
            location = job.get('Location', '').lower()
            job_type = job.get('Job Type', '').lower()
            
            # Remove special characters and replace spaces with underscores
            def clean_text(text):
                # Remove special characters and keep only alphanumeric and spaces
                cleaned = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
                # Replace spaces with underscores and remove consecutive underscores
                return '_'.join(filter(None, cleaned.split()))
            
            job_title_clean = clean_text(job_title)
            company_clean = clean_text(company)
            location_clean = clean_text(location)
            job_type_clean = clean_text(job_type)
            
            # Create a unique vector ID that doesn't change
            vector_id = f"job_{job_title_clean}_{company_clean}_{location_clean}_{job_type_clean}"
            
            # Create job text for embedding
            job_text = self.create_job_text(job)
            
            # Extract skills from job text
            extracted_skills = self.extract_skills_from_text(job_text)
            
            # Get embedding for the job text
            embedding = self.get_embedding(job_text)
            
            # Create metadata
            metadata = {
                "source": "job",
                "job_title": job.get('Job Title', ''),
                "company": job.get('Company', ''),
                "location": job.get('Location', ''),
                "job_type": job.get('Job Type', ''),
                "work_mode": job.get('Work Mode', ''),
                "seniority": job.get('Seniority', ''),
                "salary": job.get('Salary', ''),
                "experience": job.get('Experience', ''),
                "responsibilities": job.get('Responsibilities', ''),
                "qualifications": job.get('Qualifications', ''),
                "skills": job.get('Skills', ''),
                "extracted_skills": extracted_skills,
                "job_text": job_text
            }
            
            return {
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing job data: {str(e)}")
            raise

    def upsert_to_pinecone(self, vector_data: List[Dict[str, Any]]) -> int:
        """Upsert vectors to Pinecone with duplicate handling"""
        try:
            # Track seen jobs to avoid duplicates
            seen_jobs = set()
            unique_vectors = []
            
            for vector in vector_data:
                job_key = vector["id"]
                if job_key not in seen_jobs:
                    seen_jobs.add(job_key)
                    unique_vectors.append(vector)
            
            # Upsert unique vectors to Pinecone
            if unique_vectors:
                self.index.upsert(vectors=unique_vectors)
                logger.info(f"Upserted {len(unique_vectors)} unique jobs to Pinecone")
            
            return len(unique_vectors)
            
        except Exception as e:
            logger.error(f"Error upserting to Pinecone: {str(e)}")
            raise

    def process_job_file(self, file_key: str) -> Dict[str, Any]:
        """Process a job file, create embeddings and upsert to Pinecone"""
        try:
            # Load job data from S3
            job_data = self.load_job_file_from_s3(file_key)
            
            # Process each job
            vector_data = []
            for i, job in enumerate(job_data):
                try:
                    processed_job = self.process_job_data(job)
                    vector_data.append(processed_job)
                    logger.info(f"Processed job {i+1}/{len(job_data)}: {job.get('Job Title', '')} at {job.get('Company', '')}")
                except Exception as job_error:
                    logger.error(f"Error processing job {i+1}/{len(job_data)}: {str(job_error)}")
                    continue
            
            # Save embeddings to S3
            embeddings_output = {
                "original_file": file_key,
                "processing_date": datetime.now().isoformat(),
                "total_jobs": len(job_data),
                "processed_jobs": len(vector_data),
                "vectors": vector_data
            }
            embeddings_url = self.save_embeddings_to_s3(file_key, embeddings_output)
            
            # Upsert to Pinecone
            vectors_upserted = self.upsert_to_pinecone(vector_data)
            
            return {
                "status": "success",
                "original_file": file_key,
                "embeddings_url": embeddings_url,
                "total_jobs": len(job_data),
                "processed_jobs": len(vector_data),
                "vectors_upserted": vectors_upserted
            }
        
        except Exception as e:
            logger.error(f"Error processing job file {file_key}: {str(e)}")
            return {
                "status": "error",
                "original_file": file_key,
                "error": str(e)
            }

    def process_all_job_files(self) -> Dict[str, Any]:
        """Process all job files from S3 and generate embeddings"""
        try:
            # Get list of job files
            job_files = self.list_s3_job_files()
            if not job_files:
                logger.warning("No job files found in S3")
                return {"status": "error", "message": "No job files found"}
            
            # Track all processed jobs to avoid duplicates
            all_processed_jobs = []
            seen_jobs = set()
            
            # Process each file
            for file_key in job_files:
                logger.info(f"Processing file: {file_key}")
                job_data = self.load_job_file_from_s3(file_key)
                
                # Process each job in the file
                for job in job_data:
                    # Create a unique job identifier using more fields
                    job_title = job.get('Job Title', '').lower().strip()
                    company = job.get('Company', '').lower().strip()
                    location = job.get('Location', '').lower().strip()
                    job_type = job.get('Job Type', '').lower().strip()
                    
                    # Convert to ASCII-only and replace spaces with underscores
                    def clean_text(text):
                        # Remove special characters and keep only alphanumeric and spaces
                        cleaned = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
                        # Replace spaces with underscores and remove consecutive underscores
                        return '_'.join(filter(None, cleaned.split()))
                    
                    job_title_clean = clean_text(job_title)
                    company_clean = clean_text(company)
                    location_clean = clean_text(location)
                    job_type_clean = clean_text(job_type)
                    
                    # Create a unique vector ID that doesn't change
                    vector_id = f"job_{job_title_clean}_{company_clean}_{location_clean}_{job_type_clean}"
                    
                    # Skip if we've seen this exact job before
                    if vector_id in seen_jobs:
                        logger.debug(f"Skipping duplicate job: {vector_id}")
                        continue
                    
                    seen_jobs.add(vector_id)
                    
                    # Create job text for embedding
                    job_text = self.create_job_text(job)
                    
                    # Extract skills from job text
                    extracted_skills = self.extract_skills_from_text(job_text)
                    
                    # Get embedding for the job text
                    embedding = self.get_embedding(job_text)
                    
                    # Create metadata
                    metadata = {
                        "source": "job",
                        "job_title": job.get('Job Title', ''),
                        "company": job.get('Company', ''),
                        "location": job.get('Location', ''),
                        "job_type": job.get('Job Type', ''),
                        "work_mode": job.get('Work Mode', ''),
                        "seniority": job.get('Seniority', ''),
                        "salary": job.get('Salary', ''),
                        "experience": job.get('Experience', ''),
                        "responsibilities": job.get('Responsibilities', ''),
                        "qualifications": job.get('Qualifications', ''),
                        "skills": job.get('Skills', ''),
                        "extracted_skills": extracted_skills,
                        "job_text": job_text
                    }
                    
                    # Create vector data
                    vector_data = {
                        "id": vector_id,
                        "values": embedding,
                        "metadata": metadata
                    }
                    
                    all_processed_jobs.append(vector_data)
                    logger.info(f"Processed job: {job.get('Job Title')} at {job.get('Company')}")
            
            # Upsert all unique jobs to Pinecone
            if all_processed_jobs:
                # First, try to delete all vectors in the namespace if it exists
                try:
                    self.index.delete(delete_all=True, namespace=self.pinecone_namespace)
                    logger.info(f"Cleared existing vectors in namespace: {self.pinecone_namespace}")
                except Exception as e:
                    logger.info(f"No existing vectors to clear in namespace: {self.pinecone_namespace}")
                
                # Now upsert the new vectors
                total_upserted = self.upsert_to_pinecone(all_processed_jobs)
                logger.info(f"Successfully processed and upserted {total_upserted} unique jobs")
                
                return {
                    "status": "success",
                    "total_jobs_processed": len(all_processed_jobs),
                    "total_jobs_upserted": total_upserted,
                    "message": f"Successfully processed {len(all_processed_jobs)} jobs and upserted {total_upserted} unique jobs to Pinecone"
                }
            else:
                return {
                    "status": "error",
                    "message": "No jobs were processed"
                }
            
        except Exception as e:
            logger.error(f"Error processing job files: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def query_jobs(self, query_text: str, top_k: int = 10) -> Dict[str, Any]:
        """Query Pinecone for jobs matching the input text"""
        try:
            # Generate embedding for query
            query_embedding = self.get_embedding(query_text)
            
            # Query Pinecone
            query_result = self.index.query(
                vector=query_embedding,
                filter={"source": "job"},
                top_k=top_k,
                include_metadata=True
            )
            
            matches = []
            for match in query_result.matches:
                matches.append({
                    "id": match.id,
                    "score": match.score,
                    "job_title": match.metadata.get("job_title", ""),
                    "company": match.metadata.get("company", ""),
                    "location": match.metadata.get("location", ""),
                    "skills": match.metadata.get("skills", ""),
                    "extracted_skills": match.metadata.get("extracted_skills", [])
                })
            
            return {
                "status": "success",
                "query": query_text,
                "total_matches": len(matches),
                "matches": matches
            }
        
        except Exception as e:
            logger.error(f"Error querying jobs: {str(e)}")
            return {
                "status": "error",
                "query": query_text,
                "error": str(e)
            }

    def process_github_markdown(self, markdown_url: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Process GitHub markdown content to create embedding"""
        try:
            # Extract the key from the S3 URL
            key = markdown_url.replace(f"s3://{self.aws_bucket_name}/", "")
            
            # Get the markdown content from S3
            response = self.s3_client.get_object(
                Bucket=self.aws_bucket_name,
                Key=key
            )
            markdown_content = response['Body'].read().decode('utf-8')
            
            # Generate embedding
            embedding = self.get_embedding(markdown_content)
            
            # Create metadata
            metadata = {
                "source": "github",
                "content_type": "markdown",
                "processing_date": datetime.now().isoformat(),
                "user_id": user_id
            }
            
            # Generate unique vector ID
            timestamp = int(datetime.now().timestamp())
            vector_id = f"github_{user_id}_{timestamp}" if user_id else f"github_{timestamp}"
            
            return {
                "id": vector_id,
                "embedding": embedding,
                "metadata": metadata,
                "text": markdown_content
            }
            
        except Exception as e:
            logger.error(f"Error processing GitHub markdown: {str(e)}")
            raise

    def process_resume_markdown(self, markdown_url: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Process resume markdown content to create embedding"""
        try:
            # Extract the key from the S3 URL
            key = markdown_url.replace(f"s3://{self.aws_bucket_name}/", "")
            
            # Get the markdown content from S3
            response = self.s3_client.get_object(
                Bucket=self.aws_bucket_name,
                Key=key
            )
            markdown_content = response['Body'].read().decode('utf-8')
            
            # Generate embedding
            embedding = self.get_embedding(markdown_content)
            
            # Create metadata
            metadata = {
                "source": "resume",
                "content_type": "markdown",
                "processing_date": datetime.now().isoformat(),
                "user_id": user_id
            }
            
            # Generate unique vector ID
            timestamp = int(datetime.now().timestamp())
            vector_id = f"resume_{user_id}_{timestamp}" if user_id else f"resume_{timestamp}"
            
            return {
                "id": vector_id,
                "embedding": embedding,
                "metadata": metadata,
                "text": markdown_content
            }
            
        except Exception as e:
            logger.error(f"Error processing resume markdown: {str(e)}")
            raise

    def find_matching_jobs(self, query_embedding: Optional[List[float]] = None, query_text: Optional[str] = None, top_k: int = 10) -> Dict[str, Any]:
        """Find matching jobs using Pinecone vector search"""
        try:
            # If no embedding is provided but query text is, generate embedding
            if query_embedding is None and query_text:
                logger.info(f"Generating embedding for query text: {query_text[:100]}...")
                query_embedding = self.get_embedding(query_text)
            elif query_embedding is None:
                raise ValueError("Either query_embedding or query_text must be provided")
            
            logger.info(f"Querying Pinecone with {'text' if query_text else 'embedding'} and top_k={top_k}")
            
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            logger.info(f"Found {len(results.matches)} matches from Pinecone")
            
            # Process results
            matches = []
            for match in results.matches:
                similarity = match.score
                category = "high" if similarity >= self.similarity_thresholds["high"] else \
                          "medium" if similarity >= self.similarity_thresholds["medium"] else \
                          "low" if similarity >= self.similarity_thresholds["low"] else "very_low"
                
                match_data = {
                    "id": match.id,
                    "similarity": similarity,
                    "category": category,
                    "metadata": match.metadata
                }
                logger.info(f"Match: {match.metadata.get('job_title', 'Unknown')} at {match.metadata.get('company', 'Unknown')} (similarity: {similarity:.2f}, category: {category})")
                matches.append(match_data)
            
            return {
                "status": "success",
                "total_matches": len(matches),
                "matches": matches
            }
            
        except Exception as e:
            logger.error(f"Error finding matching jobs: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "matches": []
            }

# Main execution to process job files if run directly
if __name__ == "__main__":
    try:
        processor = JobEmbeddingsProcessor()
        result = processor.process_all_job_files()
        print(json.dumps(result, indent=2))
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        sys.exit(1)
