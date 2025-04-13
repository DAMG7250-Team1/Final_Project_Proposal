

import sys
import os
import json
import logging
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Fix the class name to match what's defined in the module
from jobs.embeddings import JobEmbeddingsProcessor


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# File path (local or s3)
S3_BUCKET = "skillmatchai"
S3_KEY = "jobs/raw_files/jobright_jobs_20250411_182925.json"

def read_from_s3(bucket, key):
    """Read data from S3 bucket"""
    try:
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        
        # Get the object from S3
        response = s3_client.get_object(Bucket=bucket, Key=key)
        
        # Read and return the data
        if key.endswith('.json'):
            return json.loads(response['Body'].read().decode('utf-8'))
        else:
            return response['Body'].read().decode('utf-8')
            
    except ClientError as e:
        logger.error(f"Error reading from S3: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error reading from S3: {str(e)}")
        raise

def run_job_embedding_test():
    try:
        logger.info(f"🚀 Starting embedding test for S3 job file: s3://{S3_BUCKET}/{S3_KEY}")

        # Initialize processor
        processor = JobEmbeddingsProcessor()

        # Read job JSON from S3 using our own function
        job_list = read_from_s3(S3_BUCKET, S3_KEY)
        logger.info(f"📦 Loaded {len(job_list)} job entries")

        vectors_upserted = 0

        for job in job_list:
            try:
                logger.info(f"\n🔹 Processing: {job['Job Title']} @ {job['Company']}")

                # Compose job text
                job_text = f"""
Job Title: {job.get("Job Title")}
Company: {job.get("Company")}
Location: {job.get("Location")}
Job Type: {job.get("Job Type")}
Work Mode: {job.get("Work Mode")}
Seniority: {job.get("Seniority")}
Experience: {job.get("Experience")}
Responsibilities: {job.get("Responsibilities")}
Qualifications: {job.get("Qualifications")}
Skills: {job.get("Skills")}
                """

                # Extract structured skills using OpenAI
                extracted_skills = processor.extract_skills_from_text(job_text)

                # Get embedding
                embedding = processor.get_embedding(job_text)

                # Build vector ID and metadata
                vector_id = f"job_{job['Job Title'].lower().replace(' ', '_')}_{int(datetime.now().timestamp())}"
                metadata = {
                    "source": "job",
                    "job_title": job.get("Job Title", ""),
                    "company": job.get("Company", ""),
                    "location": job.get("Location", ""),
                    "job_type": job.get("Job Type", ""),
                    "work_mode": job.get("Work Mode", ""),
                    "seniority": job.get("Seniority", ""),
                    "salary": job.get("Salary", ""),
                    "experience": job.get("Experience", ""),
                    "responsibilities": job.get("Responsibilities", ""),
                    "qualifications": job.get("Qualifications", ""),
                    "skills": list(extracted_skills),
                    "timestamp": int(datetime.now().timestamp())
                }

                # Store in Pinecone
                processor.index.upsert(vectors=[(vector_id, embedding, metadata)])
                logger.info(f"✅ Stored: {vector_id} with skills → {', '.join(extracted_skills)}")
                vectors_upserted += 1

            except Exception as job_err:
                logger.error(f"❌ Error processing job: {str(job_err)}")
                continue

        logger.info(f"\n🎉 Completed. Total job vectors stored: {vectors_upserted}")

    except Exception as e:
        logger.error(f"💥 Failed to run job embedding test: {str(e)}")

if __name__ == "__main__":
    run_job_embedding_test()
