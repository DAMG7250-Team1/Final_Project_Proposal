# import os
# import sys
# import logging
# from datetime import datetime
# from pathlib import Path
# import boto3
# import json

# # Add the parent directory to the Python path
# sys.path.append(str(Path(__file__).parent.parent))

# from user.embeddings import SkillsEmbeddingsProcessor

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# def read_from_s3(bucket, key):
#     """Read data from S3 bucket."""
#     try:
#         s3_client = boto3.client(
#             's3',
#             aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
#             aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
#             region_name=os.getenv('AWS_REGION', 'us-east-1')
#         )
        
#         response = s3_client.get_object(Bucket=bucket, Key=key)
#         if key.endswith('.json'):
#             data = json.loads(response['Body'].read().decode('utf-8'))
#         else:
#             data = response['Body'].read().decode('utf-8')
#         return data
#     except Exception as e:
#         logger.error(f"Error reading from S3: {str(e)}")
#         raise

# def create_job_chunks(job_data):
#     """Create optimized chunks from job data for vector embeddings."""
#     chunks = []
    
#     # 1. Basic Information Chunk
#     basic_info = {
#         "text": f"""
#         Job Title: {job_data.get('Job Title', '')}
#         Company: {job_data.get('Company', '')}
#         Location: {job_data.get('Location', '')}
#         Job Type: {job_data.get('Job Type', '')}
#         Work Mode: {job_data.get('Work Mode', '')}
#         Seniority: {job_data.get('Seniority', '')}
#         """,
#         "metadata": {
#             "chunk_type": "basic_info",
#             "job_title": job_data.get('Job Title', ''),
#             "company": job_data.get('Company', ''),
#             "location": job_data.get('Location', '')
#         }
#     }
#     chunks.append(basic_info)
    
#     # 2. Requirements Chunk
#     requirements = {
#         "text": f"""
#         Experience: {job_data.get('Experience', '')}
#         Qualifications: {job_data.get('Qualifications', '')}
#         Skills: {job_data.get('Skills', '')}
#         """,
#         "metadata": {
#             "chunk_type": "requirements",
#             "job_title": job_data.get('Job Title', ''),
#             "experience": job_data.get('Experience', ''),
#             "skills": job_data.get('Skills', '')
#         }
#     }
#     chunks.append(requirements)
    
#     # 3. Responsibilities Chunk
#     responsibilities = {
#         "text": f"""
#         Responsibilities: {job_data.get('Responsibilities', '')}
#         """,
#         "metadata": {
#             "chunk_type": "responsibilities",
#             "job_title": job_data.get('Job Title', '')
#         }
#     }
#     chunks.append(responsibilities)
    
#     # 4. Compensation Chunk
#     compensation = {
#         "text": f"""
#         Salary: {job_data.get('Salary', '')}
#         Benefits: {job_data.get('Benefits', '')}
#         """,
#         "metadata": {
#             "chunk_type": "compensation",
#             "job_title": job_data.get('Job Title', ''),
#             "salary": job_data.get('Salary', '')
#         }
#     }
#     chunks.append(compensation)
    
#     return chunks

# def create_resume_chunks(resume_text):
#     """Create chunks from resume markdown for vector embeddings."""
#     chunks = []
    
#     # Split resume into sections
#     sections = resume_text.split('\n\n')
    
#     for i, section in enumerate(sections):
#         if section.strip():
#             chunk = {
#                 "text": section.strip(),
#                 "metadata": {
#                     "chunk_type": "resume_section",
#                     "section_index": i,
#                     "source": "resume"
#                 }
#             }
#             chunks.append(chunk)
    
#     return chunks

# def test_s3_to_pinecone_storage():
#     """Test chunking and storing job and resume data from S3 to Pinecone."""
#     try:
#         logger.info("Starting S3 to Pinecone storage test...")
        
#         # Initialize embeddings processor
#         embeddings_processor = SkillsEmbeddingsProcessor()
        
#         # Process job data
#         logger.info("\nProcessing job data...")
#         bucket = 'skillmatchai'
#         job_key = 'jobs/raw_files/jobright_jobs_20250411_151348.json'
#         logger.info(f"Reading job data from s3://{bucket}/{job_key}")
        
#         job_data = read_from_s3(bucket, job_key)
#         logger.info(f"Successfully read {len(job_data)} jobs from S3")
        
#         # Process each job
#         for job in job_data:
#             try:
#                 chunks = create_job_chunks(job)
#                 logger.info(f"\nProcessing chunks for job: {job.get('Job Title', '')}")
                
#                 for i, chunk in enumerate(chunks, 1):
#                     embedding = embeddings_processor._get_embeddings([chunk['text']])[0]
#                     vector_id = f"job_{job.get('Job Title', '').lower().replace(' ', '_')}_chunk_{i}_{int(datetime.now().timestamp())}"
                    
#                     metadata = {
#                         'source': 'job',
#                         'chunk_type': chunk['metadata']['chunk_type'],
#                         'job_title': job.get('Job Title', ''),
#                         'company': job.get('Company', ''),
#                         'timestamp': int(datetime.now().timestamp())
#                     }
#                     metadata.update(chunk['metadata'])
                    
#                     embeddings_processor.index.upsert(vectors=[(vector_id, embedding, metadata)])
#                     logger.info(f"✅ Stored job chunk {i} in Pinecone: {vector_id}")
                
#             except Exception as e:
#                 logger.error(f"Error processing job: {str(e)}")
#                 continue
        
#         # Process resume data
#         logger.info("\nProcessing resume data...")
#         resume_key = 'resume/markdown/20250411_150545_Dhrumil Y Patel ML engineer.md'
#         logger.info(f"Reading resume from s3://{bucket}/{resume_key}")
        
#         resume_text = read_from_s3(bucket, resume_key)
#         chunks = create_resume_chunks(resume_text)
#         logger.info(f"Created {len(chunks)} chunks from resume")
        
#         # Process each resume chunk
#         for i, chunk in enumerate(chunks, 1):
#             try:
#                 embedding = embeddings_processor._get_embeddings([chunk['text']])[0]
#                 vector_id = f"resume_chunk_{i}_{int(datetime.now().timestamp())}"
                
#                 metadata = {
#                     'source': 'resume',
#                     'chunk_type': chunk['metadata']['chunk_type'],
#                     'section_index': chunk['metadata']['section_index'],
#                     'timestamp': int(datetime.now().timestamp())
#                 }
                
#                 embeddings_processor.index.upsert(vectors=[(vector_id, embedding, metadata)])
#                 logger.info(f"✅ Stored resume chunk {i} in Pinecone: {vector_id}")
                
#             except Exception as e:
#                 logger.error(f"Error processing resume chunk: {str(e)}")
#                 continue
        
#         # Extract skills from resume for job matching
#         logger.info("\nExtracting skills from resume...")
#         resume_skills = embeddings_processor._extract_skills_from_text(resume_text)
#         logger.info(f"Extracted {len(resume_skills)} skills from resume: {', '.join(resume_skills)}")
        
#         # Find matching jobs using extracted skills
#         if resume_skills:
#             logger.info("\nFinding matching jobs based on resume skills...")
#             skills_list = list(resume_skills)
            
#             # Create embedding for skills
#             skills_text = " ".join(skills_list)
#             query_embedding = embeddings_processor._get_embeddings([skills_text])[0]
            
#             # Query Pinecone for matching jobs
#             job_matches = embeddings_processor.index.query(
#                 vector=query_embedding,
#                 filter={"source": "job"},
#                 top_k=5,
#                 include_metadata=True
#             )
            
#             # Log matching jobs
#             logger.info(f"Found {len(job_matches.matches)} matching jobs:")
#             for i, match in enumerate(job_matches.matches, 1):
#                 logger.info(f"\nMatch {i}:")
#                 logger.info(f"Job Title: {match.metadata.get('job_title', 'N/A')}")
#                 logger.info(f"Company: {match.metadata.get('company', 'N/A')}")
#                 logger.info(f"Similarity Score: {match.score}")
#                 logger.info(f"Job Skills: {match.metadata.get('skills', 'N/A')}")
#                 logger.info("-" * 50)
        
#         # Verify stored data in Pinecone
#         logger.info("\nVerifying stored data in Pinecone:")
        
#         # Query Pinecone for all vectors
#         query_response = embeddings_processor.index.query(
#             vector=[0.0] * 3072,  # Dummy vector
#             top_k=20,
#             include_metadata=True
#         )
        
#         # Print details
#         logger.info(f"\nFound {len(query_response.matches)} vectors in Pinecone:")
#         for i, match in enumerate(query_response.matches, 1):
#             logger.info(f"\nVector {i}:")
#             logger.info(f"ID: {match.id}")
#             logger.info(f"Similarity Score: {match.score}")
#             logger.info(f"Source: {match.metadata.get('source', 'N/A')}")
#             logger.info(f"Chunk Type: {match.metadata.get('chunk_type', 'N/A')}")
#             if match.metadata.get('source') == 'job':
#                 logger.info(f"Job Title: {match.metadata.get('job_title', 'N/A')}")
#                 logger.info(f"Company: {match.metadata.get('company', 'N/A')}")
#             elif match.metadata.get('source') == 'resume':
#                 logger.info(f"Section Index: {match.metadata.get('section_index', 'N/A')}")
#             logger.info("-" * 50)
        
#         logger.info("Test completed successfully!")
        
#     except Exception as e:
#         logger.error(f"Error during test: {str(e)}")
#         raise

# if __name__ == "__main__":
#     test_s3_to_pinecone_storage()




import os
import sys
import logging
import json
import boto3
from datetime import datetime
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from backend.jobs.embeddings import SkillsEmbeddingsProcessor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# S3 path for job data
S3_BUCKET = 'skillmatchai'
S3_JOB_KEY = 'jobs/raw_files/jobright_jobs_20250411_151348.json'

def load_job_json_from_s3(bucket, key):
    """Load job JSON data from S3"""
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
        
        # Load and return the JSON data
        return json.loads(response['Body'].read().decode('utf-8'))
        
    except Exception as e:
        logger.error(f"Error loading job data from S3: {str(e)}")
        raise

def test_job_ingestion_and_vector_storage():
    try:
        logger.info("🔁 Starting job ingestion and vector storage test...")

        # Load job data from S3
        logger.info(f"📥 Loading job data from s3://{S3_BUCKET}/{S3_JOB_KEY}")
        job_data_list = load_job_json_from_s3(S3_BUCKET, S3_JOB_KEY)
        logger.info(f"📦 Loaded {len(job_data_list)} job listings from S3")

        # Initialize embedding processor
        embeddings_processor = SkillsEmbeddingsProcessor()

        # Process each job
        for job in job_data_list:
            try:
                logger.info(f"\n🚀 Processing: {job['Job Title']} @ {job['Company']}")

                # Compose job text
                job_text = f"""
Job Title: {job.get('Job Title')}
Company: {job.get('Company')}
Location: {job.get('Location')}
Job Type: {job.get('Job Type')}
Work Mode: {job.get('Work Mode')}
Seniority: {job.get('Seniority')}
Experience: {job.get('Experience')}
Responsibilities: {job.get('Responsibilities')}
Qualifications: {job.get('Qualifications')}
Skills: {job.get('Skills')}
                """.strip()

                # Extract skills using LLM
                extracted_skills = embeddings_processor._extract_skills_from_text(job_text)

                # Generate embedding
                embedding = embeddings_processor._get_embeddings([job_text])[0]

                # Vector ID and metadata
                vector_id = f"job_{job['Job Title'].lower().replace(' ', '_')}_{int(datetime.now().timestamp())}"
                metadata = {
                    "source": "job",
                    "job_title": job.get("Job Title", ""),
                    "company": job.get("Company", ""),
                    "location": job.get("Location", ""),
                    "skills": list(extracted_skills),
                    "timestamp": int(datetime.now().timestamp())
                }

                # Store in Pinecone
                embeddings_processor.index.upsert(vectors=[(vector_id, embedding, metadata)])
                logger.info(f"✅ Stored job vector: {vector_id}")
                logger.info(f"🧠 Skills extracted: {', '.join(extracted_skills)}")

            except Exception as job_error:
                logger.error(f"❌ Error processing job: {str(job_error)}")
                continue

        logger.info("🎉 All job vectors stored successfully!")

        # OPTIONAL: Verify storage
        logger.info("🔍 Querying Pinecone to confirm vector insertions...")
        dummy_query = embeddings_processor._get_embeddings(["placeholder query"])[0]
        result = embeddings_processor.index.query(
            vector=dummy_query,
            filter={"source": "job"},
            top_k=5,
            include_metadata=True
        )

        logger.info(f"📊 Found {len(result.matches)} job vectors in Pinecone:")
        for i, match in enumerate(result.matches, 1):
            logger.info(f"\nMatch {i}:")
            logger.info(f"ID: {match.id}")
            logger.info(f"Title: {match.metadata.get('job_title', 'N/A')}")
            logger.info(f"Company: {match.metadata.get('company', 'N/A')}")
            logger.info(f"Skills: {match.metadata.get('skills', [])}")
            logger.info(f"Similarity Score: {match.score}")
            logger.info("-" * 50)

    except Exception as e:
        logger.error(f"💥 Test failed: {str(e)}")

if __name__ == "__main__":
    test_job_ingestion_and_vector_storage()
