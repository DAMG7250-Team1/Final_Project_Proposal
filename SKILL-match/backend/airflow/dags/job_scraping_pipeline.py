import os
import sys
from datetime import timedelta
from pathlib import Path
import logging
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
os.environ["MPS_FORCE_CPU"] = "1"
# Add backend directory to Python path BEFORE importing jobs.*
backend_path = Path(__file__).resolve().parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Now it's safe to import
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# Import your scraper and embedding processor
try:
    from jobs.scraper import scrape_jobs
    from jobs.embeddings import JobEmbeddingsProcessor
except ImportError as e:
    logger.error(f"ImportError: {e}")
    raise

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'job_scraping_pipeline',
    default_args=default_args,
    description='Job scraping and embedding pipeline (every 4 hours)',
    schedule='0 */6 * * *',  # Run at minute 0 of every 4th hour (0:00, 4:00, 8:00, 12:00, 16:00, 20:00)
    start_date=days_ago(1),
    catchup=False,
    tags=['skillmatch', 'scraper'],
)

def scrape_jobs_task():
    """Task to scrape jobs from various sources"""
    try:
        logger.info("Starting job scraping task")
        success = scrape_jobs()
        if not success:
            raise Exception("Job scraping failed")
        logger.info("Job scraping completed successfully")
    except Exception as e:
        logger.error(f"Error in job scraping task: {str(e)}")
        raise

def process_embeddings_task():
    """Task to process job files and create embeddings"""
    try:
        logger.info("Starting embeddings processing task")
        processor = JobEmbeddingsProcessor()
        job_files = processor.list_s3_job_files()
        
        if not job_files:
            logger.warning("No job files found to process")
            return []
            
        results = []
        for file_key in job_files:
            try:
                logger.info(f"Processing file: {file_key}")
                result = processor.process_job_file(file_key)
                results.append(result)
                logger.info(f"Successfully processed {file_key}")
            except Exception as e:
                logger.error(f"Failed to process {file_key}: {str(e)}")
                continue
                
        logger.info(f"Completed processing {len(results)} files")
        return results
    except Exception as e:
        logger.error(f"Error in embeddings processing task: {str(e)}")
        raise

# Define operators
scrape_operator = PythonOperator(
    task_id='scrape_jobs',
    python_callable=scrape_jobs_task,
    dag=dag,
)

embed_operator = PythonOperator(
    task_id='create_embeddings',
    python_callable=process_embeddings_task,
    dag=dag,
)

# Set task dependencies
scrape_operator >> embed_operator 