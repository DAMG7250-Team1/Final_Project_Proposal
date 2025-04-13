import os
import sys
from datetime import timedelta
from pathlib import Path
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# Add backend to Python path
backend_dir = str(Path(__file__).parent.parent.parent)
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

# Import scraper and embedding processor
try:
    from jobs.scraper import scrape_jobs
    from jobs.embeddings import JobEmbeddingsProcessor
except ImportError as e:
    print(f"Import Error: {e}")
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
    description='Daily job scraping pipeline for SkillMatch',
    schedule_interval='0 0 * * *',  # Run daily at midnight
    start_date=days_ago(1),
    catchup=False,
    tags=['skillmatch', 'jobs', 'scraping'],
)

# Task 1: Scrape Jobs from JobRight
def scrape_jobs_task():
    try:
        success = scrape_jobs()
        if not success:
            raise Exception("Job scraping failed")
    except Exception as e:
        print(f"Error in job scraping: {str(e)}")
        raise

scrape_jobs_operator = PythonOperator(
    task_id='scrape_jobs',
    python_callable=scrape_jobs_task,
    dag=dag,
)

# Task 2: Generate and save embeddings (to both S3 and Pinecone)
def process_embeddings_task():
    processor = JobEmbeddingsProcessor()
    job_files = processor.list_s3_job_files()

    if not job_files:
        raise ValueError("No job files found in S3 to process.")

    results = []
    for file_key in job_files:
        try:
            result = processor.process_job_file(file_key)
            results.append(result)
        except Exception as e:
            print(f"Error processing file {file_key}: {e}")
            continue

    return results

process_embeddings_operator = PythonOperator(
    task_id='create_embeddings',
    python_callable=process_embeddings_task,
    dag=dag,
)

# DAG task order
scrape_jobs_operator >> process_embeddings_operator
