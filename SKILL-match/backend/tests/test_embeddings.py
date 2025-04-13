import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from jobs.scraper import scrape_jobs, job_listings
from jobs.embeddings import SkillsEmbeddingsProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_job_scraping_and_embeddings():
    """Test job scraping and vector embeddings creation."""
    try:
        logger.info("Starting job scraping and embeddings test...")
        
        # Initialize embeddings processor
        embeddings_processor = SkillsEmbeddingsProcessor()
        
        # Scrape jobs
        logger.info("Scraping jobs...")
        success = scrape_jobs()
        
        if not success:
            logger.error("Job scraping failed")
            return
        
        # Verify jobs in Pinecone
        logger.info("Verifying jobs in Pinecone...")
        
        # Query Pinecone for all job vectors
        query_response = embeddings_processor.index.query(
            vector=[0.0] * 3072,  # Dummy vector
            filter={"source": "job"},
            top_k=100,
            include_metadata=True
        )
        
        # Print job details
        logger.info(f"\nFound {len(query_response.matches)} jobs in Pinecone:")
        for i, match in enumerate(query_response.matches, 1):
            logger.info(f"\nJob {i}:")
            logger.info(f"ID: {match.id}")
            logger.info(f"Similarity Score: {match.score}")
            logger.info(f"Metadata:")
            for key, value in match.metadata.items():
                logger.info(f"  {key}: {value}")
            logger.info("-" * 50)
        
        # Test job matching with different skill sets
        test_skill_sets = [
            ["Python", "AWS", "Docker", "Machine Learning"],
            ["React", "JavaScript", "HTML", "CSS"],
            ["Python", "React", "AWS", "Docker"]
        ]
        
        for skills in test_skill_sets:
            logger.info(f"\nTesting job matching for skills: {skills}")
            matches = embeddings_processor.find_matching_jobs(skills)
            
            logger.info(f"Total matches: {matches['total_matches']}")
            for i, match in enumerate(matches['matches'], 1):
                logger.info(f"\nMatch {i}:")
                logger.info(f"Title: {match['job_title']}")
                logger.info(f"Company: {match['company']}")
                logger.info(f"Similarity Score: {match['similarity_score']}")
                logger.info(f"Skills: {match['skills']}")
                logger.info("-" * 50)
        
        logger.info("Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        raise

if __name__ == "__main__":
    test_job_scraping_and_embeddings() 