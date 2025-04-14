import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name='us-east-1'
)

S3_BUCKET = 'skillmatchai'
S3_BASE_PATH = 'jobs/raw_files/'

def save_to_s3(data, filename):
    """
    Save data to S3 bucket
    """
    try:
        # Convert data to JSON string
        json_data = json.dumps(data, ensure_ascii=False)
        
        # Upload to S3
        s3_key = f"{S3_BASE_PATH}{filename}"
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=json_data,
            ContentType='application/json'
        )
        
        logger.info(f"Successfully saved data to s3://{S3_BUCKET}/{s3_key}")
        return f"s3://{S3_BUCKET}/{s3_key}"
        
    except ClientError as e:
        logger.error(f"Error saving to S3: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error saving to S3: {str(e)}")
        raise

def scrape_jobs():
    """
    Main function to scrape jobs and save to S3
    """
    try:
        logger.info("Starting job scraping...")
        
        # Initialize the Chrome driver
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        wait = WebDriverWait(driver, 20)
        
        # Navigate to JobRight homepage
        driver.get("https://jobright.ai")
        time.sleep(3)
        
        # Google account credentials
        GOOGLE_EMAIL = os.getenv('GOOGLE_EMAIL', "dhrumil2510@gmail.com")
        GOOGLE_PASSWORD = os.getenv('GOOGLE_PASSWORD', "Dhrumil@123")
        
        logger.info("Attempting to log in to JobRight.ai...")
        
        # Click the SIGN IN button
        sign_in_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), 'SIGN IN')]")))
        sign_in_button.click()
        time.sleep(2)
        
        # Enter email and password
        email_input = wait.until(EC.presence_of_element_located((By.ID, "basic_email")))
        email_input.send_keys(GOOGLE_EMAIL)
        password_input = wait.until(EC.presence_of_element_located((By.ID, "basic_password")))
        password_input.send_keys(GOOGLE_PASSWORD)
        
        # Submit login form
        sign_in_submit = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(@class, 'sign-in-button')]")))
        sign_in_submit.click()
        time.sleep(5)
        
        # Navigate to job listings
        logger.info("Navigating to job listings page...")
        driver.get("https://jobright.ai/jobs/recommend")
        time.sleep(5)
        
        # Scroll to load more job cards
        logger.info("Scrolling to load more job listings...")
        for _ in range(10):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
        
        # Get all jobs after scrolling
        jobs = driver.find_elements(By.CLASS_NAME, "index_front__oxzpW")
        logger.info(f"Found {len(jobs)} jobs on the page")
        
        # Close modal if it's open
        def close_existing_modal():
            try:
                modal = driver.find_element(By.CLASS_NAME, "ant-modal-wrap")
                if modal.is_displayed():
                    close_button = driver.find_element(By.CLASS_NAME, "ant-modal-close")
                    driver.execute_script("arguments[0].click();", close_button)
                    time.sleep(2)
            except:
                pass
        
        # Extract text by icon alt attribute
        def extract_by_alt_text(text):
            try:
                return driver.find_element(By.XPATH, f"//img[@alt='{text}']/following-sibling::span").text.strip()
            except:
                return "Not specified"
        
        # Scraping starts
        logger.info("Starting to scrape job listings...")
        job_listings = []
        jobs = driver.find_elements(By.CLASS_NAME, "index_front__oxzpW")
        
        for job in jobs[:50]:
            try:
                close_existing_modal()
        
                title_element = job.find_element(By.CLASS_NAME, "index_job-title__UjuEY")
                driver.execute_script("arguments[0].click();", title_element)
        
                # Wait for modal to fully load
                modal_title = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "index_job-title__sStdA")))
                title = modal_title.text.strip()
        
                company = driver.find_element(By.CLASS_NAME, "index_company-row__vOzgg").find_element(By.TAG_NAME, "strong").text.strip()
                location = extract_by_alt_text("position")
                job_type = extract_by_alt_text("time")
                work_mode = extract_by_alt_text("remote")
                seniority = extract_by_alt_text("seniority")
                salary = extract_by_alt_text("money")
                experience = extract_by_alt_text("date")
        
                # Responsibilities
                requirements = []
                try:
                    req_spans = driver.find_elements(By.XPATH, "//section[.//h2[text()='Responsibilities']]//span[contains(@class, 'index_listText__')]")
                    for span in req_spans:
                        if span.text.strip():
                            requirements.append(span.text.strip())
                except:
                    pass
        
                # Qualifications
                qualifications = []
                try:
                    qual_spans = driver.find_elements(By.XPATH, "//section[@id='skills-section']//div[contains(@class, 'index_flex-col__')]/div/span[contains(@class, 'index_listText__')]")
                    for span in qual_spans:
                        if span.text.strip():
                            qualifications.append(span.text.strip())
                except:
                    pass
        
                # Skills
                skills = []
                try:
                    skill_elements = driver.find_elements(By.CLASS_NAME, "index_qualification-tag__5ZiFf")
                    for skill in skill_elements:
                        if skill.text.strip():
                            skills.append(skill.text.strip())
                except:
                    pass
        
                # Log the job details
                logger.info(f"Scraped job: {title} @ {company}")
        
                job_listings.append({
                    "Job Title": title,
                    "Company": company,
                    "Location": location,
                    "Job Type": job_type,
                    "Work Mode": work_mode,
                    "Seniority": seniority,
                    "Salary": salary,
                    "Experience": experience,
                    "Responsibilities": ", ".join(requirements) if requirements else "Not specified",
                    "Qualifications": ", ".join(qualifications) if qualifications else "Not specified",
                    "Skills": ", ".join(skills) if skills else "Not specified"
                })
        
                close_existing_modal()
        
            except Exception as e:
                logger.error(f"Error scraping job: {str(e)}")
                continue
        
        # Clean up browser
        driver.quit()
        logger.info("Browser closed successfully")
        
        # Save to S3 if we have job listings
        if job_listings:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"jobright_jobs_{timestamp}.json"
            
            # Save to S3
            s3_url = save_to_s3(job_listings, filename)
            logger.info(f"Successfully saved {len(job_listings)} jobs to {s3_url}")
            
            return True
        else:
            logger.warning("No jobs were scraped.")
            return False
            
    except Exception as e:
        logger.error(f"Error during job scraping: {str(e)}")
        return False

if __name__ == "__main__":
    success = scrape_jobs()
    if success:
        logger.info("Job scraping completed successfully")
    else:
        logger.error("Job scraping failed")
        sys.exit(1)