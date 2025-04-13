import streamlit as st
import requests
import re
from io import BytesIO
import os
from dotenv import load_dotenv
import PyPDF2

# Load environment variables
load_dotenv()

# Get backend API URL from environment variables
BACKEND_API_URL = os.getenv('BACKEND_API_URL', 'http://localhost:8000')

def is_valid_github_url(url):
    """Validate if the provided URL is a valid GitHub profile URL."""
    # Remove @ symbol if present at the start
    url = url.lstrip('@')
    
    # Basic GitHub URL pattern that handles both profile and repository URLs
    github_pattern = r'^https?://(?:www\.)?github\.com/[a-zA-Z0-9-]+(?:/[a-zA-Z0-9-_]+)*/?$'
    
    # Check if URL matches the pattern
    if not re.match(github_pattern, url):
        return False
        
    # Additional validation: must have github.com/ in the URL
    if 'github.com/' not in url:
        return False
        
    # Get the username part (first segment after github.com/)
    try:
        parts = url.split('github.com/')
        if len(parts) != 2:
            return False
        username = parts[1].split('/')[0]
        if not username:  # Username cannot be empty
            return False
    except:
        return False
        
    return True

def validate_pdf(pdf_file):
    """Validate if the uploaded file is a valid PDF."""
    try:
        # Read the PDF content
        pdf_content = pdf_file.read()
        # Try to create a PDF reader object
        pdf = PyPDF2.PdfReader(BytesIO(pdf_content))
        # Check if PDF has at least one page
        if len(pdf.pages) < 1:
            return False, "The PDF file appears to be empty"
        # Reset file pointer for later use
        pdf_file.seek(0)
        return True, "Valid PDF"
    except Exception as e:
        return False, f"Invalid PDF file: {str(e)}"

def process_github_profile(github_url):
    """Process GitHub profile using backend API."""
    try:
        # API endpoint
        api_url = f"{BACKEND_API_URL}/api/process-github"
        
        # Make POST request to backend
        response = requests.post(
            api_url,
            json={"url": github_url}
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            error_detail = "Unknown error"
            try:
                error_data = response.json()
                error_detail = error_data.get('detail', str(error_data))
            except:
                error_detail = response.text
            return False, f"Error processing GitHub profile: {error_detail}"
            
    except requests.exceptions.ConnectionError:
        return False, f"Could not connect to the backend server at {BACKEND_API_URL}"
    except Exception as e:
        return False, f"An error occurred: {str(e)}"

def get_job_matches(resume_skills, github_skills):
    """Get job matches based on skills."""
    try:
        api_url = f"{BACKEND_API_URL}/api/match-jobs"
        
        response = requests.post(
            api_url,
            json={
                "resume_skills": resume_skills,
                "github_skills": github_skills
            }
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            error_detail = "Unknown error"
            try:
                error_data = response.json()
                error_detail = error_data.get('detail', str(error_data))
            except:
                error_detail = response.text
            return False, f"Error getting job matches: {error_detail}"
            
    except requests.exceptions.ConnectionError:
        return False, f"Could not connect to the backend server at {BACKEND_API_URL}"
    except Exception as e:
        return False, f"An error occurred: {str(e)}"

def display_job_match(match):
    """Display a single job match in an expandable section."""
    with st.expander(f"🔍 {match['job_title']} at {match['company']} (Score: {match['similarity_score']:.2f})"):
        st.markdown(f"""
        ### 🏢 Company: {match['company']}
        ### 📍 Location: {match['location']}
        ### 💼 Job Type: {match['job_type']}
        ### 🕒 Work Mode: {match['work_mode']}
        ### 🎯 Seniority: {match['seniority']}
        ### 💰 Salary: {match['salary']}
        ### 📅 Experience: {match['experience']}
        
        #### 📋 Responsibilities
        {match['responsibilities']}
        
        #### 🎓 Qualifications
        {match['qualifications']}
        
        #### 🛠 Skills
        {match['skills']}
        """)

def main():
    st.set_page_config(
        page_title="Profile Upload",
        page_icon="📄",
        layout="centered"
    )

    st.title("Upload Your Profile")
    st.markdown("""
    Please provide your GitHub profile URL and upload your resume in PDF format.
    This information will be used to analyze your skills and find matching jobs.
    """)

    # GitHub URL input
    github_url = st.text_input("GitHub Profile URL", placeholder="https://github.com/username")
    
    # File uploader for PDF
    uploaded_file = st.file_uploader("Upload Your Resume (PDF)", type=['pdf'])

    if st.button("Submit", type="primary"):
        if not github_url:
            st.error("Please enter your GitHub profile URL")
            return
            
        if not uploaded_file:
            st.error("Please upload your resume")
            return

        # Validate GitHub URL
        if not is_valid_github_url(github_url):
            st.error("Please enter a valid GitHub profile URL")
            return

        # Validate PDF before sending to backend
        is_valid_pdf, pdf_message = validate_pdf(uploaded_file)
        if not is_valid_pdf:
            st.error(f"Invalid PDF file: {pdf_message}")
            return

        try:
            with st.spinner("Processing your profile..."):
                # First process GitHub profile
                st.info("Processing GitHub profile...")
                github_success, github_result = process_github_profile(github_url)
                
                if not github_success:
                    st.error(github_result)
                    return
                
                # Then process resume
                st.info("Processing resume...")
                # Create form data for multipart upload
                files = {
                    'file': (uploaded_file.name, uploaded_file.getvalue(), 'application/pdf')
                }
                
                # API endpoint for resume
                api_url = f"{BACKEND_API_URL}/api/upload-resume"
                
                # Make POST request to backend
                response = requests.post(
                    api_url,
                    files=files,
                    params={'github_url': github_url}
                )
                
                # Check response
                if response.status_code == 200:
                    response_data = response.json()
                    
                    if response_data.get('status') == 'success' and 'data' in response_data:
                        st.success("Profile submitted successfully!")
                        st.balloons()
                        
                        # Display the submitted information
                        st.subheader("Submitted Information:")
                        
                        # GitHub Information
                        st.write("### GitHub Profile")
                        github_data = github_result['data']
                        st.write(f"Username: {github_data['username']}")
                        st.write(f"Repositories: {github_data['repository_count']}")
                        st.write(f"READMEs Processed: {github_data['readme_count']}")
                        st.write(f"GitHub Markdown: {github_data['markdown_url']}")
                        
                        # Resume Information
                        st.write("### Resume")
                        st.write(f"Filename: {response_data['data']['filename']}")
                        st.write(f"Resume S3 URL: {response_data['data']['resume_url']}")
                        st.write(f"Resume Markdown: {response_data['data']['markdown_url']}")
                        
                        # Extract skills from embeddings info
                        resume_skills = []
                        github_skills = []
                        
                        if 'embeddings_info' in response_data['data']:
                            resume_skills = response_data['data']['embeddings_info'].get('skills', [])
                        
                        if 'github_info' in response_data['data'] and 'embeddings_info' in response_data['data']['github_info']:
                            github_skills = response_data['data']['github_info']['embeddings_info'].get('skills', [])
                        
                        # Get job matches
                        st.info("Finding matching jobs...")
                        matches_success, matches_result = get_job_matches(resume_skills, github_skills)
                        
                        if matches_success and matches_result.get('status') == 'success':
                            st.success(f"Found {matches_result['total_matches']} matching jobs!")
                            
                            # Display job matches
                            st.subheader("Matching Jobs")
                            for match in matches_result['matches']:
                                display_job_match(match)
                        else:
                            st.error(f"Error finding job matches: {matches_result.get('error', 'Unknown error')}")
                    else:
                        st.error("Unexpected response format from server")
                        st.json(response_data)
                else:
                    error_detail = "Unknown error"
                    try:
                        error_data = response.json()
                        error_detail = error_data.get('detail', str(error_data))
                    except:
                        error_detail = response.text
                    
                    st.error(f"Error submitting profile: {error_detail}")
                    
        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to the backend server at {BACKEND_API_URL}. Please make sure the backend is running.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Full error details for debugging:")
            st.exception(e)

if __name__ == "__main__":
    main()
