# Final_Project_Proposal

# SkillMatch AI – Smart Resume &amp; GitHub-based Job Matchmaker

https://codelabs-preview.appspot.com/?file_id=1h_b5P8zsedPcjZYr0giGZfOCo9leV-A5NWtyHjpxJMc/edit?tab=t.0#0


# 1. Project Abstract
SkillMatch AI is a personalized job recommendation system leveraging the power of Large Language Models (LLMs), Multi-Agent Systems, and the Model Completion Protocol (MCP) to match candidates to the most relevant job roles. Users upload their resume (PDF) and GitHub profile URL. The system parses the resume, analyzes GitHub projects using a dedicated agent, and intelligently recommends job opportunities. Additional agents enhance the resume and generate customized cover letters. The project utilizes both structured (job listings, skill taxonomies) and unstructured data (resumes, GitHub content), orchestrated with Airflow, deployed via FastAPI, and tested thoroughly.


# Methodology

## Data Sources

Unstructured
Resumes (PDFs): User-uploaded resumes containing personal details, experience, skills, and education.
GitHub Repositories: Data scraped from GitHub profiles, including repository content, README files, and project details.
Job Descriptions: Scraped from job boards or public APIs (e.g., LinkedIn, Indeed).
Structured
Job Taxonomies: Standardized skills and job titles from sources on web.
Job Metadata: Role, industry, and required skills stored in databases.
Vector Embeddings: Generated from resumes and GitHub data using Instructor for semantic search.


## Technologies and Tools


Airflow: Used for scheduling tasks, including data ingestion and skill-job mapping updates.


FastAPI: Serves as the backend for handling requests and exposing various endpoints for job matching, resume enhancement, and cover letter generation.


MCP (Model Completion Protocol): 

Orchestrates the communication between agents to ensure efficient data flow and task completion.


Data Pipeline Design

Data ingestion from resumes, GitHub, and job descriptions via Airflow.
Data cleaning, transformation, and vectorization to prepare for job matching.
Matching job descriptions with candidate resumes using similarity scoring and vector embeddings.


## Data Processing and Transformation

Parse resumes and extract relevant skills and experience.
Analyze GitHub profiles for project details and technologies used.
Match resumes to job descriptions based on skill and tech stack alignment.


## Disclosures

WE ATTEST THAT WE HAVEN'T USED ANY OTHER STUDENTS' WORK IN OUR ASSIGNMENT AND ABIDE BY THE POLICIES LISTED IN THE STUDENT HANDBOOK
We acknowledge that all team members contributed equally and worked to present the final project provided in this submission. All participants played a role in crucial ways, and the results reflect our collective efforts.
Additionally we acknowledge we have leveraged use of AI along with the provided references for code updation, generating suggestions and debugging errors for the varied issues we faced through the development process.AI tools like we utilized:
ChatGPT
Perplexity
Cursor
Deepseek
Claude

# Team Members

Contributions

Husain

33.3%

Sahil Kasliwal

33.3%

Dhrumil Patel

33.3%



