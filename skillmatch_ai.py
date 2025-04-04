#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SkillMatch AI - A personalized job recommendation system
=======================================================

This script demonstrates the complete implementation of the SkillMatch AI system,
which analyzes resumes and GitHub profiles to match candidates with relevant job opportunities.

The system includes:
1. Data generation and preparation
2. Resume parsing from PDF
3. GitHub profile analysis
4. Job description parsing
5. Skills matching and recommendation
6. Resume enhancement suggestions
7. Cover letter generation

Author: Your Name
Date: April 2025
"""

# Import necessary libraries
import os
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from getpass import getpass
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import PyPDF2
from github import Github
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set plotting styles
plt.style.use('ggplot')
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# ==============================================================================
# API Key Setup
# ==============================================================================

def set_api_key():
    """Set up OpenAI API key from environment or prompt user for input"""
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")
    return os.environ["OPENAI_API_KEY"]

def set_github_api_key():
    """Set up GitHub API key from environment or prompt user for input"""
    if not os.getenv("GITHUB_TOKEN"):
        os.environ["GITHUB_TOKEN"] = getpass("Enter your GitHub personal access token: ")
    return os.environ["GITHUB_TOKEN"]

# ==============================================================================
# Data Generation and Preparation
# ==============================================================================

def generate_sample_data():
    """Generate synthetic datasets for demonstration purposes"""
    print("Generating sample data...")
    
    # Generate sample resume data
    sample_resumes = [
        {
            "id": 1,
            "name": "Alex Johnson",
            "email": "alex.johnson@example.com",
            "location": "San Francisco, CA",
            "experience_level": "Mid-level",
            "years_experience": 4,
            "technical_skills": ["Python", "Django", "PostgreSQL", "JavaScript", "React", "Git", "Docker"],
            "soft_skills": ["Team Leadership", "Communication", "Problem-solving"],
            "education": {
                "degree": "Bachelor's in Computer Science",
                "institution": "University of California, Berkeley",
                "graduation_year": 2019
            },
            "github_username": "alexjcoder"
        },
        {
            "id": 2,
            "name": "Taylor Smith",
            "email": "taylor.smith@example.com",
            "location": "Seattle, WA",
            "experience_level": "Senior",
            "years_experience": 8,
            "technical_skills": ["Java", "Spring Boot", "AWS", "Kubernetes", "Microservices", "MongoDB", "Redis"],
            "soft_skills": ["Project Management", "Mentoring", "Cross-functional Collaboration"],
            "education": {
                "degree": "Master's in Software Engineering",
                "institution": "University of Washington",
                "graduation_year": 2015
            },
            "github_username": "tsmith-dev"
        },
        {
            "id": 3,
            "name": "Sam Rivera",
            "email": "sam.rivera@example.com",
            "location": "Austin, TX",
            "experience_level": "Entry-level",
            "years_experience": 1,
            "technical_skills": ["Python", "JavaScript", "HTML", "CSS", "React", "Node.js", "SQL"],
            "soft_skills": ["Time Management", "Adaptability", "Eagerness to Learn"],
            "education": {
                "degree": "Bachelor's in Software Development",
                "institution": "University of Texas at Austin",
                "graduation_year": 2023
            },
            "github_username": "samriver"
        },
        {
            "id": 4,
            "name": "Jordan Lee",
            "email": "jordan.lee@example.com",
            "location": "New York, NY",
            "experience_level": "Mid-level",
            "years_experience": 5,
            "technical_skills": ["JavaScript", "TypeScript", "React", "Vue.js", "Node.js", "Express", "MongoDB", "CI/CD"],
            "soft_skills": ["Product Thinking", "User Empathy", "Cross-team Collaboration"],
            "education": {
                "degree": "Bachelor's in Information Technology",
                "institution": "New York University",
                "graduation_year": 2018
            },
            "github_username": "jlee-dev"
        },
        {
            "id": 5,
            "name": "Morgan Chen",
            "email": "morgan.chen@example.com",
            "location": "Boston, MA",
            "experience_level": "Senior",
            "years_experience": 10,
            "technical_skills": ["Python", "TensorFlow", "PyTorch", "Scikit-learn", "NLP", "Computer Vision", "MLOps", "AWS"],
            "soft_skills": ["Research", "Technical Writing", "Team Leadership", "Public Speaking"],
            "education": {
                "degree": "PhD in Machine Learning",
                "institution": "Massachusetts Institute of Technology",
                "graduation_year": 2017
            },
            "github_username": "morganchen-ai"
        }
    ]
    
    # Generate sample GitHub profile data
    sample_github_profiles = [
        {
            "username": "alexjcoder",
            "public_repos": 12,
            "followers": 45,
            "top_languages": ["Python", "JavaScript", "HTML", "CSS", "TypeScript"],
            "repositories": [
                {"name": "django-ecommerce", "stars": 28, "forks": 5, "languages": ["Python", "JavaScript", "HTML"], 
                 "description": "A full-featured e-commerce platform built with Django and React"},
                {"name": "react-dashboard", "stars": 17, "forks": 3, "languages": ["JavaScript", "CSS"],
                 "description": "A responsive admin dashboard built with React and Chart.js"},
                {"name": "personal-blog", "stars": 8, "forks": 2, "languages": ["JavaScript", "HTML", "CSS"],
                 "description": "My personal blog built with Next.js and MDX"}  
            ],
            "commit_frequency": 4.2,  # Average weekly commits
            "contribution_streak": 32,  # Days
            "frameworks_detected": ["Django", "React", "Bootstrap"]
        },
        {
            "username": "tsmith-dev",
            "public_repos": 25,
            "followers": 120,
            "top_languages": ["Java", "TypeScript", "Kotlin", "Python", "Shell"],
            "repositories": [
                {"name": "spring-boot-microservices", "stars": 87, "forks": 23, "languages": ["Java", "Shell"],
                 "description": "A complete microservices architecture built with Spring Boot"},
                {"name": "kubernetes-workshop", "stars": 64, "forks": 18, "languages": ["Shell", "YAML", "Python"],
                 "description": "Workshop materials for Kubernetes and container orchestration"},
                {"name": "java-design-patterns", "stars": 42, "forks": 12, "languages": ["Java"],
                 "description": "Implementation of common design patterns in Java"}
            ],
            "commit_frequency": 8.5,
            "contribution_streak": 78,
            "frameworks_detected": ["Spring Boot", "JUnit", "Maven", "Kafka"]
        },
        {
            "username": "samriver",
            "public_repos": 5,
            "followers": 8,
            "top_languages": ["JavaScript", "HTML", "CSS", "Python"],
            "repositories": [
                {"name": "portfolio-website", "stars": 3, "forks": 1, "languages": ["JavaScript", "HTML", "CSS"],
                 "description": "My personal portfolio website showcasing my projects"},
                {"name": "weather-app", "stars": 2, "forks": 0, "languages": ["JavaScript", "HTML", "CSS"],
                 "description": "Weather application using the OpenWeatherMap API"},
                {"name": "todo-list", "stars": 1, "forks": 0, "languages": ["JavaScript", "HTML", "CSS"],
                 "description": "Simple to-do list app using React and local storage"}
            ],
            "commit_frequency": 2.1,
            "contribution_streak": 8,
            "frameworks_detected": ["React", "Express", "Bootstrap"]
        },
        {
            "username": "jlee-dev",
            "public_repos": 18,
            "followers": 62,
            "top_languages": ["TypeScript", "JavaScript", "CSS", "HTML", "Python"],
            "repositories": [
                {"name": "vue-component-library", "stars": 35, "forks": 8, "languages": ["TypeScript", "Vue", "CSS"],
                 "description": "A library of reusable Vue 3 components with TypeScript"},
                {"name": "react-native-boilerplate", "stars": 27, "forks": 9, "languages": ["TypeScript", "JavaScript"],
                 "description": "Starter template for React Native apps with TypeScript"},
                {"name": "node-express-graphql", "stars": 19, "forks": 5, "languages": ["TypeScript", "JavaScript"],
                 "description": "GraphQL API server built with Node.js, Express, and TypeScript"}
            ],
            "commit_frequency": 5.7,
            "contribution_streak": 45,
            "frameworks_detected": ["React", "Vue.js", "Express", "GraphQL", "Jest"]
        },
        {
            "username": "morganchen-ai",
            "public_repos": 32,
            "followers": 235,
            "top_languages": ["Python", "Jupyter Notebook", "C++", "CUDA", "Shell"],
            "repositories": [
                {"name": "transformer-from-scratch", "stars": 320, "forks": 67, "languages": ["Python", "Jupyter Notebook"],
                 "description": "Implementing transformer architecture from scratch with PyTorch"},
                {"name": "computer-vision-toolkit", "stars": 178, "forks": 43, "languages": ["Python", "C++", "CUDA"],
                 "description": "Collection of computer vision algorithms and utilities with GPU acceleration"},
                {"name": "nlp-benchmarks", "stars": 142, "forks": 37, "languages": ["Python", "Jupyter Notebook"],
                 "description": "Benchmarking various NLP models on common tasks"}
            ],
            "commit_frequency": 7.2,
            "contribution_streak": 104,
            "frameworks_detected": ["PyTorch", "TensorFlow", "Scikit-learn", "Hugging Face Transformers"]
        }
    ]
    
    # Generate sample job listings
    sample_job_listings = [
        {
            "id": 1,
            "title": "Senior Backend Engineer",
            "company": "TechCorp Inc.",
            "location": "San Francisco, CA (Hybrid)",
            "experience_level": "Senior",
            "salary_range": "$150,000 - $180,000",
            "required_skills": ["Python", "Django", "PostgreSQL", "RESTful APIs", "Microservices", "Docker", "AWS"],
            "preferred_skills": ["Kubernetes", "GraphQL", "Redis", "CI/CD", "Terraform"],
            "description": "TechCorp is seeking a Senior Backend Engineer to design and develop scalable services for our platform.",
            "education": "Bachelor's degree in Computer Science or equivalent experience",
            "posting_date": "2025-03-15",
            "job_type": "Full-time"
        },
        {
            "id": 2,
            "title": "Frontend Developer",
            "company": "InnovateTech",
            "location": "Remote",
            "experience_level": "Mid-level",
            "salary_range": "$100,000 - $130,000",
            "required_skills": ["JavaScript", "React", "HTML", "CSS", "Responsive Design", "Git"],
            "preferred_skills": ["TypeScript", "Next.js", "Redux", "Storybook", "Testing (Jest, React Testing Library)"],
            "description": "InnovateTech is looking for a talented Frontend Developer to create engaging user interfaces for our web applications.",
            "education": "Bachelor's degree in Computer Science, Web Development, or related field",
            "posting_date": "2025-03-20",
            "job_type": "Full-time"
        },
        {
            "id": 3,
            "title": "Machine Learning Engineer",
            "company": "AI Solutions Ltd.",
            "location": "Boston, MA (On-site)",
            "experience_level": "Senior",
            "salary_range": "$160,000 - $200,000",
            "required_skills": ["Python", "TensorFlow", "PyTorch", "Computer Vision", "NLP", "Data Science", "ML Deployment"],
            "preferred_skills": ["MLOps", "Kubernetes", "AWS SageMaker", "GCP AI Platform", "CUDA", "Model Optimization"],
            "description": "AI Solutions is looking for a Machine Learning Engineer to develop and deploy cutting-edge ML models for our healthcare products.",
            "education": "Master's or PhD in Computer Science, Machine Learning, or related field",
            "posting_date": "2025-03-10",
            "job_type": "Full-time"
        },
        {
            "id": 4,
            "title": "DevOps Engineer",
            "company": "CloudScale Systems",
            "location": "Remote",
            "experience_level": "Mid-level",
            "salary_range": "$120,000 - $150,000",
            "required_skills": ["Kubernetes", "Docker", "AWS", "CI/CD", "Infrastructure as Code", "Linux", "Bash/Python"],
            "preferred_skills": ["Terraform", "Prometheus", "Grafana", "ELK Stack", "GitOps", "Istio"],
            "description": "CloudScale Systems is seeking a DevOps Engineer to build and maintain our cloud infrastructure and deployment pipelines.",
            "education": "Bachelor's degree in Computer Science or equivalent experience",
            "posting_date": "2025-03-18",
            "job_type": "Full-time"
        },
        {
            "id": 5,
            "title": "Full Stack Developer",
            "company": "WebSolutions Inc.",
            "location": "Austin, TX (Hybrid)",
            "experience_level": "Mid-level",
            "salary_range": "$110,000 - $140,000",
            "required_skills": ["JavaScript", "Python", "React", "Node.js", "PostgreSQL", "REST APIs", "Git"],
            "preferred_skills": ["TypeScript", "GraphQL", "Redux", "Docker", "AWS", "MongoDB"],
            "description": "WebSolutions Inc. is looking for a Full Stack Developer to work on our client-facing web applications and internal tools.",
            "education": "Bachelor's degree in Computer Science or related field",
            "posting_date": "2025-03-22",
            "job_type": "Full-time"
        }
    ]
    
    # Create a data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Save the data
    with open('data/sample_resumes.pkl', 'wb') as f:
        pickle.dump(sample_resumes, f)
        
    with open('data/sample_github_profiles.pkl', 'wb') as f:
        pickle.dump(sample_github_profiles, f)
        
    with open('data/sample_job_listings.pkl', 'wb') as f:
        pickle.dump(sample_job_listings, f)
    
    print("Sample data generated and saved successfully.")
    return sample_resumes, sample_github_profiles, sample_job_listings

def load_sample_data():
    """Load sample data if it exists, otherwise generate it"""
    try:
        with open('data/sample_resumes.pkl', 'rb') as f:
            sample_resumes = pickle.load(f)
            
        with open('data/sample_github_profiles.pkl', 'rb') as f:
            sample_github_profiles = pickle.load(f)
            
        with open('data/sample_job_listings.pkl', 'rb') as f:
            sample_job_listings = pickle.load(f)
            
        print("Sample data loaded successfully.")
        return sample_resumes, sample_github_profiles, sample_job_listings
    except FileNotFoundError:
        print("Sample data files not found. Generating new sample data...")
        return generate_sample_data()

# ==============================================================================
# Resume Parser Component
# ==============================================================================

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def parse_resume_with_llm(resume_text):
    """Use LLM to extract structured information from resume text"""
    system_prompt = """
    You are a resume parser. Extract the following information from the resume text in JSON format:
    - personal_info: name, email, phone, location
    - skills: technical_skills (list), soft_skills (list)
    - experience: list of positions with company, title, start_date, end_date, description, and skills_used
    - education: list of degrees with institution, degree, field, graduation_date, and gpa if available
    - projects: list of projects with title, description, and technologies_used
    
    Format your response as valid JSON only, with no additional text or explanation.
    """
    
    try:
        api_key = set_api_key()
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4o or available model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": resume_text}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error parsing resume with LLM: {e}")
        return {}

# ==============================================================================
# GitHub Profile Analyzer
# ==============================================================================

def analyze_github_profile(username):
    """Analyze GitHub profile to extract skills and projects"""
    github_token = set_github_api_key()
    g = Github(github_token)
    
    try:
        # Get user information
        user = g.get_user(username)
        user_data = {
            "name": user.name,
            "username": user.login,
            "profile_url": user.html_url,
            "public_repos": user.public_repos,
            "followers": user.followers
        }
        
        # Get repositories information
        repos = user.get_repos()
        
        # Extract languages and repositories data
        languages_counter = Counter()
        repositories = []
        
        for repo in repos:
            if not repo.fork:  # Skip forked repositories
                # Add languages to counter
                for lang, bytes_code in repo.get_languages().items():
                    languages_counter[lang] += bytes_code
                
                repositories.append({
                    "name": repo.name,
                    "description": repo.description,
                    "url": repo.html_url,
                    "stars": repo.stargazers_count,
                    "forks": repo.forks_count,
                    "languages": list(repo.get_languages().keys())
                })
        
        # Calculate most used languages
        top_languages = [lang for lang, _ in languages_counter.most_common(10)]
        
        # Sort repositories by stars
        repositories.sort(key=lambda x: x["stars"], reverse=True)
        
        return {
            "user_info": user_data,
            "top_languages": top_languages,
            "repositories": repositories
        }
        
    except Exception as e:
        print(f"Error analyzing GitHub profile: {e}")
        return {}

def extract_github_skills(github_data, resume_skills):
    """Extract and enhance skills from GitHub data"""
    
    if not github_data:
        return []
    
    # Base skills from languages
    skills = set(github_data.get("top_languages", []))
    
    # Extract additional skills from repositories
    frameworks_libraries = {
        "Python": ["Django", "Flask", "FastAPI", "Pandas", "NumPy", "TensorFlow", "PyTorch", "Scikit-learn"],
        "JavaScript": ["React", "Vue", "Angular", "Node.js", "Express", "Next.js", "jQuery"],
        "Java": ["Spring", "Hibernate", "Maven", "Gradle", "JUnit"],
        "C#": ["ASP.NET", ".NET Core", "Entity Framework", "LINQ"],
        "PHP": ["Laravel", "Symfony", "CodeIgniter", "WordPress"],
        "Ruby": ["Rails", "Sinatra", "RSpec"],
        "TypeScript": ["Angular", "React", "Vue", "NestJS"],
        "Go": ["Gin", "Echo", "Gorilla"],
        "Rust": ["Actix", "Rocket", "Tokio"],
        "Swift": ["UIKit", "SwiftUI", "Core Data"],
        "Kotlin": ["Android SDK", "Ktor", "Jetpack Compose"]
    }
    
    # Add potential frameworks based on languages
    for lang in github_data.get("top_languages", []):
        if lang in frameworks_libraries:
            for repo in github_data.get("repositories", []):
                description = repo.get("description", "").lower() if repo.get("description") else ""
                for framework in frameworks_libraries[lang]:
                    if framework.lower() in description:
                        skills.add(framework)
    
    # Combine with resume skills
    for skill in resume_skills:
        skills.add(skill)
    
    return list(skills)

def analyze_github_repo_with_llm(repo_info):
    """Use LLM to analyze a GitHub repository and extract skills and concepts"""
    system_prompt = """
    You are a technical skills analyzer. Given information about a GitHub repository,
    extract the following in JSON format:
    - primary_skills: List of core technical skills demonstrated in this project
    - secondary_skills: List of supporting or peripheral skills likely used
    - concepts: List of software development concepts demonstrated (e.g., "CI/CD", "Microservices", "Authentication")
    - project_complexity: A rating from 1-5 of the project's technical complexity
    - categorization: Main category of the project (e.g., "Web Application", "Data Science", "DevOps")
    
    Format your response as valid JSON only, with no additional text or explanation.
    """
    
    user_prompt = f"""
    Repository Details:
    - Name: {repo_info.get('name', 'Unknown')}
    - Description: {repo_info.get('description', 'No description')}
    - Main Languages: {', '.join(repo_info.get('languages', []))}
    - Stars: {repo_info.get('stars', 0)}
    - Forks: {repo_info.get('forks', 0)}
    """
    
    try:
        api_key = set_api_key()
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error analyzing repository with LLM: {e}")
        return {}

# ==============================================================================
# Job Description Parser
# ==============================================================================

def parse_job_description(job_text):
    """Parse job description to extract structured information"""
    system_prompt = """
    You are a job description parser. Extract the following information from the job description in JSON format:
    - title: the job title
    - company: the company name
    - location: job location (remote, hybrid, or office location)
    - required_skills: list of technical skills required
    - preferred_skills: list of preferred or nice-to-have skills
    - experience_level: entry, mid, senior, etc.
    - education: minimum education requirements
    - job_type: full-time, part-time, contract, etc.
    - description: a brief summary of the role
    
    Format your response as valid JSON only, with no additional text or explanation.
    """
    
    try:
        api_key = set_api_key()
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": job_text}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error parsing job description with LLM: {e}")
        return {}

# ==============================================================================
# Comprehensive Profile Creation
# ==============================================================================

def create_comprehensive_profile(resume_data, github_data, repo_analysis=None):
    """Combine resume and GitHub data into a comprehensive candidate profile"""
    
    # Extract personal info from resume
    if 'personal_info' in resume_data:
        personal_info = resume_data['personal_info']
    else:
        personal_info = {
            'name': 'Unknown',
            'email': 'unknown@example.com',
            'location': 'Unknown'
        }
    
    # Extract skills
    technical_skills = set()
    if 'skills' in resume_data and 'technical_skills' in resume_data['skills']:
        technical_skills.update(resume_data['skills']['technical_skills'])
    
    soft_skills = set()
    if 'skills' in resume_data and 'soft_skills' in resume_data['skills']:
        soft_skills.update(resume_data['skills']['soft_skills'])
    
    # Add GitHub skills
    if github_data and 'top_languages' in github_data:
        technical_skills.update(github_data['top_languages'])
    
    # Add project concepts from repo analysis
    if repo_analysis and 'concepts' in repo_analysis:
        technical_skills.update(repo_analysis['concepts'])
    
    if repo_analysis and 'primary_skills' in repo_analysis:
        technical_skills.update(repo_analysis['primary_skills'])
    
    if repo_analysis and 'secondary_skills' in repo_analysis:
        technical_skills.update(repo_analysis['secondary_skills'])
    
    # Extract experience from resume
    experience = []
    if 'experience' in resume_data:
        experience = resume_data['experience']
    
    # Extract education from resume
    education = []
    if 'education' in resume_data:
        education = resume_data['education']
    
    # Extract projects from resume and enhance with GitHub data
    projects = []
    if 'projects' in resume_data:
        projects = resume_data['projects']
    
    # Add GitHub projects
    if github_data and 'repositories' in github_data:
        for repo in github_data['repositories']:
            # Check if the project already exists in the resume
            if not any(p.get('title', '').lower() == repo['name'].lower() for p in projects):
                projects.append({
                    'title': repo['name'],
                    'description': repo['description'] or f"GitHub project with {repo['stars']} stars",
                    'technologies_used': repo['languages'],
                    'url': repo.get('url', '')
                })
    
    # Create GitHub activity metrics
    github_activity = {}
    if github_data:
        github_activity = {
            'repositories': len(github_data.get('repositories', [])),
            'stars': sum(repo.get('stars', 0) for repo in github_data.get('repositories', [])),
            'forks': sum(repo.get('forks', 0) for repo in github_data.get('repositories', [])),
            'followers': github_data.get('user_info', {}).get('followers', 0)
        }
    
    # Determine experience level
    experience_years = 0
    for job in experience:
        if 'start_date' in job and 'end_date' in job:
            # This is simplified - in a real app, you'd parse dates and calculate properly
            experience_years += 1
    
    experience_level = 'Entry-level'
    if experience_years >= 5:
        experience_level = 'Senior'
    elif experience_years >= 2:
        experience_level = 'Mid-level'
    
    # Create the comprehensive profile
    comprehensive_profile = {
        'personal_info': personal_info,
        'technical_skills': sorted(list(technical_skills)),
        'soft_skills': sorted(list(soft_skills)),
        'experience': experience,
        'education': education,
        'projects': projects,
        'github_activity': github_activity,
        'experience_level': experience_level,
        'years_experience': experience_years
    }
    
    return comprehensive_profile

# ==============================================================================
# Skills Matching and Scoring
# ==============================================================================

def match_skills(candidate_skills, job_skills):
    """Match candidate skills with job skills and return a score"""
    # Convert skills lists to lowercase for better matching
    candidate_skills_lower = [skill.lower() for skill in candidate_skills]
    required_skills_lower = [skill.lower() for skill in job_skills.get('required_skills', [])]
    preferred_skills_lower = [skill.lower() for skill in job_skills.get('preferred_skills', [])]
    
    # Count matching required skills
    matched_required = [skill for skill in required_skills_lower if any(candidate_skill in skill or skill in candidate_skill for candidate_skill in candidate_skills_lower)]
    required_match_ratio = len(matched_required) / len(required_skills_lower) if required_skills_lower else 0
    
    # Count matching preferred skills
    matched_preferred = [skill for skill in preferred_skills_lower if any(candidate_skill in skill or skill in candidate_skill for candidate_skill in candidate_skills_lower)]
    preferred_match_ratio = len(matched_preferred) / len(preferred_skills_lower) if preferred_skills_lower else 0
    
    # Calculate overall score (required skills are weighted more)
    match_score = (required_match_ratio * 0.7) + (preferred_match_ratio * 0.3)
    match_score = round(match_score * 100, 2)  # Convert to percentage
    
    # Missing required skills
    missing_required = [skill for skill in required_skills_lower if not any(candidate_skill in skill or skill in candidate_skill for candidate_skill in candidate_skills_lower)]
    
    return {
        "match_score": match_score,
        "matched_required": len(matched_required),
        "total_required": len(required_skills_lower),
        "matched_preferred": len(matched_preferred),
        "total_preferred": len(preferred_skills_lower),
        "missing_required_skills": [skill for skill in job_skills.get('required_skills', []) if skill.lower() in missing_required]
    }

def match_candidate_to_jobs(candidate_profile, job_listings, top_n=3):
    """Match a candidate against multiple job listings and return the top matches"""
    matches = []
    
    candidate_skills = candidate_profile.get('technical_skills', [])
    
    for job in job_listings:
        # Calculate skill match score
        match_result = match_skills(candidate_skills, job)
        
        # Add to matches list
        matches.append({
            'job_title': job.get('title', 'Unknown'),
            'company': job.get('company', 'Unknown'),
            'match_score': match_result.get('match_score', 0),
            'matched_required': match_result.get('matched_required', 0),
            'total_required': match_result.get('total_required', 0),
            'matched_preferred': match_result.get('matched_preferred', 0),
            'total_preferred': match_result.get('total_preferred', 0),
            'missing_required_skills': match_result.get('missing_required_skills', []),
            'job_data': job
        })
    
    # Sort by match score (descending)
    matches.sort(key=lambda x: x['match_score'], reverse=True)
    
    # Return top N matches
    return matches[:top_n]

# ==============================================================================
# Recommendation Generation
# ==============================================================================

def generate_recommendations(candidate_data, job_data, match_results):
    """Generate personalized recommendations for the candidate"""
    
    # Prepare prompt with candidate and job data
    system_prompt = """
    You are a career coach and job matching specialist. Based on the candidate's profile and the job description, 
    provide personalized recommendations. Focus on:
    1. How well the candidate matches the job requirements
    2. Key strengths to highlight in their application
    3. Skills they should develop to better qualify for this role
    4. Suggestions for resume improvements
    5. Cover letter talking points specific to this job
    
    Format your response as JSON with the following structure:
    {
        "match_summary": "Brief assessment of how well the candidate fits the role",
        "strengths_to_highlight": ["list", "of", "strengths"],
        "skills_to_develop": ["list", "of", "skills"],
        "resume_tips": ["list", "of", "tips"],
        "cover_letter_points": ["list", "of", "talking points"]
    }
    """
    
    # Get candidate name
    candidate_name = candidate_data.get('personal_info', {}).get('name', 'Candidate')
    
    prompt = f"""
    # Candidate Information
    - Name: {candidate_name}
    - Skills: {candidate_data.get('technical_skills', [])}
    - Soft Skills: {candidate_data.get('soft_skills', [])}
    - Experience Level: {candidate_data.get('experience_level', 'Unknown')}
    - GitHub Activity: {candidate_data.get('github_activity', {})}
    
    # Job Information
    - Title: {job_data.get('title', 'Unknown')}
    - Company: {job_data.get('company', 'Unknown')}
    - Required Skills: {job_data.get('required_skills', [])}
    - Preferred Skills: {job_data.get('preferred_skills', [])}
    - Experience Level: {job_data.get('experience_level', 'Unknown')}
    - Description: {job_data.get('description', '')}
    
    # Match Results
    - Overall Match Score: {match_results.get('match_score', 0)}%
    - Matched Required Skills: {match_results.get('matched_required', 0)}/{match_results.get('total_required', 0)}
    - Matched Preferred Skills: {match_results.get('matched_preferred', 0)}/{match_results.get('total_preferred', 0)}
    - Missing Required Skills: {match_results.get('missing_required_skills', [])}
    """
    
    try:
        api_key = set_api_key()
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        return {}

def enhance_resume_bullets(current_bullets, job_data):
    """Generate enhanced resume bullet points tailored to the job"""
    system_prompt = """
    You are a professional resume writer with expertise in tailoring resumes for specific job opportunities.
    Your task is to enhance the provided resume bullet points to better align with the target job description.
    
    For each bullet point:
    1. Improve the wording to be more impactful and achievement-oriented
    2. Add quantifiable metrics where possible
    3. Incorporate relevant keywords from the job description
    4. Emphasize skills and achievements that match the job requirements
    
    Format your response as a JSON list of improved bullet points, with no additional text or explanation.
    Each item in the list should be an object with:
    - "original": the original bullet point
    - "enhanced": the improved version
    - "explanation": a brief explanation of what was improved and why
    """
    
    user_prompt = f"""
    Job Description:
    - Title: {job_data.get('title', 'Unknown')}
    - Company: {job_data.get('company', 'Unknown')}
    - Required Skills: {job_data.get('required_skills', [])}
    - Preferred Skills: {job_data.get('preferred_skills', [])}
    - Description: {job_data.get('description', '')}
    
    Current Resume Bullet Points:
    {current_bullets}
    """
    
    try:
        api_key = set_api_key()
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error enhancing resume: {e}")
        return {}

def generate_cover_letter(candidate_info, job_info, match_info):
    """Generate a personalized cover letter"""
    system_prompt = """
    You are a professional resume writer specializing in creating personalized cover letters. 
    Generate a compelling, professional cover letter that highlights how the candidate's skills and experience match the job requirements.
    Focus on bridging any gaps between the candidate's profile and the job requirements.
    Keep the tone professional but personable, emphasizing the candidate's enthusiasm for the role.
    
    The cover letter should follow this structure:
    1. Introduction (who they are, the position they're applying for, how they found it)
    2. Why they're a good fit (matching skills, relevant experience)
    3. Why they're interested in the company/position
    4. Addressing any obvious skill gaps proactively
    5. Call to action (request for interview)
    6. Professional closing
    
    Return only the cover letter text, without any additional explanations or notes.
    """
    
    # Compile information for the prompt
    candidate_name = candidate_info.get("personal_info", {}).get("name", "[Candidate Name]")
    job_title = job_info.get("title", "[Job Title]")
    company = job_info.get("company", "[Company Name]")
    
    # Extract candidate skills and experience
    technical_skills = candidate_info.get("technical_skills", [])
    soft_skills = candidate_info.get("soft_skills", [])
    experience = candidate_info.get("experience", [])
    experience_summary = "\n".join([f"- {exp.get('title')} at {exp.get('company')}" for exp in experience[:2]]) if experience else "No experience provided"
    
    # Extract job requirements
    required_skills = job_info.get("required_skills", [])
    preferred_skills = job_info.get("preferred_skills", [])
    job_description = job_info.get("description", "")
    
    # Calculate matching and missing skills
    matching_skills = [skill for skill in technical_skills 
                      if any(skill.lower() in req.lower() or req.lower() in skill.lower() for req in required_skills + preferred_skills)]
    missing_skills = match_info.get("missing_required_skills", [])
    
    user_prompt = f"""
    Candidate Information:
    - Name: {candidate_name}
    - Technical Skills: {technical_skills}
    - Soft Skills: {soft_skills}
    - Recent Experience: 
    {experience_summary}
    
    Job Information:
    - Job Title: {job_title}
    - Company: {company}
    - Required Skills: {required_skills}
    - Preferred Skills: {preferred_skills}
    - Job Description: {job_description}
    
    Matching Information:
    - Match Score: {match_info.get('match_score', 0)}%
    - Matching Skills: {matching_skills}
    - Missing Required Skills: {missing_skills}
    """
    
    try:
        api_key = set_api_key()
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating cover letter: {e}")
        return ""

def generate_candidate_report(candidate_profile, top_job_matches):
    """Generate a comprehensive report for the candidate"""
    system_prompt = """
    You are a career advisor creating a comprehensive job search report. 
    Based on the candidate's profile and job matches, create a professional and personalized report.
    
    The report should include:
    1. Executive Summary: Brief overview of the candidate's profile and top job matches
    2. Candidate Profile Analysis: Assessment of strengths and areas for improvement
    3. Top Job Recommendations: For each job match, explain why it's a good fit and what skills to highlight
    4. Skill Development Plan: Prioritized list of skills to develop based on gaps in top job matches
    5. Next Steps: Actionable recommendations for improving job application success
    
    Format your response as markdown with appropriate headings and bullet points.
    Make it professional but conversational and encouraging.
    """
    
    # Extract candidate info
    candidate_name = candidate_profile.get('personal_info', {}).get('name', 'Candidate')
    technical_skills = candidate_profile.get('technical_skills', [])
    experience_level = candidate_profile.get('experience_level', 'Unknown')
    
    # Format job matches
    job_matches_text = ""
    for i, match in enumerate(top_job_matches, 1):
        job_matches_text += f"Match {i}: {match['job_title']} at {match['company']}\n"
        job_matches_text += f"- Match Score: {match['match_score']}%\n"
        job_matches_text += f"- Required Skills: {match['matched_required']}/{match['total_required']}\n"
        job_matches_text += f"- Missing Skills: {', '.join(match['missing_required_skills'])}\n\n"
    
    user_prompt = f"""
    Candidate Profile:
    - Name: {candidate_name}
    - Experience Level: {experience_level}
    - Technical Skills: {technical_skills}
    - Soft Skills: {candidate_profile.get('soft_skills', [])}
    
    Top Job Matches:
    {job_matches_text}
    """
    
    try:
        api_key = set_api_key()
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating candidate report: {e}")
        return ""

# ==============================================================================
# Main functionality
# ==============================================================================

def print_separator():
    """Print a separator line for better console readability"""
    print("\n" + "="*80 + "\n")

def run_eda(sample_resumes, sample_github_profiles, sample_job_listings):
    """Run exploratory data analysis on the sample data"""
    print("Running Exploratory Data Analysis...")
    
    # Convert to DataFrames for easier analysis
    resume_df = pd.DataFrame(sample_resumes)
    
    # GitHub profiles
    github_df = pd.DataFrame([
        {
            "username": profile["username"],
            "public_repos": profile["public_repos"],
            "followers": profile["followers"],
            "top_languages": ", ".join(profile["top_languages"]),
            "commit_frequency": profile["commit_frequency"],
            "contribution_streak": profile["contribution_streak"],
            "frameworks_detected": ", ".join(profile["frameworks_detected"]),
            "top_repo_stars": max([repo["stars"] for repo in profile["repositories"]]) if profile["repositories"] else 0
        } for profile in sample_github_profiles
    ])
    
    # Job listings
    job_df = pd.DataFrame(sample_job_listings)
    
    # Basic statistics
    print("Resume Profiles:")
    print(resume_df[['name', 'experience_level', 'years_experience']].head())
    print("\nMost common technical skills:")
    all_tech_skills = [skill for skills_list in resume_df['technical_skills'].tolist() for skill in skills_list]
    print(pd.Series(all_tech_skills).value_counts().head(10))
    
    print("\nGitHub Profiles:")
    print(github_df[['username', 'public_repos', 'followers', 'top_repo_stars']].head())
    
    print("\nJob Listings:")
    print(job_df[['title', 'company', 'experience_level']].head())
    
    print("\nTop required skills in job listings:")
    all_required_skills = [skill for skills_list in job_df['required_skills'].tolist() for skill in skills_list]
    print(pd.Series(all_required_skills).value_counts().head(10))
    
    # If matplotlib is available, plot some visualizations
    try:
        plt.figure(figsize=(10, 6))
        
        # Experience level distribution
        plt.subplot(1, 2, 1)
        exp_counts = resume_df['experience_level'].value_counts()
        exp_counts.plot(kind='bar', title='Experience Levels', rot=45)
        
        # Years of experience distribution
        plt.subplot(1, 2, 2)
        resume_df['years_experience'].plot(kind='hist', bins=10, title='Years of Experience')
        
        plt.tight_layout()
        plt.savefig('eda_results.png')
        print("\nEDA visualizations saved as 'eda_results.png'.")
    except Exception as e:
        print(f"\nSkipping visualizations: {e}")
    
    print("\nEDA completed.")

def process_candidate(use_mock_data=True):
    """Process a candidate's resume and GitHub profile"""
    print("Processing candidate profile...")
    
    # In a real system, we would extract text from a PDF file
    # For demonstration, we'll use a mock resume text
    if use_mock_data:
        # Mock resume text
        mock_resume_text = """
        ALEX JOHNSON
        San Francisco, CA | alex.johnson@example.com | (555) 123-4567

        SUMMARY
        Experienced software developer with 4 years of expertise in Python backend development, 
        JavaScript frontend frameworks, and cloud technologies. Passionate about creating scalable 
        applications and optimizing user experiences.

        TECHNICAL SKILLS
        - Languages: Python, JavaScript, HTML, CSS, SQL
        - Frameworks: Django, React, Flask, Express.js
        - Databases: PostgreSQL, MongoDB
        - Tools: Docker, Git, AWS, CI/CD pipelines
        - Other: RESTful APIs, Microservices, Agile methodology

        PROFESSIONAL EXPERIENCE

        SOFTWARE ENGINEER
        TechCorp Inc., San Francisco, CA
        June 2021 - Present
        - Developed and maintained RESTful APIs using Django and Django REST Framework
        - Implemented CI/CD pipelines that reduced deployment time by 40%
        - Optimized database queries resulting in 30% faster page load times
        - Collaborated with frontend team to integrate React components with backend APIs
        - Mentored junior developers on best practices and architectural patterns

        JUNIOR DEVELOPER
        StartApp Labs, San Francisco, CA
        July 2019 - May 2021
        - Built responsive web applications using React and Node.js
        - Developed and maintained API endpoints with Flask and MongoDB
        - Participated in code reviews, sprint planning, and daily stand-ups
        - Implemented automated testing using Jest and Pytest

        EDUCATION
        Bachelor of Science in Computer Science
        University of California, Berkeley
        Graduated: May 2019
        GPA: 3.8/4.0

        PROJECTS
        E-commerce Platform (2022)
        - Built a full-stack e-commerce platform with Django, React, and PostgreSQL
        - Implemented user authentication, product catalog, and payment processing
        - Deployed on AWS using Docker and Kubernetes

        Weather Dashboard (2020)
        - Created a weather dashboard with React that consumes OpenWeatherMap API
        - Implemented geolocation, search functionality, and 5-day forecast display
        - Used Chart.js for data visualization and responsive design for mobile devices
        """
        
        # Parse the mock resume with LLM
        print("Parsing resume...")
        parsed_resume = parse_resume_with_llm(mock_resume_text)
        print(f"Resume parsed successfully. Found {len(parsed_resume.get('skills', {}).get('technical_skills', []))} technical skills.")
        
        # Mock GitHub profile data
        print("Analyzing GitHub profile...")
        mock_github_data = {
            "user_info": {
                "name": "Alex Johnson",
                "username": "alexjcoder",
                "profile_url": "https://github.com/alexjcoder",
                "public_repos": 12,
                "followers": 45
            },
            "top_languages": ["Python", "JavaScript", "HTML", "CSS", "TypeScript"],
            "repositories": [
                {
                    "name": "django-ecommerce",
                    "description": "A full-featured e-commerce platform built with Django and React",
                    "url": "https://github.com/alexjcoder/django-ecommerce",
                    "stars": 28,
                    "forks": 5,
                    "languages": ["Python", "JavaScript", "HTML"]
                },
                {
                    "name": "react-dashboard",
                    "description": "A responsive admin dashboard built with React and Chart.js",
                    "url": "https://github.com/alexjcoder/react-dashboard",
                    "stars": 17,
                    "forks": 3,
                    "languages": ["JavaScript", "CSS"]
                },
                {
                    "name": "weather-app",
                    "description": "A weather dashboard that uses OpenWeatherMap API and React",
                    "url": "https://github.com/alexjcoder/weather-app",
                    "stars": 12,
                    "forks": 2,
                    "languages": ["JavaScript", "HTML", "CSS"]
                }
            ]
        }
        
        github_data = mock_github_data
    else:
        # In a real system, this would extract text from a file and analyze a real GitHub profile
        pdf_path = input("Enter the path to your resume PDF: ")
        resume_text = extract_text_from_pdf(pdf_path)
        parsed_resume = parse_resume_with_llm(resume_text)
        
        github_username = input("Enter your GitHub username: ")
        github_data = analyze_github_profile(github_username)
    
    # Extract skills from the parsed resume
    resume_technical_skills = parsed_resume.get('skills', {}).get('technical_skills', [])
    
    # Combine skills from GitHub and resume
    combined_skills = extract_github_skills(github_data, resume_technical_skills)
    print(f"Combined {len(resume_technical_skills)} resume skills with GitHub data. Total unique skills: {len(combined_skills)}")
    
    # Analyze the top repository from GitHub data
    if github_data.get("repositories"):
        top_repo = github_data["repositories"][0]
        print(f"Analyzing top GitHub repository: {top_repo['name']}...")
        repo_analysis = analyze_github_repo_with_llm(top_repo)
        print(f"Repository analysis complete. Detected {len(repo_analysis.get('primary_skills', []))} primary skills.")
    else:
        repo_analysis = None
    
    # Create a comprehensive candidate profile
    comprehensive_profile = create_comprehensive_profile(parsed_resume, github_data, repo_analysis)
    print(f"Created comprehensive candidate profile with {len(comprehensive_profile.get('technical_skills', []))} technical skills.")
    
    # Save the comprehensive profile
    if not os.path.exists('results'):
        os.makedirs('results')
        
    with open('results/comprehensive_profile.json', 'w') as f:
        json.dump(comprehensive_profile, f, indent=2)
    
    print(f"Comprehensive profile saved to 'results/comprehensive_profile.json'.")
    
    return comprehensive_profile

def match_jobs(candidate_profile, job_listings):
    """Match a candidate with job listings and generate recommendations"""
    print("Matching candidate with job listings...")
    
    top_job_matches = match_candidate_to_jobs(candidate_profile, job_listings, top_n=3)
    print(f"Found {len(top_job_matches)} job matches.")
    
    # Print match results
    candidate_name = candidate_profile.get('personal_info', {}).get('name', 'Candidate')
    print(f"\nTop job matches for {candidate_name}:")
    for i, match in enumerate(top_job_matches, 1):
        print(f"{i}. {match['job_title']} at {match['company']}")
        print(f"   Match Score: {match['match_score']}%")
        print(f"   Required Skills: {match['matched_required']}/{match['total_required']}")
        print(f"   Missing Skills: {', '.join(match['missing_required_skills'])}")
    
    return top_job_matches

def generate_job_application_materials(candidate_profile, top_job_match):
    """Generate job application materials for the top job match"""
    print_separator()
    print(f"Generating application materials for {top_job_match['job_title']} at {top_job_match['company']}...")
    
    # Get detailed recommendations
    job_data = top_job_match['job_data']
    match_results = {
        "match_score": top_job_match['match_score'],
        "matched_required": top_job_match['matched_required'],
        "total_required": top_job_match['total_required'],
        "matched_preferred": top_job_match['matched_preferred'],
        "total_preferred": top_job_match['total_preferred'],
        "missing_required_skills": top_job_match['missing_required_skills']
    }
    
    # Generate recommendations
    print("Generating personalized recommendations...")
    recommendations = generate_recommendations(candidate_profile, job_data, match_results)
    
    # Sample resume bullet points
    sample_bullets = [
        "Developed and maintained web applications using Python and Django",
        "Collaborated with frontend team on API design",
        "Fixed bugs and implemented new features",
        "Participated in code reviews and team meetings"
    ]
    
    # Generate enhanced resume bullet points
    print("Generating enhanced resume bullet points...")
    enhanced_bullets = enhance_resume_bullets(sample_bullets, job_data)
    
    # Generate cover letter
    print("Generating personalized cover letter...")
    cover_letter = generate_cover_letter(candidate_profile, job_data, match_results)
    
    # Save results
    job_title_slug = job_data['title'].lower().replace(' ', '-')
    company_slug = job_data['company'].lower().replace(' ', '-').replace('.', '')
    output_prefix = f"results/{company_slug}_{job_title_slug}"
    
    with open(f"{output_prefix}_recommendations.json", 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    with open(f"{output_prefix}_enhanced_bullets.json", 'w') as f:
        json.dump(enhanced_bullets, f, indent=2)
    
    with open(f"{output_prefix}_cover_letter.txt", 'w') as f:
        f.write(cover_letter)
    
    print(f"Application materials saved to '{output_prefix}_*' files.")
    
    return recommendations, enhanced_bullets, cover_letter

def create_candidate_success_report(candidate_profile, top_job_matches):
    """Create a comprehensive candidate success report"""
    print_separator()
    print("Generating comprehensive candidate success report...")
    
    report = generate_candidate_report(candidate_profile, top_job_matches)
    
    with open("results/candidate_success_report.md", 'w') as f:
        f.write(report)
    
    print("Candidate success report saved to 'results/candidate_success_report.md'.")
    
    return report

def main():
    """Main function to run the SkillMatch AI system"""
    print_separator()
    print("SKILLMATCH AI - Personalized Job Recommendation System")
    print_separator()
    
    # Step 1: Load or generate sample data
    print("Step 1: Loading sample data...")
    sample_resumes, sample_github_profiles, sample_job_listings = load_sample_data()
    
    # Step 2: Run exploratory data analysis (optional)
    run_eda_option = input("Run exploratory data analysis? (y/n): ").lower()
    if run_eda_option == 'y':
        run_eda(sample_resumes, sample_github_profiles, sample_job_listings)
    
    print_separator()
    
    # Step 3: Process candidate profile
    print("Step 3: Processing candidate profile...")
    use_mock_data = input("Use mock data for demonstration? (y/n): ").lower() == 'y'
    candidate_profile = process_candidate(use_mock_data)
    
    print_separator()
    
    # Step 4: Match with job listings
    print("Step 4: Matching with job listings...")
    top_job_matches = match_jobs(candidate_profile, sample_job_listings)
    
    # Step 5: Generate materials for top job
    if top_job_matches:
        recommendations, enhanced_bullets, cover_letter = generate_job_application_materials(
            candidate_profile, top_job_matches[0])
    
    # Step 6: Create success report
    report = create_candidate_success_report(candidate_profile, top_job_matches)
    
    print_separator()
    print("SkillMatch AI process complete!")
    print("All results are saved in the 'results' directory.")
    print_separator()

if __name__ == "__main__":
    main()