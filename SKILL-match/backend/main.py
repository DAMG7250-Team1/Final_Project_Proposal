from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from user.resume import ResumeProcessor
import sys
from user.github import GitHubProcessor

from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
import uuid
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from jobs.embeddings import JobEmbeddingsProcessor
from user.user_embedding import UserEmbeddingProcessor
app = FastAPI(title="Resume Processing API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processors
resume_processor = ResumeProcessor()
github_processor = GitHubProcessor()
embeddings_processor = JobEmbeddingsProcessor()
user_processor = UserEmbeddingProcessor()

class GitHubProfile(BaseModel):
    url: str

class GitHubResponse(BaseModel):
    status: str
    message: str
    data: Dict[str, Any]

class ResumeResponse(BaseModel):
    status: str
    message: str
    data: Dict[str, Any]

class JobData(BaseModel):
    job_title: str
    company: str
    location: str
    job_type: str
    work_mode: str
    seniority: str
    salary: str
    experience: str
    responsibilities: str
    qualifications: str
    skills: str

class JobMatchRequest(BaseModel):
    resume_skills: List[str]
    github_skills: List[str]

class JobMatchResponse(BaseModel):
    status: str
    total_skills: int
    total_matches: int
    matches: List[Dict[str, Any]]
    error: Optional[str] = None

@app.post("/api/upload-resume", response_model=ResumeResponse)
async def upload_resume(file: UploadFile, github_url: str):
    """
    Upload and process a resume file, then find matching jobs
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Generate a unique user ID if not provided
        user_id = str(uuid.uuid4())
        
        # Read file content
        file_content = await file.read()
        
        # Process resume
        result = resume_processor.process_resume(
            file_content=file_content,
            original_filename=file.filename
        )
        
        # Ensure all required fields are present
        required_fields = ['extracted_text', 's3_url', 'filename', 'markdown_url']
        if not all(key in result for key in required_fields):
            raise HTTPException(
                status_code=500,
                detail="Missing required fields in resume processing result"
            )
        
        # Process GitHub profile if URL is provided
        github_result = None
        if github_url:
            try:
                github_result = github_processor.process_github_profile(github_url)
            except Exception as e:
                print(f"Warning: Failed to process GitHub profile: {str(e)}")
                github_result = {
                    "status": "failed",
                    "error": str(e)
                }
        
        # Use proper combined embedding from UserEmbeddingProcessor
        user_profile = user_processor.process_user_data(
            resume_url=result["markdown_url"],
            github_url=github_result["markdown_url"] if github_result else None
        )
        
        # Get the combined embedding and skills
        user_embedding = user_profile.get("combined_embedding", [])
        all_skills = user_profile.get("all_skills", [])
        
        # Log vector sanity check
        logger.info(f"[Embedding] Norm: {np.linalg.norm(user_embedding):.4f}, Dimensions: {len(user_embedding)}")
        
        # Query Pinecone with user embedding
        job_vectors = embeddings_processor.index.query(
            vector=user_embedding,
            filter={"source": "job"},
            top_k=50,  # Get more matches to filter
            include_metadata=True
        )
        
        # Process matches with skill overlap
        job_matches = []
        seen_jobs = set()  # Track seen jobs to avoid duplicates
        
        for match in job_vectors.matches:
            # Create unique job identifier
            job_key = f"{match.metadata.get('job_title', '')}_{match.metadata.get('company', '')}"
            if job_key in seen_jobs:
                continue
            seen_jobs.add(job_key)
            
            # Get job skills
            job_skills = set(match.metadata.get("extracted_skills", []))
            
            # Calculate skill overlap
            matching_skills = set(all_skills).intersection(job_skills)
            overlap_pct = (len(matching_skills) / min(len(job_skills), len(all_skills)) * 100) if job_skills and all_skills else 0
            
            # Skip if no skill overlap
            if not matching_skills:
                continue
            
            # Adjust score based on skill overlap
            base_score = match.score
            skill_bonus = (overlap_pct / 100) * 0.4  # Add up to 40% bonus for skill overlap
            adjusted_score = min(base_score + skill_bonus, 1.0)
            
            job_matches.append({
                "job_id": match.id,
                "job_title": match.metadata["job_title"],
                "company": match.metadata["company"],
                "location": match.metadata["location"],
                "job_type": match.metadata["job_type"],
                "work_mode": match.metadata["work_mode"],
                "seniority": match.metadata["seniority"],
                "experience": match.metadata["experience"],
                "similarity_score": adjusted_score,
                "skills": list(matching_skills),
                "skill_overlap_percent": overlap_pct
            })
        
        # Sort by adjusted score and skill overlap
        job_matches.sort(key=lambda x: (x["similarity_score"], x["skill_overlap_percent"]), reverse=True)
        
        # Return only top 10 matches
        job_matches = job_matches[:10]
        
        return {
            "status": "success",
            "message": "Resume processed successfully",
            "data": {
                "user_id": user_id,
                "github_url": github_url,
                "resume_url": result["s3_url"],
                "filename": result["filename"],
                "markdown_url": result["markdown_url"],
                "extracted_text_preview": result["extracted_text"][:500] + "..." if result["extracted_text"] else "",
                "embeddings_info": {
                    "status": "success",
                    "total_skills": len(all_skills),
                    "skills": all_skills,
                    "embedding_norm": float(np.linalg.norm(user_embedding)),
                    "embedding_dimensions": len(user_embedding)
                },
                "github_info": github_result,
                "job_matches": job_matches
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-github", response_model=GitHubResponse)
async def process_github(profile: GitHubProfile):
    """
    Process GitHub profile and extract information
    """
    try:
        # Generate a unique user ID if not provided
        user_id = str(uuid.uuid4())
        
        # Process GitHub profile
        result = github_processor.process_github_profile(profile.url)
        
        # Process GitHub embeddings
        try:
            embeddings_result = embeddings_processor.process_github_markdown(
                markdown_url=result["markdown_url"],
                user_id=user_id
            )
            result["embeddings_info"] = embeddings_result
        except Exception as e:
            # Log the error but don't fail the request
            print(f"Warning: Failed to process GitHub embeddings: {str(e)}")
            result["embeddings_info"] = {
                "status": "failed",
                "total_skills": 0,
                "vectors_created": 0,
                "error": str(e)
            }
        
        return {
            "status": "success",
            "message": "GitHub profile processed successfully",
            "data": {
                "user_id": user_id,
                "username": result["username"],
                "profile_url": result["profile_url"],
                "repository_count": str(result["repository_count"]),
                "readme_count": str(result["readme_count"]),
                "markdown_url": result["markdown_url"],
                "embeddings_info": result.get("embeddings_info", {})
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/match-jobs", response_model=JobMatchResponse)
async def match_jobs(request: JobMatchRequest):
    try:
        # Combine and deduplicate skills
        all_skills = list(set(request.resume_skills + request.github_skills))
        
        # Create a text query from skills
        skills_text = ", ".join(all_skills)
        query_text = f"Find jobs requiring these skills: {skills_text}"
        
        # Find matching jobs
        result = embeddings_processor.find_matching_jobs(query_text=query_text)
        
        if result["status"] == "error":
            return JobMatchResponse(
                status="error",
                total_skills=len(all_skills),
                total_matches=0,
                matches=[],
                error=result.get("error", "Unknown error occurred")
            )
        
        # Transform matches to match frontend expectations
        transformed_matches = []
        for match in result["matches"]:
            metadata = match.get("metadata", {})
            transformed_matches.append({
                "job_title": metadata.get("job_title", "Unknown Title"),
                "company": metadata.get("company", "Unknown Company"),
                "location": metadata.get("location", "Unknown Location"),
                "job_type": metadata.get("job_type", "Unknown Type"),
                "work_mode": metadata.get("work_mode", "Unknown Mode"),
                "seniority": metadata.get("seniority", "Unknown Level"),
                "salary": metadata.get("salary", "Not Specified"),
                "experience": metadata.get("experience", "Not Specified"),
                "responsibilities": metadata.get("responsibilities", "Not Specified"),
                "qualifications": metadata.get("qualifications", "Not Specified"),
                "skills": metadata.get("skills", "Not Specified"),
                "similarity_score": match.get("similarity", 0.0),
                "match_category": match.get("category", "unknown")
            })
        
        return JobMatchResponse(
            status="success",
            total_skills=len(all_skills),
            total_matches=result["total_matches"],
            matches=transformed_matches
        )
        
    except Exception as e:
        logger.error(f"Error matching jobs: {str(e)}")
        return JobMatchResponse(
            status="error",
            total_skills=0,
            total_matches=0,
            matches=[],
            error=str(e)
        )

@app.post("/api/batch-match-jobs", response_model=JobMatchResponse)
async def batch_match_jobs(jobs: List[JobData]):
    """
    Process multiple job descriptions and find matching user profiles
    """
    try:
        results = []
        for job in jobs:
            job_dict = job.dict()
            matches = embeddings_processor.get_job_matches(job_dict)
            results.append({
                "job_info": matches["job_info"],
                "matches": matches["matches"]
            })
        
        return {
            "status": "success",
            "message": f"Processed {len(results)} jobs successfully",
            "data": {
                "results": results
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
