from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import sys
import uvicorn
import uuid
import logging
import numpy as np
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from user.resume import ResumeProcessor
from user.github import GitHubProcessor
from jobs.embeddings import JobEmbeddingsProcessor
from user.user_embedding import UserEmbeddingProcessor

logger = logging.getLogger(__name__)
app = FastAPI(title="Resume Processing API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processors
resume_processor = ResumeProcessor()
github_processor = GitHubProcessor()
embeddings_processor = JobEmbeddingsProcessor()
user_processor = UserEmbeddingProcessor()

# Pydantic models
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
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        user_id = str(uuid.uuid4())
        file_content = await file.read()

        result = resume_processor.process_resume(file_content, file.filename)
        required = ['extracted_text', 's3_url', 'filename', 'markdown_url']
        if not all(k in result for k in required):
            raise HTTPException(status_code=500, detail="Missing fields in resume processing")

        github_result = None
        if github_url:
            try:
                github_result = github_processor.process_github_profile(github_url)
            except Exception as e:
                github_result = {"status": "failed", "error": str(e)}

        user_profile = user_processor.process_user_data(
            resume_url=result["markdown_url"],
            github_url=github_result["markdown_url"] if github_result else None
        )

        user_embedding = user_profile.get("combined_embedding", [])
        all_skills = user_profile.get("all_skills", [])

        job_vectors = embeddings_processor.index.query(
            vector=user_embedding,
            filter={"source": "job"},
            top_k=50,
            include_metadata=True
        )

        seen_jobs = set()
        job_matches = []
        for match in job_vectors.matches:
            job_key = f"{match.metadata.get('job_title', '')}_{match.metadata.get('company', '')}"
            if job_key in seen_jobs:
                continue
            seen_jobs.add(job_key)
            job_skills = set(match.metadata.get("extracted_skills", []))
            matching_skills = set(all_skills).intersection(job_skills)
            if not matching_skills:
                continue
            overlap_pct = (len(matching_skills) / min(len(job_skills), len(all_skills)) * 100) if job_skills and all_skills else 0
            score = min(match.score + ((overlap_pct / 100) * 0.4), 1.0)
            job_matches.append({
                "job_id": match.id,
                "job_title": match.metadata["job_title"],
                "company": match.metadata["company"],
                "location": match.metadata["location"],
                "job_type": match.metadata["job_type"],
                "work_mode": match.metadata["work_mode"],
                "seniority": match.metadata["seniority"],
                "experience": match.metadata["experience"],
                "similarity_score": score,
                "skills": list(matching_skills),
                "skill_overlap_percent": overlap_pct
            })

        job_matches.sort(key=lambda x: (x["similarity_score"], x["skill_overlap_percent"]), reverse=True)
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
                "extracted_text_preview": result["extracted_text"][:500] + "...",
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
    try:
        user_id = str(uuid.uuid4())
        result = github_processor.process_github_profile(profile.url)
        try:
            embeddings_result = embeddings_processor.process_github_markdown(
                markdown_url=result["markdown_url"],
                user_id=user_id
            )
            result["embeddings_info"] = embeddings_result
        except Exception as e:
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
        all_skills = list(set(request.resume_skills + request.github_skills))
        skills_text = ", ".join(all_skills)
        result = embeddings_processor.find_matching_jobs(query_text=skills_text)

        if result["status"] == "error":
            return JobMatchResponse(
                status="error",
                total_skills=len(all_skills),
                total_matches=0,
                matches=[],
                error=result.get("error", "Unknown error occurred")
            )

        transformed_matches = []
        for match in result["matches"]:
            m = match["metadata"]
            transformed_matches.append({
                "job_title": m.get("job_title", "Unknown Title"),
                "company": m.get("company", "Unknown Company"),
                "location": m.get("location", "Unknown Location"),
                "job_type": m.get("job_type", "Unknown Type"),
                "work_mode": m.get("work_mode", "Unknown Mode"),
                "seniority": m.get("seniority", "Unknown Level"),
                "salary": m.get("salary", "Not Specified"),
                "experience": m.get("experience", "Not Specified"),
                "responsibilities": m.get("responsibilities", "Not Specified"),
                "qualifications": m.get("qualifications", "Not Specified"),
                "skills": m.get("skills", "Not Specified"),
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
        return JobMatchResponse(
            status="error",
            total_skills=0,
            total_matches=0,
            matches=[],
            error=str(e)
        )

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)