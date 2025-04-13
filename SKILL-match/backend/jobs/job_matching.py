import os
import sys
import json
import logging
import string
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone

# Add parent directory to path to allow importing from sibling directories
sys.path.append(str(Path(__file__).parent.parent))

# Import from existing modules
from user.user_embedding import UserEmbeddingProcessor
from jobs.embeddings import JobEmbeddingsProcessor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_skill(skill: str) -> str:
    """
    Normalize a skill string while preserving important punctuation.
    - Convert to lowercase
    - Remove extra whitespace
    - Preserve important punctuation like '+', '.', '#'
    """
    # Keep important punctuation
    keep_punct = set('+.#')
    # Remove other punctuation except spaces
    normalized = ''.join(c if c.isalnum() or c in keep_punct or c.isspace() else ' ' for c in skill)
    # Normalize spaces and convert to lowercase
    return ' '.join(normalized.lower().split())

class JobMatcher:
    """
    JobMatcher uses embeddings from user profiles to find matching jobs in Pinecone.
    It leverages existing functionality in UserEmbeddingProcessor and JobEmbeddingsProcessor.
    """
    
    def __init__(self):
        """Initialize the JobMatcher with connections to Pinecone."""
        try:
            # Pinecone configuration
            self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
            self.pinecone_index_name = os.getenv('PINECONE_INDEX_NAME', 'skillmatch')
            
            if not self.pinecone_api_key:
                raise ValueError("Pinecone API key not found in environment variables")
            
            # Initialize Pinecone connection
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            self.index = self.pc.Index(self.pinecone_index_name)
            logger.info(f"Connected to Pinecone index: {self.pinecone_index_name}")
            
            # Set similarity thresholds (vector-based similarity is our main indicator)
            self.similarity_thresholds = {
                "high": 0.45,    # Very strong match
                "medium": 0.35,  # Good match
                "low": 0.25      # Minimal match
            }
            
            # Initialize supporting processors
            self.user_processor = UserEmbeddingProcessor()
            self.job_processor = JobEmbeddingsProcessor()
            
            logger.info("JobMatcher initialized successfully")
            
        except Exception as e:
            
            logger.error(f"Error initializing JobMatcher: {str(e)}")
            raise
    
    def get_similarity_category(self, score: float) -> str:
        """
        Categorize a similarity score into high, medium, or low.
        
        Args:
            score: Similarity score from Pinecone (0.0 to 1.0)
            
        Returns:
            String category: "high", "medium", "low" or "very_low"
        """
        if score >= self.similarity_thresholds["high"]:
            return "high"
        elif score >= self.similarity_thresholds["medium"]:
            return "medium"
        elif score >= self.similarity_thresholds["low"]:
            return "low"
        else:
            return "very_low"
    
    def match_profile_with_jobs(self, profile_data: Dict[str, Any], top_k: int = 10) -> Dict[str, Any]:
        """
        Match a user profile with jobs in Pinecone.
        
        Args:
            profile_data: User profile data containing a combined embedding and all_skills list.
            top_k: Number of top matches to return.
            
        Returns:
            Dictionary with match results.
        """
        try:
            if "combined_embedding" not in profile_data:
                logger.error("Profile data missing combined embedding")
                return {
                    "status": "error",
                    "error": "Profile data missing combined embedding",
                    "matches": []
                }
            
            # Extract and normalize user skills
            user_skills = set(normalize_skill(skill) for skill in profile_data.get("all_skills", []))
            logger.info(f"User skills (normalized): {user_skills}")
            
            # Query Pinecone with the combined embedding
            logger.info(f"Querying Pinecone for top {top_k} job matches")
            
            # Get total number of vectors in Pinecone
            total_vectors = self.index.describe_index_stats()["total_vector_count"]
            logger.info(f"Total vectors in Pinecone: {total_vectors}")
            
            # Query all available jobs using the combined embedding
            results = self.index.query(
                vector=profile_data["combined_embedding"],
                filter={"source": "job"},
                top_k=total_vectors,  # Query all available jobs
                include_metadata=True
            )
            
            # Track seen jobs to avoid duplicates
            seen_jobs = set()
            matches = []
            
            for match in results.matches:
                # Skip if we've seen this exact job before
                job_key = f"{match.metadata.get('job_title', '')}_{match.metadata.get('company', '')}_{match.metadata.get('location', '')}_{match.metadata.get('job_type', '')}"
                if job_key in seen_jobs:
                    continue
                seen_jobs.add(job_key)
                
                # Get and normalize job skills
                job_skills_raw = match.metadata.get("extracted_skills", [])
                normalized_job_skills = set(normalize_skill(s) for s in job_skills_raw)
                
                # Calculate skill overlap using fuzzy matching
                matching_skills = set()
                for user_skill in user_skills:
                    for job_skill in normalized_job_skills:
                        # Check for exact match or substring match
                        if user_skill == job_skill or user_skill in job_skill or job_skill in user_skill:
                            matching_skills.add(job_skill)
                
                # Calculate overlap percentage based on job skills
                # Use the smaller of the two sets as denominator to avoid over 100%
                overlap_pct = (len(matching_skills) / min(len(normalized_job_skills), len(user_skills)) * 100) if normalized_job_skills and user_skills else 0
                
                # Skip matches with no skill overlap
                if not matching_skills:
                    continue
                
                # Adjust similarity score based on skill overlap and experience
                base_score = match.score
                skill_bonus = (overlap_pct / 100) * 0.5  # Reduce skill bonus impact
                experience_bonus = 0.1 if "experience" in match.metadata.get("qualifications", "").lower() else 0
                adjusted_score = base_score + (base_score * (skill_bonus + experience_bonus))
                
                # Ensure score doesn't exceed 1.0
                adjusted_score = min(adjusted_score, 1.0)
                
                match_data = {
                    "job_id": match.id,
                    "similarity_score": adjusted_score,
                    "similarity_category": self.get_similarity_category(adjusted_score),
                    "job_title": match.metadata.get("job_title", ""),
                    "company": match.metadata.get("company", ""),
                    "location": match.metadata.get("location", ""),
                    "job_type": match.metadata.get("job_type", ""),
                    "work_mode": match.metadata.get("work_mode", ""),
                    "seniority": match.metadata.get("seniority", ""),
                    "salary": match.metadata.get("salary", ""),
                    "experience": match.metadata.get("experience", ""),
                    "responsibilities": match.metadata.get("responsibilities", ""),
                    "qualifications": match.metadata.get("qualifications", ""),
                    "skills": match.metadata.get("skills", ""),
                    "extracted_skills": list(normalized_job_skills),
                    "matching_skills": list(matching_skills),
                    "skill_overlap_percent": overlap_pct,
                    "job_context": match.metadata.get("responsibilities", "") + " " + match.metadata.get("qualifications", "")
                }
                matches.append(match_data)
            
            # Sort matches by adjusted similarity score and skill overlap
            matches.sort(key=lambda x: (x["similarity_score"], x["skill_overlap_percent"]), reverse=True)
            
            # Return only top_k matches
            matches = matches[:top_k]
            
            return {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "total_matches": len(matches),
                "user_skills": list(user_skills),
                "user_skills_count": len(user_skills),
                "matches": matches
            }
            
        except Exception as e:
            logger.error(f"Error matching profile with jobs: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "matches": []
            }
    
    def match_skills_with_jobs(self, skills: List[str], top_k: int = 10) -> Dict[str, Any]:
        """
        Match a list of skills with jobs in Pinecone.
        
        Args:
            skills: List of skill strings.
            top_k: Number of top matches to return.
            
        Returns:
            Dictionary with match results.
        """
        try:
            # Normalize input skills
            normalized_input_skills = [normalize_skill(s) for s in skills]
            skills_text = ", ".join(normalized_input_skills)
            embedding = self.job_processor.get_embedding(skills_text)
            
            logger.info(f"Querying Pinecone for top {top_k} job matches based on skills")
            results = self.index.query(
                vector=embedding,
                filter={"source": "job"},
                top_k=top_k,
                include_metadata=True
            )
            
            matches = []
            for match in results.matches:
                if match.score < self.similarity_thresholds["low"]:
                    continue
                
                # Normalize job's extracted skills
                job_skills_raw = match.metadata.get("extracted_skills", [])
                normalized_job_skills = set(normalize_skill(s) for s in job_skills_raw)
                matching_skills = list(set(normalized_input_skills).intersection(normalized_job_skills))
                overlap_pct = (len(matching_skills) / len(normalized_job_skills) * 100) if normalized_job_skills else 0
                
                match_data = {
                    "job_id": match.id,
                    "similarity_score": match.score,
                    "similarity_category": self.get_similarity_category(match.score),
                    "job_title": match.metadata.get("job_title", ""),
                    "company": match.metadata.get("company", ""),
                    "location": match.metadata.get("location", ""),
                    "job_type": match.metadata.get("job_type", ""),
                    "work_mode": match.metadata.get("work_mode", ""),
                    "seniority": match.metadata.get("seniority", ""),
                    "salary": match.metadata.get("salary", ""),
                    "experience": match.metadata.get("experience", ""),
                    "responsibilities": match.metadata.get("responsibilities", ""),
                    "qualifications": match.metadata.get("qualifications", ""),
                    "skills": match.metadata.get("skills", ""),
                    "extracted_skills": list(normalized_job_skills),
                    "matching_skills": matching_skills,
                    "skill_overlap_percent": overlap_pct,
                    "job_context": match.metadata.get("responsibilities", "") + " " + match.metadata.get("qualifications", "")
                }
                matches.append(match_data)
            
            return {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "total_matches": len(matches),
                "input_skills": normalized_input_skills,
                "input_skills_count": len(normalized_input_skills),
                "matches": matches
            }
            
        except Exception as e:
            logger.error(f"Error matching skills with jobs: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "matches": []
            }
    
    def get_user_profile_embedding(self, resume_url: str, github_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a user profile embedding from resume and GitHub URLs.
        This is a convenience method that calls the UserEmbeddingProcessor.
        
        Args:
            resume_url: S3 URL to resume markdown.
            github_url: Optional S3 URL to GitHub profile markdown.
            
        Returns:
            User profile data with embedding.
        """
        try:
            logger.info(f"Processing user profile from resume and GitHub")
            return self.user_processor.process_user_data(resume_url, github_url)
        except Exception as e:
            logger.error(f"Error getting user profile embedding: {str(e)}")
            raise
    
    def save_match_results(self, user_id: str, match_results: Dict[str, Any]) -> str:
        """
        Save job match results to S3.
        
        Args:
            user_id: User ID.
            match_results: Match results dictionary.
            
        Returns:
            S3 URL to saved results.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            key = f"job_matches/{user_id}/{timestamp}_matches.json"
            json_data = json.dumps(match_results, ensure_ascii=False)
            
            self.user_processor.s3_client.put_object(
                Bucket=self.user_processor.aws_bucket_name,
                Key=key,
                Body=json_data.encode('utf-8'),
                ContentType='application/json'
            )
            
            s3_url = f"s3://{self.user_processor.aws_bucket_name}/{key}"
            logger.info(f"Job matches saved to {s3_url}")
            return s3_url
            
        except Exception as e:
            logger.error(f"Error saving match results: {str(e)}")
            raise

# Example usage if run directly:
if __name__ == "__main__":
    try:
        matcher = JobMatcher()
        
        # Example of how to use the JobMatcher
        print("JobMatcher initialized successfully")
        print("Example usage:")
        print("1. Get user profile embedding:")
        print("   profile_data = matcher.get_user_profile_embedding(resume_url, github_url)")
        print("2. Match profile with jobs:")
        print("   match_results = matcher.match_profile_with_jobs(profile_data, top_k=10)")
        print("3. Save match results:")
        print("   match_url = matcher.save_match_results(user_id, match_results)")
        print("\nFor skill-based matching:")
        print("   skill_matches = matcher.match_skills_with_jobs(skills_list, top_k=5)")
        
    except Exception as e:
        logger.error(f"Error in example: {str(e)}")
        sys.exit(1)
