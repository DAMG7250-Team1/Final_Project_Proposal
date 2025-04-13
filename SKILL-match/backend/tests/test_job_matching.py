import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import argparse

# Add parent directory to path to allow importing from sibling directories
sys.path.append(str(Path(__file__).parent.parent))

# Import JobMatcher
from jobs.job_matching import JobMatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('job_matching_test.log')
    ]
)
logger = logging.getLogger(__name__)

def save_results_to_file(results, filename="job_match_results.json"):
    """Save match results to a file with pretty formatting"""
    try:
        # Create a directory for test results if it doesn't exist
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        
        # Create a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{timestamp}_{filename}"
        
        # Write results to file with pretty formatting
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error saving results to file: {str(e)}")
        return None

def display_match_summary(matches):
    """Display a summary of the matches"""
    if not matches:
        logger.info("No matches found")
        return
    
    logger.info(f"Found {len(matches)} matches")
    
    # Print summary of matches by category
    categories = {
        "high": [],
        "medium": [],
        "low": [],
        "very_low": []
    }
    
    for match in matches:
        category = match.get("similarity_category", "very_low")
        categories[category].append(match)
    
    # Print category summaries
    for category, matches_in_category in categories.items():
        if matches_in_category:
            logger.info(f"{category.upper()} matches: {len(matches_in_category)}")
    
    # Print top 3 matches
    logger.info("\nTop matches:")
    for i, match in enumerate(matches[:3]):
        logger.info(f"{i+1}. {match['job_title']} at {match['company']} - " +
                   f"Score: {match['similarity_score']:.4f} ({match['similarity_category'].upper()})")
        logger.info(f"   Matching skills: {len(match['matching_skills'])} skills, " +
                   f"overlap: {match['skill_overlap_percent']:.1f}%")
        if match['matching_skills']:
            logger.info(f"   Skills: {', '.join(match['matching_skills'][:5])}" +
                       ("..." if len(match['matching_skills']) > 5 else ""))
        logger.info("")

def test_profile_matching():
    """Test job matching with a specific profile"""
    try:
        logger.info("Starting job matching test with specific profile")
        
        # Initialize JobMatcher
        matcher = JobMatcher()
        logger.info("JobMatcher initialized")
        
        # Set specific S3 URLs for testing
        resume_url = "s3://skillmatchai/resume/markdown/20250411_214055_Dhrumil Y Patel ML engineer.md"
        github_url = "s3://skillmatchai/github/markdown/20250411_214052_dhrumil10_github_profile.md"
        
        # Process user profile
        logger.info(f"Processing user profile from resume: {resume_url}")
        logger.info(f"And GitHub profile: {github_url}")
        profile_data = matcher.get_user_profile_embedding(resume_url, github_url)
        
        # Log extracted skills
        skills = profile_data.get("all_skills", [])
        logger.info(f"Extracted {len(skills)} skills from user profile")
        if skills:
            logger.info(f"Skills: {', '.join(skills)}")
        else:
            logger.warning("No skills were extracted from the profile!")
        
        # Examine the source data for debugging
        for source in profile_data.get("sources", []):
            source_type = source.get("type", "unknown")
            source_skills = source.get("extracted_skills", [])
            logger.info(f"Source type: {source_type}, extracted {len(source_skills)} skills")
            if source_skills:
                logger.info(f"Source skills: {', '.join(source_skills)}")
            else:
                logger.warning(f"No skills extracted from {source_type} source")
                
            # Check if we had errors processing this source
            if "error" in source:
                logger.error(f"Error in {source_type} processing: {source['error']}")
        
        # Match profile with jobs
        logger.info("Matching profile with jobs")
        match_results = matcher.match_profile_with_jobs(profile_data, top_k=10)
        
        # Check results
        if match_results["status"] == "success":
            logger.info(f"Successfully matched profile with {match_results['total_matches']} jobs")
            
            # Display match summary
            display_match_summary(match_results.get("matches", []))
            
            # Save results to file
            output_file = save_results_to_file(match_results)
            logger.info(f"Complete results saved to {output_file}")
            
            # Save user profile data
            user_profile_file = save_results_to_file(profile_data, "user_profile_data.json")
            logger.info(f"User profile data saved to {user_profile_file}")
            
            # Examine job skill data for debugging
            if match_results["matches"]:
                for i, match in enumerate(match_results["matches"][:3]):
                    job_skills = match.get("extracted_skills", [])
                    user_skills = set(profile_data.get("all_skills", []))
                    logger.info(f"Job #{i+1}: {match['job_title']} has {len(job_skills)} skills")
                    if job_skills:
                        logger.info(f"Job skills: {', '.join(job_skills[:20])}" + 
                                   ("..." if len(job_skills) > 20 else ""))
                    
                    # Calculate intersection manually for verification
                    matching = set(job_skills).intersection(user_skills)
                    logger.info(f"Manual skill intersection: {len(matching)} skills")
                    if matching:
                        logger.info(f"Matching skills: {', '.join(matching)}")
            else:
                logger.warning("No job matches were returned")
            
            # Test matching with just skills (without full profile)
            logger.info("\nTesting skill-based matching with the same skills")
            skill_match_results = matcher.match_skills_with_jobs(skills, top_k=10)
            
            if skill_match_results["status"] == "success":
                logger.info(f"Successfully matched skills with {skill_match_results['total_matches']} jobs")
                display_match_summary(skill_match_results.get("matches", []))
                skill_output_file = save_results_to_file(skill_match_results, "skill_match_results.json")
                logger.info(f"Skill match results saved to {skill_output_file}")
            else:
                logger.error(f"Skill matching failed: {skill_match_results.get('error', 'Unknown error')}")
        else:
            logger.error(f"Matching failed: {match_results.get('error', 'Unknown error')}")
        
        return match_results
    
    except Exception as e:
        logger.error(f"Error in test_profile_matching: {str(e)}", exc_info=True)
        return {"status": "error", "error": str(e)}

def test_specific_skills_matching(skills=None):
    """Test job matching with specific skills"""
    if skills is None:
        skills = [
            "Python", "JavaScript", "React", "Node.js", "TypeScript",
            "AWS", "Docker", "MongoDB", "SQL", "Git"
        ]
    
    try:
        logger.info(f"Starting job matching test with {len(skills)} specific skills")
        logger.info(f"Skills: {', '.join(skills)}")
        
        # Initialize JobMatcher
        matcher = JobMatcher()
        
        # Match skills with jobs
        match_results = matcher.match_skills_with_jobs(skills, top_k=10)
        
        # Check results
        if match_results["status"] == "success":
            logger.info(f"Successfully matched skills with {match_results['total_matches']} jobs")
            
            # Display match summary
            display_match_summary(match_results.get("matches", []))
            
            # Save results to file
            output_file = save_results_to_file(match_results, "specific_skills_match_results.json")
            logger.info(f"Results saved to {output_file}")
        else:
            logger.error(f"Matching failed: {match_results.get('error', 'Unknown error')}")
        
        return match_results
    
    except Exception as e:
        logger.error(f"Error in test_specific_skills_matching: {str(e)}", exc_info=True)
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test job matching functionality')
    parser.add_argument('--profile', action='store_true', help='Test job matching with a specific profile')
    parser.add_argument('--skills', action='store_true', help='Test job matching with specific skills')
    parser.add_argument('--skill-list', nargs='+', help='List of specific skills to test')
    
    args = parser.parse_args()
    
    # If no args provided, run both tests
    if not args.profile and not args.skills:
        args.profile = True
        args.skills = True
    
    if args.profile:
        logger.info("=== Testing Job Matching with Specific Profile ===")
        profile_results = test_profile_matching()
        logger.info("=== Profile Matching Test Completed ===\n")
    
    if args.skills:
        logger.info("=== Testing Job Matching with Specific Skills ===")
        if args.skill_list:
            skills_results = test_specific_skills_matching(args.skill_list)
        else:
            skills_results = test_specific_skills_matching()
        logger.info("=== Skills Matching Test Completed ===\n") 