import os
import sys
import json
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to the path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from jobs.embeddings import SkillsEmbeddingsProcessor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

def save_matches_to_file(skills, matches, filename="job_matches.json"):
    """Save job matches to a JSON file for later analysis."""
    output = {
        "query_skills": skills,
        "matches": matches
    }
    
    with open(filename, "w") as f:
        json.dump(output, f, indent=4)
    
    logger.info(f"Saved matches to {filename}")

def plot_similarity_distribution(matches, filename="similarity_distribution.png"):
    """Plot the distribution of similarity scores."""
    if not matches or "matches" not in matches:
        logger.error("No matches data available")
        return
    
    similarities = [match.get("similarity", 0) for match in matches.get("matches", [])]
    
    if not similarities:
        logger.error("No similarity scores found in matches")
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=10, alpha=0.7, color='blue')
    plt.title('Distribution of Job Match Similarity Scores')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.axvline(x=np.mean(similarities), color='red', linestyle='dashed', 
                linewidth=1, label=f'Mean: {np.mean(similarities):.3f}')
    plt.legend()
    plt.savefig(filename)
    logger.info(f"Saved similarity distribution plot to {filename}")

def plot_top_skills_correlation(matches, top_n=10, filename="skills_correlation.png"):
    """Plot correlation between query skills and top job skills."""
    if not matches:
        logger.error("No matches data available")
        return
    
    query_skills = matches.get("query_skills", [])
    
    # Count skill frequency in job matches
    skill_count = {}
    
    for match in matches.get("matches", []):
        job_skills = match.get("skills", [])
        for skill in job_skills:
            if skill in skill_count:
                skill_count[skill] += 1
            else:
                skill_count[skill] = 1
    
    # Get top skills by frequency
    top_skills = sorted(skill_count.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    if not top_skills:
        logger.error("No skills found in job matches")
        return
    
    # Prepare data for plotting
    skills = [skill[0] for skill in top_skills]
    counts = [skill[1] for skill in top_skills]
    
    # Highlight skills that match query skills
    colors = ['red' if skill in query_skills else 'blue' for skill in skills]
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(skills, counts, color=colors)
    plt.title(f'Top {top_n} Skills in Job Matches')
    plt.xlabel('Skills')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Skills in Query'),
        Patch(facecolor='blue', label='Other Common Skills')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(filename)
    logger.info(f"Saved top skills correlation plot to {filename}")

def test_job_matching_visualization():
    """Test visualizing job matches based on user skills."""
    try:
        processor = SkillsEmbeddingsProcessor()
        
        # Example skills - replace with actual user skills for testing
        skills = [
            "Python", "JavaScript", "React", "FastAPI", "AWS", 
            "Data Analysis", "Machine Learning", "SQL", "NoSQL", "Docker"
        ]
        
        logger.info(f"Finding job matches for skills: {skills}")
        
        # Get job matches
        matches = processor.find_matching_jobs(skills)
        
        logger.info(f"Found {matches.get('total_matches', 0)} matching jobs")
        
        # Save matches to file for reference
        save_matches_to_file(skills, matches)
        
        # Generate visualizations
        plot_similarity_distribution(matches)
        plot_top_skills_correlation(matches)
        
        return True
    except Exception as e:
        logger.error(f"Error in test_job_matching_visualization: {str(e)}")
        return False

def test_compare_multiple_skill_sets():
    """Compare job matches for different skill sets."""
    try:
        processor = SkillsEmbeddingsProcessor()
        
        # Define different skill sets to compare
        skill_sets = {
            "Frontend Developer": ["JavaScript", "React", "HTML", "CSS", "TypeScript", "Redux", "UI/UX"],
            "Backend Developer": ["Python", "FastAPI", "Django", "SQL", "NoSQL", "AWS", "Docker"],
            "Data Scientist": ["Python", "R", "Machine Learning", "Statistics", "SQL", "Data Visualization"],
            "DevOps Engineer": ["Docker", "Kubernetes", "AWS", "CI/CD", "Terraform", "Linux", "Monitoring"]
        }
        
        # Store average similarity for each skill set
        avg_similarities = {}
        match_counts = {}
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for role, skills in skill_sets.items():
            logger.info(f"Finding job matches for {role} skills: {skills}")
            
            # Get job matches
            matches = processor.find_matching_jobs(skills)
            
            # Save matches to role-specific file
            save_matches_to_file(skills, matches, f"{role}_matches.json".replace(" ", "_"))
            
            # Calculate average similarity
            similarities = [match.get("similarity", 0) for match in matches.get("matches", [])]
            avg_similarity = np.mean(similarities) if similarities else 0
            avg_similarities[role] = avg_similarity
            
            # Store match count
            match_counts[role] = matches.get("total_matches", 0)
            
            logger.info(f"{role}: Found {match_counts[role]} matches with avg similarity of {avg_similarity:.3f}")
        
        # Plot comparison
        roles = list(skill_sets.keys())
        avgs = [avg_similarities[role] for role in roles]
        counts = [match_counts[role] for role in roles]
        
        # Plot average similarity
        ax.bar(roles, avgs, alpha=0.7, label='Avg Similarity')
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Average Similarity')
        ax.set_title('Job Match Comparison by Role')
        ax.set_xticklabels(roles, rotation=45, ha='right')
        
        # Create second y-axis for match counts
        ax2 = ax.twinx()
        ax2.plot(roles, counts, 'ro-', label='Match Count')
        ax2.set_ylabel('Number of Matches')
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig("role_comparison.png")
        logger.info("Saved role comparison plot to role_comparison.png")
        
        return True
    except Exception as e:
        logger.error(f"Error in test_compare_multiple_skill_sets: {str(e)}")
        return False

if __name__ == "__main__":
    print("Select a test to run:")
    print("1. Run job matching visualization test")
    print("2. Run comparison of multiple skill sets")
    print("3. Run all tests")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == "1":
        test_job_matching_visualization()
    elif choice == "2":
        test_compare_multiple_skill_sets()
    elif choice == "3":
        test_job_matching_visualization()
        test_compare_multiple_skill_sets()
    else:
        print("Invalid choice. Exiting.") 