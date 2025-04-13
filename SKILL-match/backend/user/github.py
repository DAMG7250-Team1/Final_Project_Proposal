import os
import requests
import base64
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv
import boto3
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class GitHubProcessor:
    def __init__(self):
        self.github_token = os.getenv('GITHUB_ACCESS_TOKEN')
        if not self.github_token:
            raise ValueError("GitHub access token not found in environment variables")
        
        # AWS Configuration
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_bucket_name = os.getenv('AWS_BUCKET_NAME')
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
        self.github_base_path = 'github/markdown/'
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
            region_name=self.aws_region
        )
        
        # GitHub API headers
        self.headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }

    def extract_username_from_url(self, github_url: str) -> str:
        """Extract username from GitHub URL."""
        # Remove @ symbol if present at the start
        github_url = github_url.lstrip('@')
        # Remove trailing slash if present
        github_url = github_url.rstrip('/')
        # Split URL and get username (it's always after github.com/)
        parts = github_url.split('github.com/')
        if len(parts) != 2:
            raise Exception("Invalid GitHub URL format")
        # Get the username which is the first part after github.com/
        username = parts[1].split('/')[0]
        return username

    def get_user_profile(self, username: str) -> Dict:
        """Fetch user profile information from GitHub API."""
        url = f'https://api.github.com/users/{username}'
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch user profile: {response.text}")
        return response.json()

    def get_user_repositories(self, username: str) -> List[Dict]:
        """Fetch all repositories for a user."""
        repos = []
        page = 1
        while True:
            url = f'https://api.github.com/users/{username}/repos?page={page}&per_page=100'
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                raise Exception(f"Failed to fetch repositories: {response.text}")
            
            page_repos = response.json()
            if not page_repos:
                break
                
            repos.extend(page_repos)
            page += 1
        
        return repos

    def get_repo_readme(self, username: str, repo_name: str) -> Optional[str]:
        """Fetch README content for a repository."""
        url = f'https://api.github.com/repos/{username}/{repo_name}/readme'
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 404:
            return None
        elif response.status_code != 200:
            raise Exception(f"Failed to fetch README: {response.text}")
            
        content = response.json().get('content', '')
        if content:
            return base64.b64decode(content).decode('utf-8')
        return None

    def generate_profile_markdown(self, profile: Dict, repos: List[Dict], readmes: Dict[str, str]) -> str:
        """Generate markdown content from GitHub data."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        markdown = f"""---
generated_at: {timestamp}
github_username: {profile['login']}
profile_url: https://github.com/{profile['login']}
repository_count: {profile['public_repos']}
followers: {profile['followers']}
following: {profile['following']}
---

# GitHub Profile: {profile['name'] or profile['login']}

{profile['bio'] or ''}

## Profile Information
- Location: {profile['location'] or 'Not specified'}
- Public Repositories: {profile['public_repos']}
- Followers: {profile['followers']}
- Following: {profile['following']}
- Profile Created: {profile['created_at']}
- Last Updated: {profile['updated_at']}

## Repositories

"""
        # Sort repositories by stars
        sorted_repos = sorted(repos, key=lambda x: x['stargazers_count'], reverse=True)
        
        for repo in sorted_repos:
            markdown += f"""### [{repo['name']}]({repo['html_url']})
- Description: {repo['description'] or 'No description provided'}
- Primary Language: {repo['language'] or 'Not specified'}
- Stars: {repo['stargazers_count']}
- Forks: {repo['forks_count']}
- Last Updated: {repo['updated_at']}
- Topics: {', '.join(repo.get('topics', [])) or 'None'}

"""
            # Add README content if available
            if repo['name'] in readmes and readmes[repo['name']]:
                markdown += f"""#### README Content
<details>
<summary>Click to expand README</summary>

```markdown
{readmes[repo['name']]}
```

</details>

"""
        
        return markdown

    def upload_to_s3(self, content: str, username: str) -> str:
        """Upload markdown content to S3."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.github_base_path}{timestamp}_{username}_github_profile.md"
            
            self.s3_client.put_object(
                Bucket=self.aws_bucket_name,
                Key=filename,
                Body=content.encode('utf-8'),
                ContentType='text/markdown'
            )
            
            return f"s3://{self.aws_bucket_name}/{filename}"
        except Exception as e:
            logger.error(f"Failed to upload to S3: {str(e)}")
            raise Exception("Failed to upload GitHub profile to S3")

    def process_github_profile(self, github_url: str) -> Dict[str, str]:
        """Process GitHub profile and return results."""
        try:
            # Extract username from URL
            username = self.extract_username_from_url(github_url)
            
            # Fetch user profile
            logger.info(f"Fetching profile for user: {username}")
            profile = self.get_user_profile(username)
            
            # Fetch repositories
            logger.info(f"Fetching repositories for user: {username}")
            repos = self.get_user_repositories(username)
            
            # Fetch READMEs for each repository
            logger.info("Fetching READMEs for repositories")
            readmes = {}
            for repo in repos:
                readme = self.get_repo_readme(username, repo['name'])
                if readme:
                    readmes[repo['name']] = readme
            
            # Generate markdown content
            logger.info("Generating markdown content")
            markdown_content = self.generate_profile_markdown(profile, repos, readmes)
            
            # Upload to S3
            logger.info("Uploading to S3")
            s3_url = self.upload_to_s3(markdown_content, username)
            
            return {
                "username": username,
                "profile_url": github_url,
                "repository_count": len(repos),
                "readme_count": len(readmes),
                "markdown_url": s3_url
            }
            
        except Exception as e:
            logger.error(f"Error processing GitHub profile: {str(e)}")
            raise Exception(f"Error processing GitHub profile: {str(e)}")
