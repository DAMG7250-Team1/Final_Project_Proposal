# SKILL-match Test Scripts

This directory contains various test scripts for testing the functionality of the SKILL-match application.

## Available Tests

### 1. Job Embeddings Test

The `test_job_embeddings.py` script allows you to test the job embeddings functionality. This includes:

- Processing specific job files from S3
- Processing all job files in S3
- Querying jobs based on skills

#### Prerequisites

Make sure you have the following environment variables set:
- `OPENAI_API_KEY` - Your OpenAI API key
- `PINECONE_API_KEY` - Your Pinecone API key
- `PINECONE_ENVIRONMENT` - Your Pinecone environment
- `AWS_ACCESS_KEY_ID` - Your AWS access key
- `AWS_SECRET_ACCESS_KEY` - Your AWS secret key

#### Running the Job Embeddings Test

```bash
cd backend
python -m tests.test_job_embeddings
```

When prompted, select one of the available options:
1. Process a specific job file
2. Process all job files
3. Query jobs for matching skills
4. Run all tests

### 2. Pinecone Storage Test

The `test_pinecone_storage.py` script allows you to test reading job data from S3 and storing it in Pinecone.

#### Running the Pinecone Storage Test

```bash
cd backend
python -m tests.test_pinecone_storage
```

### 3. Job Matching Visualization Test

The `test_match_visualization.py` script provides visualization tools to analyze the effectiveness of job matching algorithms. This includes:

- Visualizing the distribution of similarity scores
- Analyzing the correlation between user skills and job skills
- Comparing job matches across different skill sets

#### Prerequisites

In addition to the environment variables mentioned above, ensure you have the following Python packages installed:
- `matplotlib`
- `pandas`
- `numpy`

#### Running the Job Matching Visualization Test

```bash
cd backend
python -m tests.test_match_visualization
```

When prompted, select one of the available options:
1. Run job matching visualization test
2. Run comparison of multiple skill sets
3. Run all tests

The script will generate several visualization files:
- `similarity_distribution.png` - Histogram showing the distribution of similarity scores
- `skills_correlation.png` - Bar chart showing top skills in job matches
- `role_comparison.png` - Comparison of job matches across different skill sets
- JSON files containing match data for further analysis

## Adding New Tests

When adding new tests:

1. Create a new Python file in the `tests` directory with a descriptive name
2. Follow the pattern of existing tests
3. Add appropriate logging
4. Update this README with information about your new test 