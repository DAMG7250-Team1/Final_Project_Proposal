```mermaid
graph TD
    %% Main Components
    subgraph User Interface
        UI[Streamlit Web App]
    end

    subgraph Backend Services
        API[FastAPI Backend]
        Airflow[Airflow Orchestration]
    end

    subgraph Data Sources
        JobScraper[Job Scraper]
        GitHubAPI[GitHub API]
        ResumeUpload[Resume Upload]
    end

    subgraph Processing Pipeline
        JobParser[Job Parser]
        ResumeParser[Resume Parser]
        GitHubParser[GitHub Parser]
        EmbeddingGenerator[Embedding Generator]
    end

    subgraph Storage
        Pinecone[(Pinecone Vector DB)]
        S3[(S3 Storage)]
        Snowflake[(Snowflake DB)]
    end

    subgraph AI Agents
        JobMatching[Job Matching Agent]
        ResumeScoring[Resume Scoring Agent]
        CoverLetter[Cover Letter Generator]
        ResumeEnhancer[Resume Enhancer]
    end

    %% Connections
    UI --> API
    API --> Airflow

    %% Data Flow
    JobScraper --> JobParser
    GitHubAPI --> GitHubParser
    ResumeUpload --> ResumeParser

    JobParser --> EmbeddingGenerator
    GitHubParser --> EmbeddingGenerator
    ResumeParser --> EmbeddingGenerator

    EmbeddingGenerator --> Pinecone

    %% Agent Connections
    Pinecone --> JobMatching
    JobMatching --> ResumeScoring
    ResumeScoring --> ResumeEnhancer
    ResumeScoring --> CoverLetter

    %% Storage Connections
    JobMatching --> S3
    ResumeEnhancer --> S3
    CoverLetter --> S3
    ResumeScoring --> Snowflake

    %% UI Feedback Loop
    S3 --> API
    API --> UI

    %% Styling
    classDef component fill:#f9f,stroke:#333,stroke-width:2px
    classDef storage fill:#bbf,stroke:#333,stroke-width:2px
    classDef agent fill:#bfb,stroke:#333,stroke-width:2px
    classDef pipeline fill:#fbb,stroke:#333,stroke-width:2px

    class UI,API,Airflow component
    class Pinecone,S3,Snowflake storage
    class JobMatching,ResumeScoring,CoverLetter,ResumeEnhancer agent
    class JobParser,ResumeParser,GitHubParser,EmbeddingGenerator pipeline
```

## System Architecture Overview

The SkillMatch AI system consists of several key components working together to provide an intelligent job matching and resume enhancement platform:

1. **User Interface Layer**
   - Streamlit-based web application for user interaction
   - Handles resume uploads and displays results

2. **Backend Services**
   - FastAPI backend for handling API requests
   - Airflow for orchestrating data pipelines

3. **Data Sources**
   - Job scraper for collecting job listings
   - GitHub API integration
   - Resume upload functionality

4. **Processing Pipeline**
   - Job parsing and structuring
   - Resume parsing using LLM
   - GitHub profile analysis
   - Embedding generation for semantic search

5. **Storage Layer**
   - Pinecone for vector storage and similarity search
   - S3 for storing generated documents
   - Snowflake for structured data storage

6. **AI Agents**
   - Job matching agent for finding relevant positions
   - Resume scoring agent for quality assessment
   - Cover letter generator for personalized applications
   - Resume enhancer for improvement suggestions

The system follows a modular architecture where each component has a specific responsibility, making it scalable and maintainable. The use of vector embeddings and LLM-based agents enables sophisticated matching and content generation capabilities. 