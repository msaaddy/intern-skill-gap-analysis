# Intern Skill Gap Analysis

## Project Overview

This project analyzes intern skills and compares them with skills required by industry job descriptions.

The system uses Natural Language Processing (NLP), TF-IDF, K-Means clustering, and cosine similarity to identify skill gaps and recommend suitable training areas.

## Objective

Analyze intern skills and identify gaps compared to industry demands.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- K-Means Clustering
- Cosine Similarity
- Matplotlib

## Dataset

The project uses two datasets:

### Intern Skills

Contains:

- Intern ID
- Intern Name
- Current Skills

### Industry Job Descriptions

Contains:

- Job ID
- Job Title
- Required Skills

## Methodology

### 1. Data Collection

Intern skill data and industry job descriptions are loaded from CSV files.

### 2. Text Preprocessing

Skill and job-description text is converted to lowercase and cleaned.

### 3. TF-IDF

TF-IDF converts skill and job-description text into numerical representations.

### 4. K-Means Clustering

K-Means groups interns according to similar skill profiles.

### 5. Job Matching

Cosine similarity compares each intern's skill profile with industry job descriptions.

### 6. Skill Gap Analysis

The system identifies skills required by the matched job that are missing from the intern's current skill set.

### 7. Training Recommendations

Missing skills are converted into recommended training areas.

## Project Structure

```text
intern-skill-gap/
│
├── data/
│   ├── intern_skills.csv
│   └── job_descriptions.csv
│
├── skill_gap_analysis.py
├── skill_gap_results.csv
├── intern_clusters.png
├── requirements.txt
└── README.md