import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATA
# ============================================================

interns = pd.read_csv("data/intern_skills.csv")
jobs = pd.read_csv("data/job_descriptions.csv")

print("\n===== INTERN DATA =====")
print(interns)

print("\n===== JOB DESCRIPTIONS =====")
print(jobs)


# ============================================================
# 2. CLEAN TEXT
# ============================================================

interns["skills"] = (
    interns["skills"]
    .fillna("")
    .str.lower()
    .str.strip()
)

jobs["description"] = (
    jobs["description"]
    .fillna("")
    .str.lower()
    .str.strip()
)


# ============================================================
# 3. TF-IDF NLP
# ============================================================

all_text = pd.concat(
    [
        interns["skills"],
        jobs["description"]
    ],
    ignore_index=True
)

vectorizer = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = vectorizer.fit_transform(all_text)

print("\n===== TF-IDF =====")
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)


# Separate intern and job vectors

intern_vectors = tfidf_matrix[:len(interns)]

job_vectors = tfidf_matrix[len(interns):]


# ============================================================
# 4. K-MEANS CLUSTERING
# ============================================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

interns["cluster"] = kmeans.fit_predict(
    intern_vectors
)

print("\n===== INTERN CLUSTERS =====")

print(
    interns[
        [
            "intern_id",
            "intern_name",
            "skills",
            "cluster"
        ]
    ]
)


# ============================================================
# 5. MATCH INTERNS WITH INDUSTRY JOBS
# ============================================================

similarity_matrix = cosine_similarity(
    intern_vectors,
    job_vectors
)

best_job_indexes = similarity_matrix.argmax(axis=1)

best_scores = similarity_matrix.max(axis=1)


interns["best_job"] = [
    jobs.iloc[index]["job_title"]
    for index in best_job_indexes
]

interns["match_score"] = best_scores


print("\n===== JOB MATCHING =====")

print(
    interns[
        [
            "intern_name",
            "best_job",
            "match_score"
        ]
    ]
)


# ============================================================
# 6. EXTRACT SKILLS
# ============================================================

def extract_skills(text):

    return set(
        skill.strip().lower()
        for skill in text.split(",")
        if skill.strip()
    )


# ============================================================
# 7. IDENTIFY SKILL GAPS
# ============================================================

skill_gaps = []

for i, intern in interns.iterrows():

    # Intern's current skills
    intern_skills = extract_skills(
        intern["skills"]
    )

    # Best matching job
    job_index = best_job_indexes[i]

    # Skills required by the job
    required_skills = extract_skills(
        jobs.iloc[job_index]["description"]
    )

    # Find missing skills
    missing_skills = (
        required_skills - intern_skills
    )

    skill_gaps.append(
        ", ".join(sorted(missing_skills))
    )


interns["skill_gaps"] = skill_gaps


# ============================================================
# 8. TRAINING RECOMMENDATIONS
# ============================================================

training_map = {

    "python": "Advanced Python",

    "sql": "Advanced SQL",

    "machine learning": "Machine Learning",

    "pandas": "Pandas and Data Analysis",

    "numpy": "NumPy",

    "scikit-learn": "Scikit-learn",

    "tensorflow": "Deep Learning with TensorFlow",

    "statistics": "Statistics for Data Science",

    "data visualization": "Data Visualization",

    "power bi": "Power BI",

    "tableau": "Tableau",

    "excel": "Advanced Excel",

    "git": "Git and GitHub",

    "docker": "Docker",

    "nlp": "Natural Language Processing",

    "transformers": "Transformers and LLMs",

    "javascript": "JavaScript",

    "react": "React",

    "html": "HTML",

    "css": "CSS",

    "rest api": "REST API Development"
}


def recommend_training(gaps):

    recommendations = []

    for skill in gaps.split(","):

        skill = skill.strip().lower()

        if not skill:
            continue

        if skill in training_map:

            recommendations.append(
                training_map[skill]
            )

        else:

            recommendations.append(
                f"Training in {skill.title()}"
            )

    return ", ".join(recommendations)


interns["training_recommendations"] = (
    interns["skill_gaps"].apply(
        recommend_training
    )
)


# ============================================================
# 9. FINAL RESULTS
# ============================================================

print("\n==========================================")
print("FINAL SKILL GAP ANALYSIS")
print("==========================================")

print(
    interns[
        [
            "intern_name",
            "best_job",
            "match_score",
            "cluster",
            "skill_gaps",
            "training_recommendations"
        ]
    ].to_string(index=False)
)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

output_file = "skill_gap_results.csv"

interns.to_csv(
    output_file,
    index=False
)

print("\n==========================================")
print("RESULTS SAVED")
print("==========================================")

print(f"Output file: {output_file}")


# ============================================================
# 11. CREATE CLUSTER VISUALIZATION
# ============================================================

cluster_counts = (
    interns["cluster"]
    .value_counts()
    .sort_index()
)


plt.figure(figsize=(8, 5))

plt.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

plt.xlabel("Cluster")

plt.ylabel("Number of Interns")

plt.title("Intern Skill Clusters")

plt.tight_layout()

plt.savefig(
    "intern_clusters.png"
)

plt.show()


print("\nCluster visualization saved as:")
print("intern_clusters.png")


# ============================================================
# 12. PROJECT COMPLETED
# ============================================================

print("\n==========================================")
print("SKILL GAP ANALYSIS COMPLETED")
print("==========================================")