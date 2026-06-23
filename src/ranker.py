import json
import csv
import argparse
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser()
parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
parser.add_argument("--jd", default="data/job_description.txt", help="Path to JD text file")
parser.add_argument("--out", required=True, help="Path to output submission CSV")
args = parser.parse_args()

career_keywords = [
    "retrieval", "ranking", "recommendation",
    "search", "embedding", "embeddings",
    "semantic search", "faiss", "pinecone",
    "weaviate", "milvus", "vector",
    "evaluation", "a/b", "ndcg", "mrr"
]

skill_keywords = [
    "python", "faiss", "pinecone", "weaviate",
    "milvus", "embeddings",
    "sentence transformers",
    "information retrieval",
    "machine learning",
    "mlops",
    "hugging face transformers"
]

service_companies = [
    "tcs", "infosys", "wipro",
    "accenture", "cognizant", "capgemini"
]

product_companies = [
    "swiggy", "zomato", "uber", "meta",
    "google", "amazon", "microsoft",
    "dream11", "razorpay", "flipkart",
    "haptik", "wyna", "glance"
]

results = []
partial_results = []

model = SentenceTransformer("all-MiniLM-L6-v2")
model.max_seq_length = 128 

# Load Job Description
with open(args.jd, "r", encoding="utf-8") as jd_file:

    job_description = jd_file.read().lower()

jd_embedding = model.encode(job_description)


   
with open(args.candidates, "r", encoding="utf-8") as f:
   
    for i, line in enumerate(f):

        candidate = json.loads(line)

        profile = candidate.get("profile", {})

        candidate_text = ""

        candidate_text += profile.get("current_title", "") + " "
        candidate_text += profile.get("headline", "") + " "
        candidate_text += profile.get("summary", "") + " "

        for job in candidate.get("career_history", []):
            candidate_text += job.get("company", "") + " "
            candidate_text += job.get("title", "") + " "
            candidate_text += job.get("description", "") + " "

        for skill in candidate.get("skills", []):
           candidate_text += skill.get("name", "") + " "

        candidate_text = candidate_text[:800] 

        score = 0

        # Career Text
        career_text = ""

        for job in candidate.get("career_history", []):
            career_text += " "
            career_text += job.get("title", "").lower()
            career_text += " "
            career_text += job.get("description", "").lower()

        # Career Score
        career_score = 0

        for keyword in career_keywords:
            if keyword in career_text:
                career_score += 5

        # Skill Score
        skills = [
            s.get("name", "").lower()
            for s in candidate.get("skills", [])
        ]

        skill_score = 0

        for keyword in skill_keywords:
            if keyword in skills:
                skill_score += 3

        # Behaviour Score
        signals = candidate.get("redrob_signals", {})

        behavior_score = 0

        if signals.get("open_to_work_flag"):
            behavior_score += 5

        # Recruiter response rate
        response_rate = signals.get(
           "recruiter_response_rate", 0
        )

        behavior_score += response_rate * 10

        if response_rate < 0.10:
            behavior_score -= 25
        elif response_rate < 0.20:
            behavior_score -= 15
        elif response_rate < 0.40:
            behavior_score -= 5

        behavior_score += (
            signals.get("interview_completion_rate", 0) * 5
        )

        github = signals.get("github_activity_score", 0)

        if github > 50:
            behavior_score += 5
        elif github > 20:
            behavior_score += 3

        # Logistics Score
        logistics_score = 0

        if signals.get("willing_to_relocate"):
            logistics_score += 5

        notice = signals.get("notice_period_days", 180)

        if notice <= 30:
            logistics_score += 5
        elif notice <= 60:
            logistics_score += 3

        # Experience Score
        years = profile.get("years_of_experience", 0)

        experience_score = 0

        if 5 <= years <= 8:
            experience_score += 20
        elif 8 < years <= 10:
            experience_score += 10
        elif 3 <= years < 5:
            experience_score += 5
        elif 10 < years <= 12:
            experience_score += 0
        elif years > 12:
            experience_score -= 30
        else:
            experience_score -= 5

        score += experience_score

        # Product company bonus
        all_companies = " ".join([
           job.get("company", "").lower()
           for job in candidate.get(
           "career_history", []
        )
        ])

        if any(
           comp in all_companies
           for comp in product_companies
        ):
           score += 15

        # AI/ML title bonus
        title = profile.get("current_title", "").lower()

        good_titles = [
            "machine learning engineer", "applied ml engineer", "nlp engineer",
            "ai engineer", "recommendation systems engineer", "ai research engineer",
            "applied scientist", "search engineer", "retrieval engineer",
            "ranking engineer", "relevance engineer", "data scientist"
        ]

        # Penalize non-technical titles (keyword-stuffing trap, see sample_submission.csv)
        bad_titles = [
            "hr manager", "marketing manager", "content writer",
            "business analyst", "graphic designer", "mechanical engineer",
            "accountant", "civil engineer", "sales executive",
            "operations manager", "project manager", "recruiter",
            "account manager"
        ]

        if any(t in title for t in bad_titles):
            score -= 40   # clearly non-technical title, AI skill keywords likely stuffed/irrelevant

        if (
            "data scientist" in title and
            "recommendation" not in career_text and
            "search" not in career_text
        ):
            score -= 5

        research_titles = [
            "research scientist",
            "research fellow",
            "phd researcher",
            "research engineer"
        ]

        important_terms = [
            "recommendation system",
            "search", 
            "retrieval",
            "ranking",
            "relevance",
            "semantic search"
        ]

        matched_terms = 0
        for term in important_terms:
            if term in career_text:
                matched_terms += 1

        score += min(matched_terms * 3, 12)

        if notice > 90:
            logistics_score -= 10
        elif notice > 60:
            logistics_score -= 5

        # Title bonus
        title_bonus = 0

        if any(t in title for t in good_titles):
            title_bonus += 15

        if any(x in title for x in ["recommendation", "search", "retrieval", "ranking"]):
            title_bonus += 10 

        if any(t in title for t in research_titles):
            title_bonus -= 10   

        score += title_bonus

        # Honeypot detection
        suspicious_skills = sum(
            1 for s in candidate.get("skills", [])
            if s.get("proficiency") == "expert" and s.get("duration_months", 0) < 6
        )
        if suspicious_skills >= 3:
            score -= 30

        # Impossible-tenure honeypot: skill used longer than entire career
        years_months = years * 12
        impossible_skill_count = sum(
            1 for s in candidate.get("skills", [])
            if s.get("duration_months", 0) > years_months + 12   # 1-year buffer, rounding కోసం
        )
        if impossible_skill_count >= 1:
            score -= 35

        # Penalize service companies
        all_companies = " ".join([
            job.get("company", "").lower()
            for job in candidate.get(
                "career_history", []
            )
        ])

        if any(
            comp in all_companies
            for comp in service_companies
        ):
            score -= 10

        last_active = signals.get("last_active_date")
        if last_active:
            days_inactive = (datetime(2026, 6, 23) - datetime.strptime(last_active, "%Y-%m-%d")).days
            if days_inactive > 180:
                behavior_score -= 10

        # Final Score
        score += (
            career_score +
            skill_score +
            behavior_score +
            logistics_score
        )

        # Skills text
        skill_text = " ".join([
            s.get("name", "").lower()
            for s in candidate.get("skills", [])
        ])


        # Reasoning
        title = profile.get("current_title", "Unknown")

        response_rate = signals.get(
           "recruiter_response_rate", 0
        )

        # Count only relevant AI skills
        ai_core_skills = 0

        for keyword in skill_keywords:
             if keyword in skill_text:
                ai_core_skills += 1

        reasoning = (
            f"{title} with {years:.1f} yrs; "
            f"{ai_core_skills} AI core skills; "
            f"response rate {response_rate:.2f}."
        )

        partial_results.append({
            "candidate_id": candidate.get("candidate_id", "UNKNOWN"),
            "score_without_semantic": score,
            "candidate_text": candidate_text,
            "title": title,
            "years": years,
            "ai_core_skills": ai_core_skills,
            "response_rate": response_rate,
            "notice_period": notice,        
            "matched_terms": matched_terms,
        })


partial_results.sort(
    key=lambda x: x["score_without_semantic"],
    reverse=True
)

top3000 = partial_results[:3000]

top3000_texts = [item["candidate_text"] for item in top3000]

all_embeddings = model.encode(
    top3000_texts,
    batch_size=64,
    show_progress_bar=True
)


for idx, item in enumerate(top3000):

    candidate_embedding = all_embeddings[idx]

    semantic_score = cosine_similarity(
        [jd_embedding],
        [candidate_embedding]
    )[0][0]

    semantic_score = semantic_score * 70

    final_score = item["score_without_semantic"] + semantic_score

    reasoning = (
        f"{item['title']} with {item['years']:.1f} yrs; "
        f"{item['ai_core_skills']} AI core skills; "
        f"response rate {item['response_rate']:.2f}."
    )

    results.append({
        "candidate_id": item["candidate_id"],
        "score": round(final_score / 250, 3),
        "title": item["title"],
        "years": item["years"],
        "ai_core_skills": item["ai_core_skills"],
        "response_rate": item["response_rate"],
        "notice_period": item["notice_period"],
        "matched_terms": item["matched_terms"],
    })

# Sort
results.sort(
    key=lambda x: (
        -x["score"],
        x["candidate_id"]
    )
)

top100 = results[:100]

# Rank-aware reasoning 
for rank, cand in enumerate(top100, start=1):

    parts = [f"{cand['title']} with {cand['years']:.1f} yrs experience."]

    # JD-specific connection
    if cand["matched_terms"] >= 4:
        parts.append(f"Career history shows extensive hands-on retrieval/ranking/search work across multiple roles, a strong match for the JD's core mandate.")
    elif cand["matched_terms"] >= 2:
        parts.append(f"Career history shows direct retrieval/ranking/search work, matching the JD's core requirement.")
    elif cand["ai_core_skills"] >= 3:
        parts.append(f"Has {cand['ai_core_skills']} core AI/ML skills relevant to the role, though direct retrieval/ranking project experience is less explicit.")
    else:
        parts.append(f"Limited direct evidence of retrieval/ranking experience in career history.")

    # Honest concerns 
    concerns = []
    if cand["response_rate"] < 0.30:
        concerns.append(f"low recruiter response rate ({cand['response_rate']:.2f})")
    if cand["notice_period"] > 60:
        concerns.append(f"{cand['notice_period']}-day notice period")

    if concerns:
        parts.append("Concern: " + ", ".join(concerns) + ".")
    elif rank <= 10:
        parts.append("Strong overall fit with no major red flags.")

    cand["reasoning"] = " ".join(parts)

# Save Score Breakdown CSV

with open(args.out, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow([
       "candidate_id",
       "rank",
       "score",
       "reasoning"
    ])

    for rank, cand in enumerate(top100, start=1):

        writer.writerow([
            cand["candidate_id"],
            rank,
            cand["score"],
            cand["reasoning"]
        ])

print("Final ranking generated successfully.")

