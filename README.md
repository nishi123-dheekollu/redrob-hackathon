\# Redrob Hackathon Submission



\## Setup

pip install -r requirements.txt



\## Running the ranker

python src/ranker.py --candidates data/candidates.jsonl --out outputs/team\_nishitha123.csv



\## Approach

Two-stage pipeline:

1\. Stage 1 (cheap filter): All candidates scored using keyword matches, experience-band

&#x20;  fit, behavioral signals (recruiter response rate, notice period), and honeypot detection

&#x20;  (impossible skill-tenure, non-technical title penalty). Top 3000 candidates selected.

2\. Stage 2 (semantic re-rank): Top 3000 candidates encoded with sentence-transformers

&#x20;  (all-MiniLM-L6-v2) and compared against the JD via cosine similarity.



\## Compute

Runtime: \~3.5 minutes on CPU for the 3000-candidate embedding stage, within the 5-minute budget.



\## AI tools used

Claude (Anthropic) was used for debugging and iterative improvement during development.

