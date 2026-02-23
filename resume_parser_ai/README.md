# Resume Parser AI (NLTK + FastAPI)

Production-ready AI-powered resume parsing system that:
- Ingests a dataset with `Job Title` and `Resume` columns.
- Preprocesses text using **NLTK**.
- Builds reusable **TF-IDF** features (1-2 grams).
- Clusters resumes with **KMeans** and elbow-based cluster selection.
- Scores resume relevance to a target job title on a **0-10** scale.
- Exposes a **FastAPI** endpoint for real-time scoring.

## Project Structure

```text
resume_parser_ai/
├── data/
│   └── raw_resume_dataset.csv
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── clustering.py
│   ├── scoring.py
│   ├── model_pipeline.py
│   └── utils.py
├── api/
│   └── app.py
├── models/
│   └── saved_model.pkl
├── tests/
│   └── test_pipeline.py
├── Dockerfile
├── requirements.txt
├── README.md
└── main.py
```

## Core Capabilities

1. **Data preprocessing (NLTK)**
   - Lowercasing
   - Punctuation removal
   - Stopword removal
   - Tokenization
   - Lemmatization
   - Number removal
   - Extra whitespace cleanup

2. **Feature engineering**
   - TF-IDF vectorization
   - N-grams `(1,2)`
   - `max_features` control
   - Reusable vectorizer and keyword extraction

3. **Clustering**
   - KMeans
   - Elbow-based automatic cluster selection
   - Cluster assignment for new resumes

4. **Scoring**
   - Cosine similarity between resume and job title vectors
   - 0-10 normalized score
   - Keyword stuffing penalty

5. **Extra features**
   - TF-IDF top keyword extraction
   - Skill frequency analysis
   - Training metrics (including silhouette score)
   - Unit test coverage for train/inference flow

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train Model Locally

```bash
python main.py
```

Optional environment variables:

```bash
export DATA_PATH=data/raw_resume_dataset.csv
export MODEL_PATH=models/saved_model.pkl
export LOG_LEVEL=INFO
```

## Run API Locally

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

## Example API Request

```bash
curl -X POST "http://127.0.0.1:8000/score" \
  -H "Content-Type: application/json" \
  -d '{
    "job_title": "Data Scientist",
    "resume_text": "Python machine learning NLP, model deployment, SQL and statistics experience in production systems"
  }'
```

Example response:

```json
{
  "score": 8.4,
  "cluster": 2,
  "top_keywords": ["python", "machine learning", "nlp"]
}
```

## Docker Deployment

Build and run:

```bash
docker build -t resume-parser-ai .
docker run -p 8000:8000 resume-parser-ai
```

## Deploy on Render

1. Push code to GitHub.
2. Create a **Web Service** on Render.
3. Use Docker deployment.
4. Set env vars if needed:
   - `MODEL_PATH=models/saved_model.pkl`
   - `LOG_LEVEL=INFO`
5. Render will expose your FastAPI app on public URL.

## Deploy on AWS

### Option A: ECS (recommended)
1. Push Docker image to ECR.
2. Create ECS Fargate service using the image.
3. Attach ALB and expose port 8000.
4. Configure task env vars (`MODEL_PATH`, `LOG_LEVEL`).

### Option B: EC2
1. Launch EC2 instance with Docker.
2. Pull image and run container on port 8000.
3. Use Nginx reverse proxy + HTTPS (ACM/Certbot).

## Notes

- Ensure model is trained (`python main.py`) before serving API.
- For real Kaggle data, replace `data/raw_resume_dataset.csv` with your downloaded dataset and keep required columns unchanged.
