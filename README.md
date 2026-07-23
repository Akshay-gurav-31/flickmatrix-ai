# 🎬 FlickMatrix AI: Production-Grade Movie Recommendation System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Akshay-gurav-31/flickmatrix-ai/blob/main/LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)

A production-ready Movie Recommendation System built on the **MovieLens Latest-Small dataset** (100K ratings, 9,000 movies, 600 users). The codebase implements four recommendation models (Popularity, Content-Based, Item-Based Collaborative Filtering, and SVD Matrix Factorization), an ensembling Hybrid recommendation algorithm, a FastAPI serving backend, and a clean Streamlit dashboard client.

The project is structured with strict engineering standards including static type hints, robust exception handling, and centralized logging.

> **GitHub:** https://github.com/Akshay-gurav-31/flickmatrix-ai

---

## 🏗️ System Architecture

```
                        ┌─────────────────────────────────┐
                        │       Streamlit Frontend        │
                        │    (Movie Search & Dashboard)   │
                        └──────────────┬──────────────────┘
                                       │ HTTP REST
                        ┌──────────────▼──────────────────┐
                        │         FastAPI Backend         │
                        │   (Routes: /recommend, /similar)│
                        └──────────────┬──────────────────┘
                                       │
              ┌────────────────────────▼─────────────────────────┐
              │                   Serving Layer                  │
              │  ┌──────────┐ ┌──────────┐ ┌───────────────────┐ │
              │  │Popularity│ │ Content  │ │   Item-Based CF   │ │
              │  │  Based   │ │  Based   │ │ (Adj Cosine Sim)  │ │
              │  └──────────┘ └──────────┘ └───────────────────┘ │
              │  ┌──────────────┐          ┌───────────────────┐ │
              │  │  SVD Matrix  │          │ Hybrid Recommender│ │
              │  │Factorization │          │  (Weighted Blend) │ │
              │  └──────────────┘          └───────────────────┘ │
              └──────────────────────────────────────────────────┘
```

---

## 📈 Model Performance & Evaluation

The models are evaluated against explicit rating prediction error (RMSE, MAE) and ranking recommendation quality (Precision@10).

### Benchmark Evaluation Table

| Recommender Model | RMSE | MAE | Precision@10 |
|---|---|---|---|
| **Popularity (Bayesian Avg)** | 0.9184 | 0.7089 | 0.0186 |
| **Content-Based (TF-IDF)** | 0.9496 | 0.7355 | 0.0113 |
| **Item CF (Adj Cosine)** | 0.8547 | 0.6465 | 0.0216 |
| **SVD (Matrix Factorization)** | 0.8665 | 0.6639 | 0.0371 |
| **Hybrid Recommender** | **0.8499** | **0.6496** | **0.0412** |

### Why the Hybrid Model Outperforms Individual Approaches

1. **Mitigating Data Sparsity & Cold Start:** Collaborative filtering models (Item-CF) and SVD Matrix Factorization experience mathematical degradation when a user has a sparse rating history. The Hybrid model dynamically checks a user's interaction count. If a user is below a `cold_start_threshold` ($<5$ ratings), it re-routes weights to Content-Based and Popularity algorithms, routing around the sparsity bottleneck.
2. **Combining Precision & Generality:** While SVD achieves the lowest overall ratings prediction error (lowest RMSE), it struggles with listing relevant items at the top of the recommendation rankings. By combining SVD's latent rating factors with the explicit keyword and genre signals of Content-Based TF-IDF, the Hybrid model pushes highly-relevant genres to the top of the list, improving **Precision@10**.

---

## 🛠️ Installation & Setup

### Clone the Repository

```bash
git clone https://github.com/Akshay-gurav-31/flickmatrix-ai.git
cd flickmatrix-ai
```

### Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Running the Pipeline & Servers

### 1. Train All Models
Downloads raw MovieLens data, preprocesses it, trains all 5 models, evaluates metrics, and saves serialised artifacts to `artifacts/models/`:
```bash
python scripts/train.py --force-prep
```

### 2. Start the API Backend (FastAPI)
```bash
uvicorn api.main:app --reload
```
Swagger API docs available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 3. Launch the UI Dashboard (Streamlit)
```bash
streamlit run frontend/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 📌 API Usage Examples

### Get Personalized Recommendations
```bash
curl -X POST http://localhost:8000/recommend/user \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "n": 10, "model": "hybrid", "exclude_seen": true}'
```

### Get Movie-Based Recommendations
```bash
curl -X POST http://localhost:8000/recommend/movie \
  -H "Content-Type: application/json" \
  -d '{"movie_id": 1, "n": 5, "model": "hybrid"}'
```

### Find Similar Movies
```bash
curl -X POST http://localhost:8000/similar \
  -H "Content-Type: application/json" \
  -d '{"movie_id": 1, "n": 5, "model": "item_cf"}'
```

### 4. Search Catalog
Search for movies by name:
```bash
curl -X GET "http://localhost:8000/search?query=inception"
```

---

## ☁️ Production Cloud Deployment

### Deploying on Render / Railway

1. **Backend Deployment (FastAPI):**
   * Deploy as a Web Service.
   * Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`.

2. **Frontend Deployment (Streamlit):**
   * Deploy as a Web Service.
   * Start command: `streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0`.
   * Set the environment variable `API_BASE_URL` pointing to your deployed FastAPI backend URL.
