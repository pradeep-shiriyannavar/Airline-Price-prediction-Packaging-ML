# ✈️ Airline Price Prediction API 

A production-ready Machine Learning API for predicting airline ticket prices using a trained regression model.

This project demonstrates:
- ML model packaging
- FastAPI deployment
- Docker containerization

---

## 📂 Project Structure

```

Airline-Price-prediction-Packaging-ML/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
│
└── airline_prediction/
├── **init**.py
├── predict.py
├── pipeline/
├── config/
├── artifacts/
└── utils/

````

---

## 🚀 Tech Stack

- Python 3.10
- FastAPI
- Uvicorn
- Scikit-learn
- Docker

---

## 📊 Dataset Source

The dataset used for model training was collected using the **Amadeus Flight Offers Search API**.

Acknowledgement:
Data sourced from the Amadeus API for educational and research purposes.

---

## 🔧 Run Locally (Without Docker)

### 1️⃣ Create Virtual Environment

```bash
conda create -n myproject python=3.10
conda activate myproject
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run FastAPI App

```bash
uvicorn app:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## 🐳 Run With Docker

### 1️⃣ Build Docker Image

```bash
docker build -t airline-price-api .
```

### 2️⃣ Run Container

```bash
docker run -p 8000:8000 airline-price-api
```

Open:

```
http://localhost:8000/docs
```


## 🎯 Future Enhancements

* Push Docker image to AWS ECR
* Deploy using AWS ECS (Fargate)
* Add CloudWatch Logging
* Implement Auto-scaling

---

