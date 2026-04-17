# MLOps Assignment: Docker-Based ML Deployment

This project trains an Iris classification model, serves predictions with FastAPI, and is ready for Docker containerization and Docker Hub deployment.

## Project Structure

```text
.
├── app/
│   └── main.py
├── artifacts/                 # model files generated after training
├── train.py
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

## Task 1: Train ML Model

```powershell
python -m pip install -r requirements.txt
python train.py
```

This creates:
- `artifacts/iris_model.joblib`
- `artifacts/class_names.joblib`

## Task 2: Run API

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Endpoints
- `GET /health`
- `POST /predict`

Example prediction request:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

## Task 3-5: Dockerfile, Build, and Run

```powershell
docker build -t iris-ml-api:latest .
docker images
docker run -d --name iris-api -p 8000:8000 iris-ml-api:latest
docker ps
```

## Task 6: Push to Docker Hub

Replace `<dockerhub-username>` with your Docker Hub username.

```powershell
docker login
docker tag iris-ml-api:latest <dockerhub-username>/iris-ml-api:latest
docker push <dockerhub-username>/iris-ml-api:latest
```

## Task 7-8: Pull and Run Pulled Image

```powershell
docker pull <dockerhub-username>/iris-ml-api:latest
docker run -d --name iris-api-pulled -p 8001:8000 <dockerhub-username>/iris-ml-api:latest
```

Test pulled image:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8001/health" -Method Get
```

## Mandatory Screenshot Checklist

1. Model training output (`python train.py`)
2. Docker build command (`docker build ...`)
3. Docker images list (`docker images`)
4. Running container (`docker ps`)
5. Docker push success (`docker push ...`)
6. Docker pull success (`docker pull ...`)
7. API prediction result (`POST /predict`)
