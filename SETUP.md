# Setup Guide

## Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) Docker for containerization
- (Optional) Kubernetes cluster with Helm for deployment
- (Optional) Redis server for caching

## Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/imrohankataria/rag-cost-optimized-vdb-helm.git
cd rag-cost-optimized-vdb-helm
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI Configuration (required for LLM features)
OPENAI_API_KEY=your-openai-api-key-here

# Redis Configuration (optional, for caching)
REDIS_HOST=localhost
REDIS_PORT=6379

# Application Configuration
LOG_LEVEL=INFO
TRACK_COSTS=true
```

### 5. Run Tests

```bash
pytest tests/ -v
```

## Running the System

### Option 1: Run Demo Examples

```bash
python examples/demo.py
```

This will run all demo scenarios and generate cost visualizations.

### Option 2: Run API Server

```bash
python src/api_server.py
```

The API server will start on `http://localhost:8000`

Access the API documentation at: `http://localhost:8000/docs`

### Option 3: Use as a Library

```python
from src.rag_system import RAGSystem

# Initialize
rag = RAGSystem(enable_cache=True)

# Add documents
documents = ["Document 1 text...", "Document 2 text..."]
rag.add_documents(documents)

# Query
result = rag.query("What is machine learning?")
print(result['answer'])
print(f"Cost: ${result['cost']:.4f}")
```

## Docker Setup

### Build Docker Image

```bash
docker build -t rag-cost-optimized:1.0.0 .
```

### Run Container

```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  -e REDIS_HOST=host.docker.internal \
  rag-cost-optimized:1.0.0
```

### Docker Compose (with Redis)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  rag-system:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    depends_on:
      - redis
    volumes:
      - ./data:/data

volumes:
  redis-data:
```

Run with:

```bash
docker-compose up
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (minikube, GKE, EKS, AKS, etc.)
- Helm 3.x installed
- kubectl configured

### Deploy with Helm

1. **Create namespace:**

```bash
kubectl create namespace rag-system
```

2. **Create secret for OpenAI API key:**

```bash
kubectl create secret generic openai-secret \
  --from-literal=api-key=your-openai-api-key \
  -n rag-system
```

3. **Install Redis (if not already available):**

```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install redis bitnami/redis -n rag-system
```

4. **Install RAG system:**

```bash
cd helm/rag-cost-optimized

# Install with default values
helm install rag-system . -n rag-system

# Or with custom values
helm install rag-system . -n rag-system \
  --set openai.apiKey=your-key \
  --set redis.host=redis-master \
  --set replicaCount=2
```

5. **Verify deployment:**

```bash
kubectl get pods -n rag-system
kubectl get svc -n rag-system
```

6. **Access the service:**

```bash
# Port forward for local access
kubectl port-forward svc/rag-system-rag-cost-optimized 8000:8000 -n rag-system

# Or create an ingress
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-system-ingress
  namespace: rag-system
spec:
  rules:
  - host: rag-system.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rag-system-rag-cost-optimized
            port:
              number: 8000
EOF
```

### Helm Chart Values

Key configuration options in `values.yaml`:

```yaml
# Scaling
replicaCount: 2

# Resources
resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 1Gi

# OpenAI
openai:
  apiKey: "your-key"
  embeddingModel: "text-embedding-ada-002"
  llmModel: "gpt-3.5-turbo"

# Cache
cache:
  enabled: true
  ttl: 3600

# Persistence
persistence:
  enabled: true
  size: 10Gi
```

## Monitoring and Observability

### View Logs

```bash
# Docker
docker logs -f <container-id>

# Kubernetes
kubectl logs -f deployment/rag-system-rag-cost-optimized -n rag-system
```

### Check Costs

```bash
# Via API
curl http://localhost:8000/costs

# Via kubectl (in Kubernetes)
kubectl exec -it deployment/rag-system-rag-cost-optimized -n rag-system -- \
  curl http://localhost:8000/costs
```

### Generate Visualizations

```bash
# Via API
curl -X GET http://localhost:8000/visualizations/generate
```

## Troubleshooting

### Issue: Cannot connect to Redis

**Solution:** Ensure Redis is running and accessible:

```bash
# Check Redis connection
redis-cli ping

# Or disable cache in environment
export CACHE_ENABLED=false
```

### Issue: OpenAI API errors

**Solution:** Verify API key and rate limits:

```bash
# Test API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Issue: Out of memory

**Solution:** Increase resource limits:

```yaml
# In values.yaml
resources:
  limits:
    memory: 4Gi
```

## Upgrading

### Helm Upgrade

```bash
helm upgrade rag-system ./helm/rag-cost-optimized -n rag-system
```

### Rolling Update

```bash
kubectl set image deployment/rag-system-rag-cost-optimized \
  rag-cost-optimized=rag-cost-optimized:1.1.0 \
  -n rag-system
```

## Uninstall

### Helm

```bash
helm uninstall rag-system -n rag-system
kubectl delete namespace rag-system
```

### Docker

```bash
docker-compose down -v
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/imrohankataria/rag-cost-optimized-vdb-helm/issues
- Documentation: See README.md
