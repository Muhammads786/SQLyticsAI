# LLMOps Starter (FastAPI + LangGraph + vLLM/TGI + Qdrant + MLflow)

Production-ready scaffold to bootstrap a local lab or cloud-ready LLMOps stack.

## Stack
- **API**: FastAPI (OpenAI-compatible responses where helpful)
- **Orchestration**: LangGraph (retrieve → generate → guard → cite)
- **Serving**: vLLM (OpenAI-compatible) or TGI (swap service)
- **Vector DB**: Qdrant (with simple chunking/embedding pipeline)
- **Tracking**: MLflow (experiments, metrics, artifacts)
- **Prompt CI**: promptfoo (goldens & regressions)
- **RAG evals**: RAGAS (faithfulness, context precision/recall)
- **IaC**: Terraform (AWS stub) & Bicep (Azure stub)
- **Observability**: OpenTelemetry hooks (extensible), basic logs

## Quickstart
```bash
# 0) Copy env template and adjust
cp .env.example .env

# 1) Start core services (Qdrant, MLflow, vLLM, API)
docker compose up -d --build

# 2) Ingest sample data to Qdrant
docker compose exec api python scripts/ingest.py data

# 3) Call the chat endpoint
curl -s http://localhost:8000/chat -H "Content-Type: application/json"   -d '{"message":"What is this project about? Please cite."}' | jq .

# 4) Run RAG evals (offline set in evals/goldens.jsonl)
docker compose exec api python evals/ragas_eval.py

# 5) Prompt CI (requires Node for promptfoo)
npx -y promptfoo@latest eval --config promptci/promptfoo.yaml
```

## Services
- API: http://localhost:8000 (docs at `/docs`)
- Qdrant: http://localhost:6333
- MLflow: http://localhost:5000
- vLLM (OpenAI-compatible): http://localhost:8001/v1

## Swap TGI for vLLM
Edit `docker-compose.yml` and switch the `llm` service to TGI image:
- hfai/ text-generation-inference with `--model-id` and `--port 8001`.
Make sure the API uses the selected backend URL via `.env`.

## Deploying to Cloud
- **AWS**: Use `infra/aws/terraform` (S3, ECR/ECS/EKS, Bedrock/SageMaker add-ons).
- **Azure**: Use `infra/azure/bicep` (AKS or App Service, AI Search, Azure OpenAI).

> This is a minimal but extensible scaffold: add auth (JWT/OAuth), API Gateway, content safety, and enterprise logs as you move to prod.
