FROM python:3.12-slim
# git: used by the git_diff_summarize tool. The torch wheel pulled in by sentence-transformers is large
# (RAG embeddings/reranker run locally); a CPU-only torch index can slim the image.
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* \
 && useradd --uid 10001 --create-home app
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
USER app
ENV HOME=/home/app WORKSPACE_ROOT=/workspace
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
