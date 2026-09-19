FROM python:3.12-slim

# git: the git_diff_summarize tool shells out to it.
RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --uid 10001 --create-home app

WORKDIR /srv
COPY requirements.txt .

# sentence-transformers (local RAG embeddings + reranker) pulls in torch, and torch from PyPI drags
# ~2.2 GB of CUDA libraries (cudnn, cublas, nccl, cusparselt, nvshmem) with it. This container does
# inference on CPU - the chat model runs in a separate Ollama workload - so install the CPU-only
# build FIRST and let sentence-transformers resolve against it.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -r requirements.txt

COPY app ./app
USER app
ENV HOME=/home/app WORKSPACE_ROOT=/workspace
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
