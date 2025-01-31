FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 必要なパッケージをまとめてインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        ca-certificates \
        && add-apt-repository ppa:deadsnakes/ppa -y \
        && apt-get update \
        && apt-get install -y --no-install-recommends \
            python3.11 \
            python3.11-dev \
            python3.11-distutils \
        && ln -s /usr/bin/python3.11 /usr/local/bin/python3 \
        && ln -s /usr/bin/python3.11 /usr/local/bin/python \
        && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash appuser
USER appuser
WORKDIR /app

ENV PATH="/home/appuser/.local/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python -

COPY --chown=appuser:appuser pyproject.toml poetry.lock* ./
RUN poetry install --no-root

COPY --chown=appuser:appuser . .
RUN poetry install

CMD ["python", "src/main.py", "--model", "gpt-4o", "--judge_model", "gpt-4o"]
