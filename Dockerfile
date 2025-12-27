FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
WORKDIR /workspace
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml .
RUN pip install --no-cache-dir .[dev]
COPY . .
RUN pip install .
ENV CONFIG_PATH="/workspace/config.yaml"
ENTRYPOINT ["aetheria"]
CMD ["--help"]
