FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

LABEL description="Chemia: Molecular ML Framework"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxext6 libsm6 libx11-6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -c conda-forge rdkit>=2022.03.1 -y && conda clean -afy

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --no-cache-dir unimol-tools>=0.1.5

COPY . /app

ENV PYTHONPATH=/app
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

RUN useradd -m -s /bin/bash chemia && chown -R chemia:chemia /app
USER chemia
RUN mkdir -p /home/chemia/.cache /app/output

ENTRYPOINT ["python"]
CMD ["scripts/run_training_only.py", "--help"]
