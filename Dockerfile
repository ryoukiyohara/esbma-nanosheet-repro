# Same base image and package versions as the environment used for the analyses.
FROM python:3.9
WORKDIR /repro
COPY requirements.txt /repro/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# build:  docker build -t esbma-repro .
# run:    docker run --rm -v "$PWD":/repro esbma-repro \
#           jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
#           esbma_paper_reproduction.ipynb --output esbma_paper_reproduction.executed.ipynb
