FROM nvidia/cuda:12.9.0-base-ubuntu24.04
# FROM nvidia/cuda:13.0.1-base-ubuntu24.04
RUN apt-get update && apt-get install -y python3-pip curl git

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

COPY ./pyproject.toml /app/pyproject.toml
WORKDIR /app
RUN uv sync
COPY ./ /app/
RUN uv pip install -e .

CMD ["uv", "run", "src/rank_comparia/export.py"]