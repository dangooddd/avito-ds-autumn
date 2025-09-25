FROM python:3.12-slim-bookworm

RUN apt update
RUN apt install -y wget unzip
RUN pip install --root-user-action ignore --no-cache-dir uv

COPY uv.lock pyproject.toml LICENSE start.sh yd-wget.py download.sh /app/
COPY src/ /app/src/

WORKDIR /app

RUN chmod +x start.sh
RUN chmod +x download.sh
RUN chmod +x yd-wget.py
RUN uv venv .venv
RUN uv sync --no-dev
RUN uv cache clean

EXPOSE 8000

CMD ["./start.sh"]
