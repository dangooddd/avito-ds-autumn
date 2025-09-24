FROM python:3.12-slim-bookworm

RUN apt update
RUN apt install -y wget
RUN pip install --root-user-action ignore uv

COPY uv.lock pyproject.toml LICENSE start.sh /app/
COPY src/ /app/src/

WORKDIR /app

RUN chmod +x start.sh
RUN uv venv .venv
RUN uv sync --no-dev

EXPOSE 8000

CMD ["./start.sh"]
