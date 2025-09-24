#/bin/sh

uvicorn space_restorer.service:app --reload --host 0.0.0.0 --port 8000
