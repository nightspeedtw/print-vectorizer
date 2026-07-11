# Print Vectorizer API

Public FastAPI backend for the Lovable Print Vectorizer frontend.

## Environment

- `ALLOWED_ORIGINS`: comma-separated frontend origins.
- `MAX_UPLOAD_MB`: upload limit, default `50`.
- `FILE_RETENTION_HOURS`: temporary result retention window, default `24`.
- `API_KEY`: optional bearer token. Leave empty for a public demo.
- `STORAGE_DIR`: temporary output folder.

## Health Check

`GET /health`

```json
{"status":"ok","service":"print-vectorizer"}
```

## Lovable

Set:

```text
VITE_VECTORIZER_API_URL=https://your-render-service.onrender.com
```
