# Hosted API Testing (Isolated)

This folder contains only hosted/deployed API test artifacts.
Existing project files remain unchanged.

## Target API
- Service: Mindspace Text Classifier API
- URL: `http://88.222.212.15:9000`
- Required header: `X-API-Key`

## Script
- `test_text_api_hosted.py`

## Run
From project root:

```powershell
C:/Users/vicky/OneDrive/Desktop/deployment-text/myenv/Scripts/python.exe hosted_api_testing/test_text_api_hosted.py
```

Optional override for another environment:

```powershell
$env:TEXT_API_BASE_URL = "http://127.0.0.1:9000"
C:/Users/vicky/OneDrive/Desktop/deployment-text/myenv/Scripts/python.exe hosted_api_testing/test_text_api_hosted.py
```
