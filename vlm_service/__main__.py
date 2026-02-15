"""Run the VLM service: python -m vlm_service"""
import os
import uvicorn

if __name__ == "__main__":
    host = os.environ.get("VLM_HOST", "127.0.0.1")
    port = int(os.environ.get("VLM_PORT", "5000"))
    uvicorn.run("vlm_service.server:app", host=host, port=port, reload=False)
