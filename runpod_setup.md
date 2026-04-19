# Deploying to RunPod

To transition your AI transcription workload to an external NVIDIA GPU on RunPod.io, follow these exact deployment steps.

## 1. Prepare Docker Image (Optional / Suggested)
This repository includes a `Dockerfile` pre-configured to bundle CUDA + Faster-Whisper + Ollama + the Streaming WS endpoints.
You can build and push it to a public docker registry (DockerHub), or simply paste the code inside a basic RunPod instance.

**Pre-built approach:**
```bash
docker build -t your-docker-name/runpod-subtitles:latest .
docker push your-docker-name/runpod-subtitles:latest
```

## 2. Booting a Serverless / Secure Pod
1. Go to **RunPod.io** -> **Pods**.
2. Click **Deploy**. Choose a GPU (e.g., RTX 4090 or RTX 3090, 24GB is enough for `small` whisper and `8b` LLMs).
3. **Template selection**: Select a standard PyTorch/CUDA template, or use your custom Docker Hub image if you pushed it.
4. **Ports**: *CRITICAL* — Expose HTTP Port **8000**.
5. Once your pod is running, click **Connect** -> **HTTP Interface** to grab your public port URL.
It will look something like: `https://abcd1234efgh-8000.proxy.runpod.net`

## 3. Local App Configuration
1. Open your Mac's local `.env` file for Vertaal.
2. Change the pipeline mode:
   `PIPELINE_MODE=runpod`
3. Convert the HTTP URL to a WSS URL (WebSocket Secure) and paste it:
   `RUNPOD_WSS_URL=wss://abcd1234efgh-8000.proxy.runpod.net/stream`

## 4. Run!
Boot your local `app.py`. It will immediately connect to your RunPod instance via WebSockets. Your Mac CPU will remain dormant while the NVIDIA GPU crushes the STT & Translations!
