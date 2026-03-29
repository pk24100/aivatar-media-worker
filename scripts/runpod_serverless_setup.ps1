# RunPod Serverless Endpoint Setup Script
# This script automates the creation of a RunPod Serverless endpoint for the SoulX-FlashHead Lite worker
# Usage: Fill in your credentials below and run: .\runpod_serverless_setup.ps1

# ===============================
# CONFIGURATION - Fill these in
# ===============================
$RUNPOD_API_KEY = "<YOUR_RUNPOD_API_KEY>"  # Get from https://www.runpod.io/console/settings
$DOCKER_IMAGE = "docker.io/pk24100/aivatar-worker:flashhead-lite"
$TEMPLATE_NAME = "aivatar-flashhead-template"
$ENDPOINT_NAME = "aivatar-flashhead-serverless"

$LIVEKIT_URL = "wss://<YOUR_LIVEKIT_URL>"      # e.g., wss://your-project.livekit.cloud
$LIVEKIT_API_KEY = "<YOUR_LIVEKIT_API_KEY>"
$LIVEKIT_API_SECRET = "<YOUR_LIVEKIT_API_SECRET>"

# GPU configuration
$GPU_TYPES = @("NVIDIA GeForce RTX 4090")
$MIN_WORKERS = 0
$MAX_WORKERS = 10
$WORKER_CONCURRENCY = 3
$IDLE_TIMEOUT = 5
$EXECUTION_TIMEOUT_MS = 1800000

# ===============================
# DO NOT EDIT BELOW THIS LINE
# ===============================

$Headers = @{
    "Authorization" = "Bearer $RUNPOD_API_KEY"
    "Content-Type" = "application/json"
}

Write-Host "=== RunPod Serverless Endpoint Setup ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Create Serverless Template
Write-Host "Step 1: Creating serverless template '$TEMPLATE_NAME'..." -ForegroundColor Yellow

$TemplateBody = @{
    imageName = $DOCKER_IMAGE
    name = $TEMPLATE_NAME
    category = "NVIDIA"
    containerDiskInGb = 20
    dockerEntrypoint = @()
    dockerStartCmd = @()
    env = @{
        FLASHHEAD_CKPT_DIR = "/app/models/SoulX-FlashHead-1_3B"
        WAV2VEC_DIR = "/app/models/wav2vec2-base-960h"
        FLASHHEAD_REPO_PATH = "/app/SoulX-FlashHead"
        AIVATAR_STREAMING = "true"
        AIVATAR_RUNTIME_MODE = "serverless"
        AIVATAR_WORKER_CONCURRENCY = "$WORKER_CONCURRENCY"
        LIVEKIT_URL = $LIVEKIT_URL
    }
    isPublic = $false
    isServerless = $true
    ports = @()
    readme = "SoulX-FlashHead Lite Serverless Worker for AiVatar"
}

try {
    $templateResponse = Invoke-RestMethod -Method Post -Uri "https://rest.runpod.io/v1/templates" -Headers $Headers -Body ($TemplateBody | ConvertTo-Json -Depth 10)
    $TemplateId = $templateResponse.id
    Write-Host "  Template ID: $TemplateId" -ForegroundColor Green
} catch {
    Write-Error "Failed to create template: $_"
    exit 1
}

Write-Host ""

# Step 2: Create Serverless Endpoint
Write-Host "Step 2: Creating serverless endpoint '$ENDPOINT_NAME'..." -ForegroundColor Yellow

$EndpointBody = @{
    templateId = $TemplateId
    computeType = "GPU"
    gpuCount = 1
    gpuTypeIds = $GPU_TYPES
    allowedCudaVersions = @("12.2", "12.1")
    name = $ENDPOINT_NAME
    workersMin = $MIN_WORKERS
    workersMax = $MAX_WORKERS
    idleTimeout = $IDLE_TIMEOUT
    executionTimeoutMs = $EXECUTION_TIMEOUT_MS
    scalerType = "QUEUE_DELAY"
    scalerValue = 4
    flashboot = $true
}

try {
    $endpointResponse = Invoke-RestMethod -Method Post -Uri "https://rest.runpod.io/v1/endpoints" -Headers $Headers -Body ($EndpointBody | ConvertTo-Json -Depth 10)
    $EndpointId = $endpointResponse.id
    Write-Host "  Endpoint ID: $EndpointId" -ForegroundColor Green
} catch {
    Write-Error "Failed to create endpoint: $_"
    exit 1
}

Write-Host ""

Write-Host "=== Setup Complete! ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Summary:" -ForegroundColor White
Write-Host "  Template ID: $TemplateId"
Write-Host "  Endpoint ID: $EndpointId"
Write-Host "  GPU Types: $($GPU_TYPES -join ', ')"
Write-Host "  Worker Concurrency: $WORKER_CONCURRENCY"
Write-Host ""
Write-Host "Your endpoint will be available at:" -ForegroundColor White
Write-Host "  https://api.runpod.ai/v2/$EndpointId/run" -ForegroundColor Green
Write-Host ""
Write-Host "Configure the endpoint to expose TCP/HTTP port 8765 for websocket ingestion." -ForegroundColor Yellow
Write-Host "If you reuse the image for overflow pods, also expose HTTP port 8000 for /readyz and /sessions endpoints." -ForegroundColor Yellow
Write-Host ""
Write-Host "Recommended environment variables:" -ForegroundColor White
Write-Host "  FLASHHEAD_CKPT_DIR=/app/models/SoulX-FlashHead-1_3B"
Write-Host "  WAV2VEC_DIR=/app/models/wav2vec2-base-960h"
Write-Host "  FLASHHEAD_REPO_PATH=/app/SoulX-FlashHead"
Write-Host "  LIVEKIT_URL=$LIVEKIT_URL"
Write-Host "  AIVATAR_STREAMING=true"
Write-Host "  AIVATAR_RUNTIME_MODE=serverless"
Write-Host "  AIVATAR_WORKER_CONCURRENCY=$WORKER_CONCURRENCY"
Write-Host ""
Write-Host "To smoke test the endpoint, use:" -ForegroundColor White
Write-Host "  .\runpod_endpoint_test.ps1 -EndpointId $EndpointId -RunPodToken $RUNPOD_API_KEY" -ForegroundColor Green
