# RunPod Serverless Endpoint Setup Script
# This script automates the creation of a RunPod Serverless endpoint for the SoulX-FlashHead Lite worker
# Usage: Fill in your credentials below and run: .\runpod_serverless_setup.ps1

# ===============================
# CONFIGURATION - Fill these in
# ===============================
$RUNPOD_API_KEY = "<YOUR_RUNPOD_API_KEY>"  # Get from https://www.runpod.io/console/settings
$DOCKER_IMAGE = "docker.io/pk24100/aivatar-worker:flashhead-lite-v3"
$TEMPLATE_NAME = "aivatar-flashhead-template"
$ENDPOINT_NAME = "aivatar-flashhead-serverless"
$FLASHHEAD_MODEL_REPO = "pkam24100/aivatar-flashhead-model"

$LIVEKIT_URL = "wss://<YOUR_LIVEKIT_URL>"      # e.g., wss://your-project.livekit.cloud

# GPU configuration
$GPU_TYPES = @("NVIDIA GeForce RTX 4090")
$MIN_WORKERS = 0
$MAX_WORKERS = 10
$WORKER_CONCURRENCY = 3
$IDLE_TIMEOUT = 300
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
    scalerType = "REQUEST_COUNT"
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
Write-Host "  https://$EndpointId.api.runpod.ai" -ForegroundColor Green
Write-Host ""
Write-Host "Manual console step required for cached models:" -ForegroundColor Yellow
Write-Host "  RunPod Console -> Serverless -> $ENDPOINT_NAME -> Manage -> Edit Endpoint" -ForegroundColor White
Write-Host "  Set Model = $FLASHHEAD_MODEL_REPO" -ForegroundColor White
Write-Host "  Add your Hugging Face access token for the private repo" -ForegroundColor White
Write-Host ""
Write-Host "Required environment variables:" -ForegroundColor White
Write-Host "  LIVEKIT_URL=$LIVEKIT_URL"
Write-Host ""
Write-Host "To smoke test the endpoint, use:" -ForegroundColor White
Write-Host "  .\runpod_endpoint_test.ps1 -EndpointId $EndpointId -RunPodToken $RUNPOD_API_KEY" -ForegroundColor Green
