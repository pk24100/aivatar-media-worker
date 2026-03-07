# RunPod Serverless Endpoint Setup Script
# This script automates the creation of a RunPod Serverless endpoint for the Ditto TalkingHead model
# Usage: Fill in your credentials below and run: .\runpod_serverless_setup.ps1

# ===============================
# CONFIGURATION - Fill these in
# ===============================
$RUNPOD_API_KEY = "<YOUR_RUNPOD_API_KEY>"  # Get from https://www.runpod.io/console/settings
$NETWORK_VOLUME_NAME = "aivatar-models"     # Your network volume name
$DOCKER_IMAGE = "docker.io/pk24100/aivatar-worker:latest"
$TEMPLATE_NAME = "aivatar-ditto-template"
$ENDPOINT_NAME = "aivatar-ditto"

# LiveKit credentials (optional - for streaming functionality)
$LIVEKIT_URL = "wss://<YOUR_LIVEKIT_URL>"      # e.g., wss://your-project.livekit.cloud
$LIVEKIT_API_KEY = "<YOUR_LIVEKIT_API_KEY>"
$LIVEKIT_API_SECRET = "<YOUR_LIVEKIT_API_SECRET>"

# GPU configuration
$GPU_TYPES = @("NVIDIA L4", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090")
$MIN_WORKERS = 0
$MAX_WORKERS = 3
$IDLE_TIMEOUT = 5
$EXECUTION_TIMEOUT_MS = 300000  # 5 minutes

# ===============================
# DO NOT EDIT BELOW THIS LINE
# ===============================

$Headers = @{
    "Authorization" = "Bearer $RUNPOD_API_KEY"
    "Content-Type" = "application/json"
}

Write-Host "=== RunPod Serverless Endpoint Setup ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Get Network Volume ID
Write-Host "Step 1: Looking up network volume '$NETWORK_VOLUME_NAME'..." -ForegroundColor Yellow
try {
    $volumes = Invoke-RestMethod -Method Get -Uri "https://rest.runpod.io/v1/networkvolumes" -Headers $Headers
    $volume = $volumes | Where-Object { $_.name -eq $NETWORK_VOLUME_NAME } | Select-Object -First 1
    
    if (-not $volume) {
        Write-Error "Network volume '$NETWORK_VOLUME_NAME' not found!"
        Write-Host "Available volumes:" -ForegroundColor Red
        $volumes | ForEach-Object { Write-Host "  - $($_.name) (ID: $($_.id))" -ForegroundColor Red }
        exit 1
    }
    
    $VolumeId = $volume.id
    Write-Host "  Network Volume ID: $VolumeId" -ForegroundColor Green
} catch {
    Write-Error "Failed to get network volumes: $_"
    exit 1
}

Write-Host ""

# Step 2: Create Serverless Template
Write-Host "Step 2: Creating serverless template '$TEMPLATE_NAME'..." -ForegroundColor Yellow

$TemplateBody = @{
    imageName = $DOCKER_IMAGE
    name = $TEMPLATE_NAME
    category = "NVIDIA"
    containerDiskInGb = 20
    dockerEntrypoint = @()
    dockerStartCmd = @()
    env = @{
        MODEL_ROOT = "/app/models/ditto"
        DITTO_REPO_PATH = "/app/ditto-talkinghead"
        AIVATAR_STREAMING = "true"
    }
    isPublic = $false
    isServerless = $true
    ports = @()
    readme = "Ditto TalkingHead Serverless Worker for AiVatar"
    volumeInGb = 20
    volumeMountPath = "/runpod-volume"
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

# Step 3: Create Serverless Endpoint
Write-Host "Step 3: Creating serverless endpoint '$ENDPOINT_NAME'..." -ForegroundColor Yellow

$EndpointBody = @{
    templateId = $TemplateId
    computeType = "GPU"
    gpuCount = 1
    gpuTypeIds = $GPU_TYPES
    allowedCudaVersions = @("12.2", "12.1")
    name = $ENDPOINT_NAME
    networkVolumeId = $VolumeId
    workersMin = $MIN_WORKERS
    workersMax = $MAX_WORKERS
    idleTimeout = $IDLE_TIMEOUT
    executionTimeoutMs = $EXECUTION_TIMEOUT_MS
    scalerType = "QUEUE_DELAY"
    scalerValue = 4
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

# Step 4: Update Template with LiveKit Credentials (optional)
if ($LIVEKIT_URL -ne "wss://<YOUR_LIVEKIT_URL>" -and 
    $LIVEKIT_API_KEY -ne "<YOUR_LIVEKIT_API_KEY>" -and 
    $LIVEKIT_API_SECRET -ne "<YOUR_LIVEKIT_API_SECRET>") {
    
    Write-Host "Step 4: Updating template with LiveKit credentials..." -ForegroundColor Yellow
    
    $UpdateBody = @{
        containerDiskInGb = 20
        dockerEntrypoint = @()
        dockerStartCmd = @()
        env = @{
            MODEL_ROOT = "/runpod-volume/models/ditto"
            DITTO_REPO_PATH = "/runpod-volume/ditto-talkinghead"
            AIVATAR_STREAMING = "true"
            LIVEKIT_URL = $LIVEKIT_URL
            LIVEKIT_API_KEY = $LIVEKIT_API_KEY
            LIVEKIT_API_SECRET = $LIVEKIT_API_SECRET
        }
        imageName = $DOCKER_IMAGE
        isPublic = $false
        name = $TEMPLATE_NAME
        ports = @()
        readme = "Ditto TalkingHead Serverless Worker for AiVatar"
        volumeInGb = 20
        volumeMountPath = "/runpod-volume"
    }
    
    try {
        Invoke-RestMethod -Method Patch -Uri "https://rest.runpod.io/v1/templates/$TemplateId" -Headers $Headers -Body ($UpdateBody | ConvertTo-Json -Depth 10) | Out-Null
        Write-Host "  LiveKit credentials added successfully" -ForegroundColor Green
    } catch {
        Write-Warning "Failed to update template with LiveKit credentials: $_"
    }
} else {
    Write-Host "Step 4: Skipping LiveKit setup (credentials not configured)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Setup Complete! ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Summary:" -ForegroundColor White
Write-Host "  Template ID: $TemplateId"
Write-Host "  Endpoint ID: $EndpointId"
Write-Host "  Network Volume: $VolumeId"
Write-Host ""
Write-Host "Your endpoint will be available at:" -ForegroundColor White
Write-Host "  https://api.runpod.ai/v2/$EndpointId/run" -ForegroundColor Green
Write-Host ""
Write-Host "To test your endpoint, use:" -ForegroundColor White
Write-Host "  .\runpod_endpoint_test.ps1 -EndpointId $EndpointId" -ForegroundColor Green
