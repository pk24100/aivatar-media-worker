# RunPod Serverless endpoint setup for the AiVatar media worker.
# Create the named RunPod secret before running this script. The template maps
# that secret into WORKER_AUTH_SECRET without placing its value in this file.

[CmdletBinding()]
param(
    [string]$RunPodApiKey = $env:RUNPOD_API_KEY,
    [Parameter(Mandatory = $true)]
    [string]$BackendInternalUrl,
    [string]$WorkerAuthSecretName = "aivatar_worker_auth_secret",
    [string]$LiveKitUrl = $env:LIVEKIT_URL,
    [string]$DockerImage = "docker.io/pk24100/aivatar-worker:flashhead-lite-v3",
    [string]$TemplateName = "aivatar-flashhead-template",
    [string]$EndpointName = "aivatar-flashhead-serverless",
    [string]$FlashHeadModelRepo = "pkam24100/aivatar-flashhead-model",
    [string[]]$GpuTypes = @("NVIDIA GeForce RTX 4090"),
    [ValidateRange(0, 100)]
    [int]$MinWorkers = 0,
    [ValidateRange(1, 100)]
    [int]$MaxWorkers = 10,
    [ValidateRange(1, 32)]
    [int]$WorkerConcurrency = 3,
    [ValidateRange(1, 300)]
    [int]$LifecycleHeartbeatSeconds = 15,
    [ValidateRange(5, 3600)]
    [int]$IdleTimeout = 300,
    [ValidateRange(1000, 86400000)]
    [int]$ExecutionTimeoutMs = 1800000
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RunPodApiKey)) {
    throw "Set RUNPOD_API_KEY or pass -RunPodApiKey. Do not store the key in this script."
}

$backendUri = $null
if (-not [Uri]::TryCreate($BackendInternalUrl, [UriKind]::Absolute, [ref]$backendUri) -or $backendUri.Scheme -ne "https") {
    throw "BackendInternalUrl must be an absolute HTTPS URL reachable from RunPod workers."
}

if ($WorkerAuthSecretName -notmatch '^[A-Za-z0-9_]+$') {
    throw "WorkerAuthSecretName may contain only letters, digits, and underscores."
}

if ($MinWorkers -gt $MaxWorkers) {
    throw "MinWorkers cannot exceed MaxWorkers."
}

$headers = @{
    Authorization = "Bearer $RunPodApiKey"
    "Content-Type" = "application/json"
}

$templateEnvironment = @{
    BACKEND_INTERNAL_URL = $BackendInternalUrl.TrimEnd('/')
    WORKER_AUTH_SECRET = "{{ RUNPOD_SECRET_$WorkerAuthSecretName }}"
    WORKER_LIFECYCLE_HEARTBEAT_SECONDS = "$LifecycleHeartbeatSeconds"
    AIVATAR_WORKER_CONCURRENCY = "$WorkerConcurrency"
}
if (-not [string]::IsNullOrWhiteSpace($LiveKitUrl)) {
    $templateEnvironment.LIVEKIT_URL = $LiveKitUrl
}

Write-Host "=== RunPod Serverless Endpoint Setup ===" -ForegroundColor Cyan
Write-Host "Prerequisite: RunPod secret '$WorkerAuthSecretName' must match the backend WORKER_AUTH_SECRET." -ForegroundColor Yellow
Write-Host ""

Write-Host "Step 1: Creating serverless template '$TemplateName'..." -ForegroundColor Yellow
$templateBody = @{
    imageName = $DockerImage
    name = $TemplateName
    category = "NVIDIA"
    containerDiskInGb = 20
    dockerEntrypoint = @()
    dockerStartCmd = @()
    env = $templateEnvironment
    isPublic = $false
    isServerless = $true
    ports = @()
    readme = "SoulX-FlashHead Lite Serverless Worker for AiVatar"
}

try {
    $templateResponse = Invoke-RestMethod `
        -Method Post `
        -Uri "https://rest.runpod.io/v1/templates" `
        -Headers $headers `
        -Body ($templateBody | ConvertTo-Json -Depth 10)
    $templateId = $templateResponse.id
    Write-Host "  Template ID: $templateId" -ForegroundColor Green
} catch {
    Write-Error "Failed to create the RunPod template: $($_.Exception.Message)"
    exit 1
}

Write-Host ""
Write-Host "Step 2: Creating serverless endpoint '$EndpointName'..." -ForegroundColor Yellow
$endpointBody = @{
    templateId = $templateId
    computeType = "GPU"
    gpuCount = 1
    gpuTypeIds = $GpuTypes
    allowedCudaVersions = @("12.2", "12.1")
    name = $EndpointName
    workersMin = $MinWorkers
    workersMax = $MaxWorkers
    idleTimeout = $IdleTimeout
    executionTimeoutMs = $ExecutionTimeoutMs
    scalerType = "REQUEST_COUNT"
    scalerValue = 4
    flashboot = $true
}

try {
    $endpointResponse = Invoke-RestMethod `
        -Method Post `
        -Uri "https://rest.runpod.io/v1/endpoints" `
        -Headers $headers `
        -Body ($endpointBody | ConvertTo-Json -Depth 10)
    $endpointId = $endpointResponse.id
    Write-Host "  Endpoint ID: $endpointId" -ForegroundColor Green
} catch {
    Write-Error "Failed to create the RunPod endpoint: $($_.Exception.Message)"
    exit 1
}

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Cyan
Write-Host "  Template ID: $templateId"
Write-Host "  Endpoint ID: $endpointId"
Write-Host "  GPU Types: $($GpuTypes -join ', ')"
Write-Host "  Worker Concurrency: $WorkerConcurrency"
Write-Host "  Lifecycle Heartbeat: $LifecycleHeartbeatSeconds seconds"
Write-Host ""
Write-Host "Endpoint API base:" -ForegroundColor White
Write-Host "  https://$endpointId.api.runpod.ai" -ForegroundColor Green
Write-Host ""
Write-Host "Manual console step required for cached models:" -ForegroundColor Yellow
Write-Host "  RunPod Console -> Serverless -> $EndpointName -> Manage -> Edit Endpoint"
Write-Host "  Set Model = $FlashHeadModelRepo"
Write-Host "  Add a Hugging Face secret only if the model repository requires it"
Write-Host ""
Write-Host "RunPod strict HTTP worker affinity remains a backend feature flag."
Write-Host "Do not enable RUNPOD_WORKER_AFFINITY_ENABLED until the deployed HTTP path is validated."
Write-Host "This setup does not claim that RunPod WebSocket Upgrade requests honor worker affinity."
Write-Host ""
Write-Host "Smoke test command:" -ForegroundColor White
Write-Host "  .\runpod_endpoint_test.ps1 -EndpointId $endpointId -RunPodToken `$env:RUNPOD_API_KEY" -ForegroundColor Green
