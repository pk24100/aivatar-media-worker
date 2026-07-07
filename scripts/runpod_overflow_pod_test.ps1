# RunPod Overflow Pod Smoke Test
# Usage: .\runpod_overflow_pod_test.ps1 -PodApiBase "https://your-pod-host" [-ApiToken "token"]

param(
    [Parameter(Mandatory=$false)]
    [string]$PodApiBase = "http://127.0.0.1:8000",

    [Parameter(Mandatory=$false)]
    [string]$ApiToken = "",

    [Parameter(Mandatory=$false)]
    [string]$RoomName = "overflow_room_123",

    [Parameter(Mandatory=$false)]
    [string]$SessionId = "overflow-session-123",

    [Parameter(Mandatory=$false)]
    [string]$ImageUrl = "https://example.com/test-avatar.png",

    [Parameter(Mandatory=$false)]
    [string]$LivekitToken = "test-livekit-token",

    [Parameter(Mandatory=$false)]
    [string]$LivekitUrl = "wss://example.livekit.cloud"
)

$PodApiBase = $PodApiBase.TrimEnd('/')
$Headers = @{
    "Content-Type" = "application/json"
}

if ($ApiToken) {
    $Headers["Authorization"] = "Bearer $ApiToken"
}

Write-Host "=== Overflow Pod Smoke Test ===" -ForegroundColor Cyan
Write-Host "Pod API Base: $PodApiBase" -ForegroundColor White
Write-Host "Session ID: $SessionId" -ForegroundColor White
Write-Host ""

Write-Host "Test 1: Health Check..." -ForegroundColor Yellow
try {
    $healthResponse = Invoke-RestMethod -Method Get -Uri "$PodApiBase/healthz" -Headers $Headers
    Write-Host "  Status: $($healthResponse.status)" -ForegroundColor Green
    Write-Host "  Active Sessions: $($healthResponse.activeSessions)" -ForegroundColor Green
} catch {
    Write-Error "Health check failed: $_"
    exit 1
}

Write-Host ""
Write-Host "Test 2: Readiness Check..." -ForegroundColor Yellow
try {
    $readyResponse = Invoke-RestMethod -Method Get -Uri "$PodApiBase/readyz" -Headers $Headers
    Write-Host "  Ready: $($readyResponse.ready)" -ForegroundColor Green
    Write-Host "  Available Pipelines: $($readyResponse.availablePipelines)" -ForegroundColor Green
} catch {
    Write-Error "Readiness check failed: $_"
    exit 1
}

Write-Host ""
Write-Host "Test 3: Start Streaming Session..." -ForegroundColor Yellow
$StartPayload = @{
    mode = "streaming"
    streaming = $true
    roomName = $RoomName
    sessionId = $SessionId
    livekitToken = $LivekitToken
    customLivekitUrl = $LivekitUrl
    ingestionMethod = "websocket"
    sourceImage = $ImageUrl
    ingestionToken = "smoke-test-token"
} | ConvertTo-Json -Depth 10

try {
    $startResponse = Invoke-RestMethod -Method Post -Uri "$PodApiBase/sessions/start" -Headers $Headers -Body $StartPayload
    Write-Host "  Start Status: $($startResponse.status)" -ForegroundColor Green
    Write-Host "  Job ID: $($startResponse.jobId)" -ForegroundColor Green
    Write-Host "  Expected WebSocket URL: ws://$(([System.Uri]$PodApiBase).Host):8765/$SessionId (auth via Sec-WebSocket-Protocol: aivatar.<jwt>)" -ForegroundColor White
} catch {
    Write-Error "Start session failed: $_"
    exit 1
}

Write-Host ""
Write-Host "Test 4: End Streaming Session..." -ForegroundColor Yellow
try {
    $endResponse = Invoke-RestMethod -Method Post -Uri "$PodApiBase/sessions/$SessionId/end" -Headers $Headers
    Write-Host "  End Status: $($endResponse.status)" -ForegroundColor Green
} catch {
    Write-Error "End session failed: $_"
    exit 1
}

Write-Host ""
Write-Host "=== Smoke Test Complete ===" -ForegroundColor Cyan
