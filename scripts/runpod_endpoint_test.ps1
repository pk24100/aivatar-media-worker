# RunPod Endpoint Test Script
# Tests the SoulX-FlashHead Lite serverless endpoint
# Usage: .\runpod_endpoint_test.ps1 -EndpointId "your-endpoint-id" -RunPodToken "your-api-key"

param(
    [Parameter(Mandatory=$false)]
    [string]$EndpointId = "<YOUR_ENDPOINT_ID>",
    
    [Parameter(Mandatory=$false)]
    [string]$RunPodToken = "<YOUR_RUNPOD_API_KEY>",
    
    [Parameter(Mandatory=$false)]
    [string]$TestMode = "sync",  # "sync" or "async"
    
    [Parameter(Mandatory=$false)]
    [string]$AudioUrl = "https://example.com/test-audio.wav",
    
    [Parameter(Mandatory=$false)]
    [string]$ImageUrl = "https://example.com/test-avatar.png",
    
    [Parameter(Mandatory=$false)]
    [string]$RoomName = "test_room_123"
    ,
    [Parameter(Mandatory=$false)]
    [string]$SessionId = "test-session-123"
    ,
    [Parameter(Mandatory=$false)]
    [string]$IngestionMethod = "websocket"
    ,
    [Parameter(Mandatory=$false)]
    [string]$LivekitToken = "test-livekit-token"
    ,
    [Parameter(Mandatory=$false)]
    [string]$LivekitUrl = "wss://example.livekit.cloud"
)

$Headers = @{
    "Authorization" = "Bearer $RunPodToken"
    "Content-Type" = "application/json"
}

Write-Host "=== RunPod Endpoint Test ===" -ForegroundColor Cyan
Write-Host "Endpoint ID: $EndpointId" -ForegroundColor White
Write-Host "Test Mode: $TestMode" -ForegroundColor White
Write-Host "Ingestion Method: $IngestionMethod" -ForegroundColor White
Write-Host ""

# Test 1: Health Check (GET endpoint info)
Write-Host "Test 1: Health Check..." -ForegroundColor Yellow
try {
    $healthResponse = Invoke-RestMethod -Method Get -Uri "https://rest.runpod.io/v1/endpoints/$EndpointId" -Headers $Headers
    Write-Host "  Endpoint Name: $($healthResponse.name)" -ForegroundColor Green
    Write-Host "  Workers: $($healthResponse.workersMin) - $($healthResponse.workersMax)" -ForegroundColor Green
    Write-Host "  Status: Healthy" -ForegroundColor Green
} catch {
    Write-Error "Health check failed: $_"
    exit 1
}

Write-Host ""

# Test 2: Streaming Inference Test
Write-Host "Test 2: Streaming Inference Test..." -ForegroundColor Yellow
$StreamingPayload = @{
    input = @{
        mode = "streaming"
        streaming = $true
        roomName = $RoomName
        sessionId = $SessionId
        livekitToken = $LivekitToken
        customLivekitUrl = $LivekitUrl
        ingestionMethod = $IngestionMethod
        sourceImage = $ImageUrl
        ingestionToken = "smoke-test-token"
    }
}

try {
    if ($TestMode -eq "sync") {
        Write-Host "  Sending sync request..." -ForegroundColor Gray
        $response = Invoke-RestMethod -Method Post -Uri "https://api.runpod.ai/v2/$EndpointId/run" -Headers $Headers -Body ($StreamingPayload | ConvertTo-Json -Depth 10)
        Write-Host "  Job ID: $($response.id)" -ForegroundColor Green
        Write-Host "  Status: $($response.status)" -ForegroundColor Green
        
        # Poll for results
        Write-Host "  Polling for results..." -ForegroundColor Gray
        $maxAttempts = 30
        $attempt = 0
        
        while ($attempt -lt $maxAttempts) {
            Start-Sleep -Seconds 2
            $statusResponse = Invoke-RestMethod -Method Get -Uri "https://api.runpod.ai/v2/$EndpointId/status/$($response.id)" -Headers $Headers
            
            if ($statusResponse.status -eq "COMPLETED") {
                Write-Host "  Job completed!" -ForegroundColor Green
                Write-Host "  Output: $($statusResponse.output | ConvertTo-Json -Depth 5)" -ForegroundColor Green
                break
            } elseif ($statusResponse.status -eq "FAILED") {
                Write-Error "Job failed: $($statusResponse.error)"
                break
            }
            
            $attempt++
            Write-Host "  Status: $($statusResponse.status) (attempt $attempt/$maxAttempts)" -ForegroundColor Gray
        }
    } else {
        Write-Host "  Sending async request..." -ForegroundColor Gray
        $response = Invoke-RestMethod -Method Post -Uri "https://api.runpod.ai/v2/$EndpointId/run" -Headers $Headers -Body ($StreamingPayload | ConvertTo-Json -Depth 10)
        Write-Host "  Job ID: $($response.id)" -ForegroundColor Green
        Write-Host "  Status: $($response.status)" -ForegroundColor Green
        Write-Host "  Check status with: GET /v2/$EndpointId/status/$($response.id)" -ForegroundColor Yellow
    }
} catch {
    Write-Error "Streaming test failed: $_"
}

Write-Host ""

Write-Host "=== Tests Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "API Endpoint: https://$EndpointId.api.runpod.ai/sessions/start" -ForegroundColor White
Write-Host "Expected websocket ingestion URL: wss://$EndpointId.api.runpod.ai/ws/$SessionId?token=smoke-test-token" -ForegroundColor White
Write-Host ""
Write-Host "Example curl command:" -ForegroundColor White
Write-Host @"
curl -X POST "https://api.runpod.ai/v2/$EndpointId/run" `
  -H "Authorization: Bearer $RunPodToken" `
  -H "Content-Type: application/json" `
  -d '{"input":{"mode":"streaming","streaming":true,"roomName":"$RoomName","sessionId":"$SessionId","livekitToken":"$LivekitToken","customLivekitUrl":"$LivekitUrl","ingestionMethod":"$IngestionMethod","sourceImage":"$ImageUrl","ingestionToken":"smoke-test-token"}}'
"@ -ForegroundColor Green
