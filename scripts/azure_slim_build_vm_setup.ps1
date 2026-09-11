# Azure Slim Docker Builder Setup Script
# This script automates the Azure VM setup and remote Linux preparation steps for building
# the slim AiVatar media worker image (wav2vec2 baked, FlashHead loaded from the Modal baked image).
# Usage: Fill in the configuration values below and run: .\azure_slim_build_vm_setup.ps1

# ===============================
# CONFIGURATION - Fill these in
# ===============================
$Location             = "australiaeast"
$ResourceGroup        = "aivatar-slim-builder-rg"
$VmName               = "aivatar-slim-builder-vm"
$NsgName              = "aivatar-slim-builder-nsg"
$PublicIpName         = "aivatar-slim-builder-pip"
$AdminUser            = "azureuser"
$VmSize               = "Standard_D2s_v3"
$VmImage              = "Ubuntu2204"
$OsDiskGb             = 64
$StorageSku           = "Premium_LRS"
$SwapSizeGb           = 4
$SshPrivateKeyPath    = "$HOME\.ssh\id_rsa"

$RepoUrl              = "https://github.com/pk24100/aivatar-media-worker.git"
$RepoBranch           = "test2"
$RepoDirectoryName    = "aivatar-media-worker"
$WorkRoot             = "/home/$AdminUser/work"
$ProjectSubdirectory  = "aivatar-media-worker"
$DockerImageTag       = "pk24100/aivatar-worker:flashhead-lite-v3"

# ===============================
# DO NOT EDIT BELOW THIS LINE
# ===============================
$ErrorActionPreference = "Stop"

function Require-Command {
    param([string]$CommandName)
    if (-not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        throw "Required command '$CommandName' was not found in PATH."
    }
}

function Run-Az {
    param([string[]]$Arguments)
    & az @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Azure CLI command failed: az $($Arguments -join ' ')"
    }
}

function Wait-ForSsh {
    param(
        [string]$HostName,
        [string]$UserName,
        [string]$KeyPath,
        [int]$TimeoutSeconds = 600
    )

    $start = Get-Date
    while (((Get-Date) - $start).TotalSeconds -lt $TimeoutSeconds) {
        $sshArgs = "-i `"$KeyPath`" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 `"$UserName@$HostName`" `"echo ssh-ready`""
        $processInfo = New-Object System.Diagnostics.ProcessStartInfo
        $processInfo.FileName = "ssh"
        $processInfo.Arguments = $sshArgs
        $processInfo.UseShellExecute = $false
        $processInfo.RedirectStandardOutput = $true
        $processInfo.RedirectStandardError = $true
        $processInfo.CreateNoWindow = $true

        $process = New-Object System.Diagnostics.Process
        $process.StartInfo = $processInfo
        [void]$process.Start()
        $process.WaitForExit()

        if ($process.ExitCode -eq 0) {
            return
        }
        Start-Sleep -Seconds 10
    }

    throw "Timed out waiting for SSH on $HostName"
}

function Invoke-RemoteScript {
    param(
        [string]$HostName,
        [string]$UserName,
        [string]$KeyPath,
        [string]$ScriptContent
    )

    $tempFile = [System.IO.Path]::GetTempFileName()
    try {
        Set-Content -Path $tempFile -Value $ScriptContent -NoNewline
        Get-Content -Path $tempFile -Raw | ssh -i $KeyPath -o StrictHostKeyChecking=accept-new "$UserName@$HostName" "bash -s"
        if ($LASTEXITCODE -ne 0) {
            throw "Remote setup script failed on $HostName"
        }
    }
    finally {
        Remove-Item -Path $tempFile -ErrorAction SilentlyContinue
    }
}

function Invoke-RemoteInteractiveScript {
    param(
        [string]$HostName,
        [string]$UserName,
        [string]$KeyPath,
        [string]$ScriptContent
    )

    $tempFile = [System.IO.Path]::GetTempFileName()
    $remoteTempFile = "/tmp/$([System.IO.Path]::GetFileName($tempFile)).sh"
    try {
        Set-Content -Path $tempFile -Value $ScriptContent -NoNewline
        & scp -i $KeyPath -o StrictHostKeyChecking=accept-new $tempFile "$UserName@${HostName}:$remoteTempFile"
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to upload remote interactive script to $HostName"
        }

        $remoteCommand = "chmod +x '$remoteTempFile'; bash '$remoteTempFile'; exitCode=`$?; rm -f '$remoteTempFile'; exit `$exitCode"
        & ssh -tt -i $KeyPath -o StrictHostKeyChecking=accept-new "$UserName@$HostName" $remoteCommand
        if ($LASTEXITCODE -ne 0) {
            throw "Remote interactive script failed on $HostName"
        }
    }
    finally {
        Remove-Item -Path $tempFile -ErrorAction SilentlyContinue
    }
}

Require-Command -CommandName "az"
Require-Command -CommandName "ssh"
Require-Command -CommandName "scp"

if (-not (Test-Path $SshPrivateKeyPath)) {
    throw "SSH private key not found at '$SshPrivateKeyPath'. Update `$SshPrivateKeyPath or generate SSH keys before running this script."
}

if ($RepoUrl -eq "https://github.com/<owner>/<repo>.git" -or $RepoDirectoryName -eq "<repo>") {
    throw "Set both `$RepoUrl and `$RepoDirectoryName before running this script."
}

Write-Host "=== Azure Slim Docker Builder Setup ===" -ForegroundColor Cyan
Write-Host ""

Write-Host "Step 1: Creating resource group '$ResourceGroup' in '$Location'..." -ForegroundColor Yellow
Run-Az -Arguments @("group", "create", "--name", $ResourceGroup, "--location", $Location, "--output", "table")

Write-Host "Step 2: Creating VM '$VmName'..." -ForegroundColor Yellow
Run-Az -Arguments @(
    "vm", "create",
    "--resource-group", $ResourceGroup,
    "--name", $VmName,
    "--location", $Location,
    "--image", $VmImage,
    "--size", $VmSize,
    "--admin-username", $AdminUser,
    "--generate-ssh-keys",
    "--public-ip-sku", "Standard",
    "--public-ip-address", $PublicIpName,
    "--nsg", $NsgName,
    "--os-disk-size-gb", "$OsDiskGb",
    "--storage-sku", $StorageSku,
    "--output", "table"
)

Write-Host "Step 3: Ensuring inbound SSH access is allowed..." -ForegroundColor Yellow
Run-Az -Arguments @(
    "network", "nsg", "rule", "create",
    "--resource-group", $ResourceGroup,
    "--nsg-name", $NsgName,
    "--name", "Allow-SSH-All",
    "--access", "Allow",
    "--protocol", "Tcp",
    "--direction", "Inbound",
    "--priority", "120",
    "--source-address-prefix", "Internet",
    "--source-port-range", "*",
    "--destination-address-prefix", "VirtualNetwork",
    "--destination-port-range", "22",
    "--output", "table"
)

Write-Host "Step 4: Fetching public IP address..." -ForegroundColor Yellow
$VmPublicIp = az vm show --resource-group $ResourceGroup --name $VmName -d --query publicIps -o tsv
if (-not $VmPublicIp) {
    throw "Could not determine VM public IP address."
}
Write-Host "  Public IP: $VmPublicIp" -ForegroundColor Green

Write-Host "Step 5: Waiting for SSH to become available..." -ForegroundColor Yellow
Wait-ForSsh -HostName $VmPublicIp -UserName $AdminUser -KeyPath $SshPrivateKeyPath
Write-Host "  SSH is ready." -ForegroundColor Green

Write-Host "Step 6-13: Running remote Linux setup, Docker Buildx setup, and wav2vec2-only model download..." -ForegroundColor Yellow
$RemoteScript = @"
set -euo pipefail

sudo apt-get update
sudo apt-get install -y git git-lfs python3-pip ca-certificates curl

if [ ! -f /swapfile ]; then
  sudo fallocate -l ${SwapSizeGb}G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
fi
sudo swapon /swapfile || true
free -h

git lfs install
sudo mkdir -p "$WorkRoot"
sudo chown -R ${AdminUser}:${AdminUser} "$WorkRoot"

cd "$WorkRoot"
if [ -d "$RepoDirectoryName" ]; then
  rm -rf "$RepoDirectoryName"
fi

git clone -b "$RepoBranch" "$RepoUrl"
cd "$RepoDirectoryName"

test -f SoulX-FlashHead/flash_head/ltx_video/models/autoencoders/causal_video_autoencoder.py && echo "FlashHead source package OK"

for pkg in docker.io docker-doc docker-compose podman-docker containerd runc; do
  sudo apt-get remove -y "$pkg" || true
done

sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu jammy stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable docker || true
sudo systemctl start docker || true
sudo usermod -aG docker $AdminUser

sudo systemctl daemon-reload
sudo systemctl restart containerd
sudo systemctl restart docker
sudo systemctl status docker.service --no-pager -l

sg docker -c 'docker version'
sg docker -c 'docker buildx version'
sg docker -c 'docker buildx ls'

if sg docker -c 'docker buildx inspect aivatar-builder >/dev/null 2>&1'; then
  sg docker -c 'docker buildx use aivatar-builder'
else
  sg docker -c 'docker buildx create --name aivatar-builder --use'
fi
sg docker -c 'docker buildx inspect --bootstrap'

sudo python3 -m pip install -U "huggingface_hub[cli]"
chmod +x scripts/download_models.sh
./scripts/download_models.sh
if [ -d models/wav2vec2-base-960h ]; then
  du -sh models/wav2vec2-base-960h
fi

echo "REMOTE_PROJECT_DIR=$WorkRoot/$RepoDirectoryName"
echo "BUILDX_BUILDER=aivatar-builder"
"@
Invoke-RemoteScript -HostName $VmPublicIp -UserName $AdminUser -KeyPath $SshPrivateKeyPath -ScriptContent $RemoteScript

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting remote Docker login (device-code flow)..." -ForegroundColor Yellow
Write-Host "Complete the Docker device-code login in your browser when prompted." -ForegroundColor White

$BuildScript = @"
set -euo pipefail

cd "$WorkRoot/$RepoDirectoryName"

sg docker -c 'docker version'
sg docker -c 'docker buildx use aivatar-builder'
sg docker -c 'docker login'
"@
Invoke-RemoteInteractiveScript -HostName $VmPublicIp -UserName $AdminUser -KeyPath $SshPrivateKeyPath -ScriptContent $BuildScript

Write-Host ""
Write-Host "Docker login complete. Build and push the slim image MANUALLY over SSH:" -ForegroundColor Green
Write-Host ""
Write-Host "  ssh -i `"$SshPrivateKeyPath`" $AdminUser@$VmPublicIp" -ForegroundColor White
Write-Host "  cd $WorkRoot/$RepoDirectoryName" -ForegroundColor White
Write-Host "  sg docker -c 'docker buildx use aivatar-builder'" -ForegroundColor White
Write-Host "  sg docker -c 'docker buildx build --platform linux/amd64 -t $DockerImageTag --cache-from type=registry,ref=pk24100/aivatar-worker:buildcache --cache-to type=registry,ref=pk24100/aivatar-worker:buildcache,mode=max --push .'" -ForegroundColor White
Write-Host ""
Write-Host "Cleanup when finished:" -ForegroundColor White
Write-Host "  az group delete --name $ResourceGroup --yes --no-wait" -ForegroundColor Green
