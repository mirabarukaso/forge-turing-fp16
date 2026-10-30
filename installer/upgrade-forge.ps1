param(
    [string]$ForgeRoot,
    [switch]$Pull,
    [switch]$Apply,
    [switch]$UseUv,
    [int]$DeviceId = -1
)

$ErrorActionPreference = "Stop"

$InstallerDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ExtensionRoot = Split-Path -Parent $InstallerDir

if (-not $ForgeRoot) {
    $maybe = Split-Path -Parent (Split-Path -Parent $ExtensionRoot)
    if (Test-Path (Join-Path $maybe "webui.bat")) {
        $ForgeRoot = $maybe
    }
}

if (-not $ForgeRoot) {
    Write-Error "Pass -ForgeRoot. Auto-detect only works when this extension lives in extensions\forge-turing-fp16."
}

$ForgeRoot = (Resolve-Path $ForgeRoot).Path
$VenvPython = Join-Path $ForgeRoot "venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Error "Missing $VenvPython. Create the virtualenv with install-turing.ps1 before upgrading."
}

if ($Pull) {
    Write-Host "git pull --ff-only in $ForgeRoot"
    Push-Location $ForgeRoot
    & git pull --ff-only
    $pull = $LASTEXITCODE
    Pop-Location
    if ($pull -ne 0) {
        exit $pull
    }
}

$Plan = Join-Path $InstallerDir "forge_plan.py"
$Arguments = @($Plan, "--forge-root", $ForgeRoot)
if ($UseUv) {
    $Arguments += "--uv"
}
if ($Apply) {
    $Arguments += "--apply"
}
if ($DeviceId -ge 0) {
    $Arguments += @("--device-id", "$DeviceId")
}

& $VenvPython @Arguments
exit $LASTEXITCODE
