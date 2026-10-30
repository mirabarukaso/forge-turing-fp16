param(
    [string]$ForgeRoot,
    [int]$DeviceId = -1,
    [switch]$UseUv,
    [switch]$Requirements
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
    Write-Error "Pass -ForgeRoot with the Forge Neo directory. Auto-detect only works when this extension lives in extensions\forge-turing-fp16."
}

$ForgeRoot = (Resolve-Path $ForgeRoot).Path
$VenvPython = Join-Path $ForgeRoot "venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    $VenvDir = Join-Path $ForgeRoot "venv"
    if ($UseUv) {
        $Uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
        if (-not $Uv) {
            Write-Error "uv was requested but was not found on PATH."
        }
        Write-Host "Creating uv venv at $VenvDir (no pip module; later installs use uv pip --python)"
        & $Uv venv --python python $VenvDir
    } else {
        $BasePython = (Get-Command python -ErrorAction SilentlyContinue).Source
        if (-not $BasePython) {
            Write-Error "python was not found on PATH, and venv\Scripts\python.exe does not exist."
        }
        Write-Host "Creating venv at $VenvDir"
        & $BasePython -m venv $VenvDir
    }
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$Helper = Join-Path $InstallerDir "turing_install.py"
$Arguments = @($Helper, "--forge-root", $ForgeRoot)
if ($UseUv) {
    $Arguments += "--uv"
}
if ($Requirements) {
    $Arguments += "--requirements"
}
if ($DeviceId -ge 0) {
    $Arguments += @("--device-id", "$DeviceId")
}

& $VenvPython @Arguments
exit $LASTEXITCODE
