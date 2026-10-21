param(
  [Parameter(Mandatory=$true)][string]$Python,
  [Parameter(Mandatory=$true)][string]$BlenderDirectory,
  [switch]$SkipNative
)
$ErrorActionPreference='Stop'
$taskRepo=Split-Path -Parent $PSScriptRoot
$taskOriginalPath=$env:PATH
Push-Location $taskRepo
try {
  # Conda's ICU DLL can shadow Windows ICU and break Qt WebEngine. Scope the
  # clean PATH to this build only; no global environment is modified.
  $env:PATH=($env:PATH -split ';' | Where-Object { $_ -notmatch '(?i)anaconda|miniconda' }) -join ';'
  & $Python -c "import sys; assert 'conda' not in sys.base_prefix.lower(), 'Use standalone CPython 3.12 x64, not a Conda-based venv'"
  if ($LASTEXITCODE) { throw 'Standalone CPython required' }
  if (-not $SkipNative) { & $Python scripts/build_native_forest.py; if ($LASTEXITCODE) { throw 'Native build failed' } }
  # An unrelated PDF/image tool may expose another icuuc.dll on PATH. Qt on
  # Windows uses the OS ICU ABI; such a DLL must never enter the frozen root.
  $taskPythonDirectory=Split-Path -Parent (Resolve-Path -LiteralPath $Python).Path
  $env:PATH=@($taskPythonDirectory,(Join-Path $env:SystemRoot 'System32'),$env:SystemRoot) -join ';'
  $env:LIQUID_BLENDER_DIR=(Resolve-Path -LiteralPath $BlenderDirectory).Path
  & $Python -m PyInstaller --clean --noconfirm packaging/liquid_simulation.spec
  if ($LASTEXITCODE) { throw 'PyInstaller build failed' }
  Write-Host 'Portable EXE: dist/LiquidDepthSimulation/LiquidDepthSimulation.exe'
  Write-Host 'Copy the ENTIRE directory. CPU inference and CPU Cycles are the desktop defaults.'
} finally { $env:PATH=$taskOriginalPath; Pop-Location }
