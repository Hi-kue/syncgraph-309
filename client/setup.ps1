$ErrorActionPreference = "Stop"

if (-not (Get-Command "python")) {
    Write-Output "Python is not installed, please install python and try again..."
    exit 1
}

$pythonVersion = python --version
$pyversion = py --version

Write-Output "Python Version: $pythonVersion"
Write-Output "Py Version: $pyversion"

if (-not (Get-Command "pipx")) {
    Write-Output "Pipx is not installed, Installing Pipx..."
    python -m pip install --user pipx
    python -m pipx ensurepath


    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","User") + ";" + [System.Environment]::GetEnvironmentVariable("Path","Machine")
} else {
    Write-Output "Pipx is already installed, skipping installation..."

}

if (-not (Get-Command "poetry")) {
    Write-Output "Poetry is not installed, Installing Poetry..."
    pipx install poetry
} else  {
    Write-Output "Poetry is already installed, skipping installation..."

}

if (-not (Test-Path "pyproject.toml")) {
    Write-Output "No pyproject.toml file found, creating one..."
    poetry init
    poetry shell
    Write-Output "The shell created by Poetry should be your active interpreter."

    poetry add $(cat requirements.txt)

} elseif ((Test-Path "pyproject.toml") -and (Test-Path "requirements.txt")) {
    Write-Output "Requirements.txt file found, installing dependencies..."
    poetry add $(cat requirements.txt)

} else {
    Write-Output "Requirements.txt file not found, skipping dep installation..."

}

Write-Output "Setup is now complete!"
