# Windows PowerShell: download every file listed in a text file (one URL per line).
# 1. Open the UNB/CIC download page for CICIoT2023, copy the file links into urls.txt
# 2. Run:  powershell -ExecutionPolicy Bypass -File tools\download_ciciot2023.ps1 urls.txt data\raw\CICIoT2023
param([string]$List = "urls.txt", [string]$Out = "data\raw\CICIoT2023")
New-Item -ItemType Directory -Force -Path $Out | Out-Null
Get-Content $List | Where-Object { $_.Trim() -ne "" } | ForEach-Object {
    $url = $_.Trim()
    $name = [System.Uri]::UnescapeDataString(($url -split "/")[-1])
    $dest = Join-Path $Out $name
    if (Test-Path $dest) { Write-Host "skip $name"; return }
    Write-Host "get  $name"
    curl.exe -L --retry 5 -C - -o $dest $url
}
