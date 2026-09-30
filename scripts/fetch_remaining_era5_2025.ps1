<#
Downloads the five 2025 monthly archives that are not already local and are
not the separately running April transfer. Each download lands as .partial and
is moved into the raw-data directory only after curl reports success.

Run hidden by the setup task or manually with:
  powershell -ExecutionPolicy Bypass -File scripts/fetch_remaining_era5_2025.ps1
#>

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$destinationRoot = Join-Path $root 'data\raw\era5_2025_full_source'
$logPath = Join-Path $root 'runtime\era5_2025_drive_download.log'
New-Item -ItemType Directory -Force -Path $destinationRoot, (Split-Path -Parent $logPath) | Out-Null

$downloads = @(
    @{ Name = 'may_2025.zip'; Id = '1_Z-LgQh1tR1RXpzs1x7J6S9c7Snpir3w' },
    @{ Name = 'june_2025.zip'; Id = '1h22ySSa0NtlwAvtI66IH0Ps4Rnx6t6fY' },
    @{ Name = 'july_2025.zip'; Id = '1LZ9jayPWoJazg8t8HReKnt29EBoOGFQB' },
    @{ Name = 'september_2025.zip'; Id = '1WrHA-TS3bfL0y5WdAzg2kn6QWmHz6dP2' },
    @{ Name = 'october_2025.zip'; Id = '1LT2MKsc-aP5RacDor-p-Ij-6lOeeHVVB' }
)

foreach ($download in $downloads) {
    $destination = Join-Path $destinationRoot $download.Name
    if (Test-Path -LiteralPath $destination) {
        "$(Get-Date -Format o) already present: $($download.Name)" | Add-Content -Path $logPath
        continue
    }
    $partial = "$destination.partial"
    $uri = "https://drive.usercontent.google.com/download?id=$($download.Id)&export=download&confirm=t"
    "$(Get-Date -Format o) downloading: $($download.Name)" | Add-Content -Path $logPath
    & curl.exe --location --fail --retry 3 --retry-all-errors --output $partial $uri 2>&1 | Add-Content -Path $logPath
    if ($LASTEXITCODE -ne 0) {
        throw "Drive download failed for $($download.Name); partial file is retained for inspection."
    }
    & tar.exe -tf $partial *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Drive returned a non-ZIP response for $($download.Name); partial file is retained for inspection."
    }
    Move-Item -LiteralPath $partial -Destination $destination
    "$(Get-Date -Format o) complete: $($download.Name)" | Add-Content -Path $logPath
}
