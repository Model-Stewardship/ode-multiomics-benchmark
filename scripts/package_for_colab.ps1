# Package project for Colab upload
# Zips src/, experiments/, and key config files into a single archive
# Usage: .\scripts\package_for_colab.ps1
# Then upload the resulting .zip to Google Drive

$items = @('src', 'experiments', 'requirements.txt', 'pyproject.toml', 'environment.yml')
$zipFile = 'ode-multiomics-benchmark.zip'

Write-Host "Creating $zipFile..."
Compress-Archive -Path $items -DestinationPath $zipFile -Force

if (Test-Path $zipFile) {
    $size = (Get-Item $zipFile).Length / 1MB
    Write-Host "✓ Created $zipFile (${size:.1f} MB)"
    Write-Host "Next: Upload this file to Google Drive, then use it in the Colab notebook"
} else {
    Write-Host "✗ Failed to create archive"
    exit 1
}
