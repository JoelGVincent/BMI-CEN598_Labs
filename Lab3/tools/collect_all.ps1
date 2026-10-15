# collect_all.ps1
# Walks through every trial in the Lab 3 data collection checklist,
# prompting you to reposition the board between each one, then captures
# each trial straight to its CSV file automatically (no copy/paste).
#
# Usage:
#   .\collect_all.ps1 -Port COM4
#
# Safe to stop (Ctrl+C) and re-run later -- it skips any file that already
# exists, so it picks up where you left off.

param(
    [Parameter(Mandatory=$true)][string]$Port,
    [string]$OutDir = "$PSScriptRoot\..\IMU_Data_Collection",
    [int]$Baud = 115200,
    [int]$TimeoutSec = 30
)

. "$PSScriptRoot\SerialTrialLib.ps1"

if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir | Out-Null
}

# Each entry: the instruction to show you, and the trial's output filename.
$trials = @(
    @{ Label = "Class 1: SUPINE, USB toward HEAD";  File = "supine-usbhead-trial1.csv" }
    @{ Label = "Class 1: SUPINE, USB toward HEAD";  File = "supine-usbhead-trial2.csv" }
    @{ Label = "Class 1: SUPINE, USB toward HEAD";  File = "supine-usbhead-trial3.csv" }
    @{ Label = "Class 1: SUPINE, USB toward FEET (band flipped)"; File = "supine-usbfeet-trial1.csv" }
    @{ Label = "Class 1: SUPINE, USB toward FEET (band flipped)"; File = "supine-usbfeet-trial2.csv" }
    @{ Label = "Class 1: SUPINE, USB toward FEET (band flipped)"; File = "supine-usbfeet-trial3.csv" }

    @{ Label = "Class 2: PRONE, USB toward HEAD";  File = "prone-usbhead-trial1.csv" }
    @{ Label = "Class 2: PRONE, USB toward HEAD";  File = "prone-usbhead-trial2.csv" }
    @{ Label = "Class 2: PRONE, USB toward HEAD";  File = "prone-usbhead-trial3.csv" }
    @{ Label = "Class 2: PRONE, USB toward FEET (band flipped)"; File = "prone-usbfeet-trial1.csv" }
    @{ Label = "Class 2: PRONE, USB toward FEET (band flipped)"; File = "prone-usbfeet-trial2.csv" }
    @{ Label = "Class 2: PRONE, USB toward FEET (band flipped)"; File = "prone-usbfeet-trial3.csv" }

    @{ Label = "Class 3: SIDE, LEFT";  File = "side-left-trial1.csv" }
    @{ Label = "Class 3: SIDE, LEFT";  File = "side-left-trial2.csv" }
    @{ Label = "Class 3: SIDE, LEFT";  File = "side-left-trial3.csv" }
    @{ Label = "Class 3: SIDE, RIGHT"; File = "side-right-trial1.csv" }
    @{ Label = "Class 3: SIDE, RIGHT"; File = "side-right-trial2.csv" }
    @{ Label = "Class 3: SIDE, RIGHT"; File = "side-right-trial3.csv" }

    @{ Label = "Class 4: SITTING, USB toward HEAD (chin)";  File = "sitting-usbhead-trial1.csv" }
    @{ Label = "Class 4: SITTING, USB toward HEAD (chin)";  File = "sitting-usbhead-trial2.csv" }
    @{ Label = "Class 4: SITTING, USB toward HEAD (chin)";  File = "sitting-usbhead-trial3.csv" }
    @{ Label = "Class 4: SITTING, USB toward FEET (waist)"; File = "sitting-usbfeet-trial1.csv" }
    @{ Label = "Class 4: SITTING, USB toward FEET (waist)"; File = "sitting-usbfeet-trial2.csv" }
    @{ Label = "Class 4: SITTING, USB toward FEET (waist)"; File = "sitting-usbfeet-trial3.csv" }

    @{ Label = "Class 5: UNKNOWN, standing upright, USB toward HEAD"; File = "standing-usbhead-trial1.csv" }
    @{ Label = "Class 5: UNKNOWN, standing upright, USB toward HEAD"; File = "standing-usbhead-trial2.csv" }
    @{ Label = "Class 5: UNKNOWN, standing upright, USB toward HEAD"; File = "standing-usbhead-trial3.csv" }
    @{ Label = "Class 5: UNKNOWN, ~45deg tilt between supine and side (recreate the ambiguous boundary case)"; File = "unknown-random1.csv" }
    @{ Label = "Class 5: UNKNOWN, compound diagonal tilt (~30-40deg in two directions at once, distinct from random1)"; File = "unknown-random2.csv" }
)

$total = $trials.Count
$i = 0
foreach ($t in $trials) {
    $i++
    $outPath = Join-Path $OutDir $t.File

    if (Test-Path $outPath) {
        Write-Host "[$i/$total] SKIP (already exists): $($t.File)"
        continue
    }

    Write-Host ""
    Write-Host "[$i/$total] $($t.Label)"
    Write-Host "  -> will save to: $($t.File)"
    Read-Host "  Position the board now, then press Enter to start the ~20s recording"

    $ok = Read-Trial -Port $Port -OutFile $outPath -Baud $Baud -TimeoutSec $TimeoutSec
    if (-not $ok) {
        Write-Host "  Warning: trial may be incomplete. Re-run this script to retry it (it will skip the ones already saved)." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "All done. $total trials processed into $OutDir"
