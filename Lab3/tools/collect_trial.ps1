# collect_trial.ps1
# Capture a single trial straight to a CSV file -- no Serial Monitor, no
# copy/paste. Board must be running IMU_Data_Logger.ino and be idle
# (i.e. not mid-recording; if it is, wait for its current trial to finish).
#
# Usage:
#   .\collect_trial.ps1 -Port COM4 -OutFile ..\IMU_Data_Collection\supine-usbhead-trial1.csv

param(
    [Parameter(Mandatory=$true)][string]$Port,
    [Parameter(Mandatory=$true)][string]$OutFile,
    [int]$Baud = 115200,
    [int]$TimeoutSec = 30
)

. "$PSScriptRoot\SerialTrialLib.ps1"

Write-Host "Recording on $Port -> $OutFile ..."
$ok = Read-Trial -Port $Port -OutFile $OutFile -Baud $Baud -TimeoutSec $TimeoutSec
if ($ok) { Write-Host "Done." } else { Write-Host "Finished with a warning -- check the file." }
