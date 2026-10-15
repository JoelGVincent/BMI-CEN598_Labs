# request_prediction.ps1
# A minimal "base station" for Posture_Inference.ino: sends a sensor
# selection (1=accel, 2=gyro, 3=mag) over serial and prints whatever the
# board replies with. This is the required "interface that initiates
# real-time prediction" -- the board itself never predicts unless asked.
#
# Usage:
#   .\request_prediction.ps1 -Port COM4 -Sensor 1
#   .\request_prediction.ps1 -Port COM4 -Sensor 2
#   .\request_prediction.ps1 -Port COM4 -Sensor 3
#
# Board must already be running Posture_Inference.ino, and the Arduino IDE
# Serial Monitor must be closed (only one program can hold the port).

param(
    [Parameter(Mandatory=$true)][string]$Port,
    [Parameter(Mandatory=$true)][ValidateSet(1, 2, 3)][int]$Sensor,
    [int]$Baud = 115200,
    [int]$TimeoutSec = 10
)

$sensorNames = @{ 1 = "accelerometer"; 2 = "gyroscope"; 3 = "magnetometer" }

$sp = New-Object System.IO.Ports.SerialPort $Port, $Baud, ([System.IO.Ports.Parity]::None), 8, ([System.IO.Ports.StopBits]::One)
$sp.ReadTimeout = 300
$sp.NewLine = "`n"

$sp.Open()
$sp.DtrEnable = $true
$sp.RtsEnable = $true

# Drain any boot banner / stale output first, same reasoning as the data
# collection tooling: opening the port often resets these boards.
Write-Host "(waiting for board to boot / settle...)"
$bootDeadline = (Get-Date).AddSeconds(5)
$sawAnything = $false
while ((Get-Date) -lt $bootDeadline) {
    try {
        $line = $sp.ReadLine()
        $sawAnything = $true
    } catch [System.TimeoutException] {
        if ($sawAnything) { break }
    }
}
$sp.DiscardInBuffer()
$sp.ReadTimeout = $TimeoutSec * 1000

Write-Host "Requesting a prediction using the $($sensorNames[$Sensor])..."
$sp.Write("$Sensor")

try {
    $response = $sp.ReadLine().TrimEnd("`r")
    Write-Host "Board replied: $response"
} catch [System.TimeoutException] {
    Write-Warning "No response within $TimeoutSec s. Check the board is running Posture_Inference.ino and its window buffer has had ~3s to fill since boot."
}

$sp.Close()
