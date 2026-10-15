# request_prediction.ps1
# A minimal "base station" for KWS_Inference.ino: sends a single trigger
# character over serial and prints whatever the board replies with. This
# is the required "interface that initiates real-time prediction" -- the
# board itself never predicts unless asked. Simpler than Lab 4's version
# since there is only one microphone here, no sensor selection needed.
#
# Usage:
#   .\request_prediction.ps1 -Port COM4
#
# Board must already be running KWS_Inference.ino, and the Arduino IDE
# Serial Monitor must be closed (only one program can hold the port).

param(
    [Parameter(Mandatory=$true)][string]$Port,
    [int]$Baud = 115200,
    [int]$TimeoutSec = 10
)

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

# The board resets when this script opens the port, and its audio ring
# buffer needs a full second of fresh samples after that before a
# prediction is possible. Wait that out first, retrying quietly rather
# than making the user re-run this.
$maxWarmupAttempts = 5
for ($attempt = 1; $attempt -le $maxWarmupAttempts; $attempt++) {
    $sp.Write("p")
    try {
        $response = $sp.ReadLine().TrimEnd("`r")
    } catch [System.TimeoutException] {
        Write-Warning "No response within $TimeoutSec s. Check the board is running KWS_Inference.ino."
        $sp.Close()
        return
    }
    if ($response -notmatch "buffer not full yet") { break }
    Start-Sleep -Milliseconds 400
}

# A prediction covers whatever's in the most recent 1-second window at the
# moment 'p' is sent, so the word needs to be spoken right up against that
# moment, not back when the script started. Give a short countdown, then
# send the real request right as the "speak now" window ends.
Write-Host "Get ready..."
Start-Sleep -Milliseconds 1000
Write-Host "Speak now!"
Start-Sleep -Milliseconds 1000

Write-Host "Requesting a prediction..."
$sp.Write("p")
try {
    $response = $sp.ReadLine().TrimEnd("`r")
} catch [System.TimeoutException] {
    Write-Warning "No response within $TimeoutSec s. Check the board is running KWS_Inference.ino."
    $sp.Close()
    return
}

Write-Host "Board replied: $response"

$sp.Close()
