# SerialTrialLib.ps1
# Shared function for capturing one IMU_Data_Logger.ino trial straight to a
# CSV file over the COM port, using .NET's built-in serial port support --
# no external tools/libraries needed.
#
# Protocol (matches IMU_Data_Logger.ino): send 's' to start a bounded
# recording; the board streams a header line, then data rows, then an
# "# === END trial" marker line and returns to idle. We keep everything
# except '#'-prefixed lines and write it straight to $OutFile.

function Read-Trial {
    param(
        [Parameter(Mandatory=$true)][string]$Port,
        [Parameter(Mandatory=$true)][string]$OutFile,
        [int]$Baud = 115200,
        [int]$TimeoutSec = 30
    )

    $sp = New-Object System.IO.Ports.SerialPort $Port, $Baud, ([System.IO.Ports.Parity]::None), 8, ([System.IO.Ports.StopBits]::One)
    $sp.ReadTimeout = 300
    $sp.NewLine = "`n"

    $sp.Open()
    $sp.DtrEnable = $true
    $sp.RtsEnable = $true

    # Opening the port commonly resets these boards, which then take a few
    # seconds to reboot, re-init the IMU, and print their boot banner.
    # Instead of guessing a fixed delay, actively drain and show whatever
    # boot output shows up for a few seconds before sending 's'.
    Write-Host "  (waiting for board to boot / settle...)"
    $bootDeadline = (Get-Date).AddSeconds(5)
    $sawAnything = $false
    while ((Get-Date) -lt $bootDeadline) {
        try {
            $bootLine = $sp.ReadLine()
            $sawAnything = $true
            Write-Host "  (boot: $($bootLine.TrimEnd([char]13)))"
        } catch [System.TimeoutException] {
            if ($sawAnything) { break }  # boot text stopped arriving -> board is idle, move on
        }
    }
    $sp.DiscardInBuffer()
    $sp.ReadTimeout = 3000

    $sp.WriteLine("s")

    $rows = New-Object System.Collections.Generic.List[string]
    $rawLinesSeen = 0
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $gotEnd = $false

    while ((Get-Date) -lt $deadline) {
        try {
            $line = $sp.ReadLine()
        } catch [System.TimeoutException] {
            continue
        }
        $line = $line.TrimEnd("`r")
        $rawLinesSeen++
        if ($rawLinesSeen -le 3) {
            Write-Host "  (received: $line)"
        }

        if ($line -match "END trial") {
            $gotEnd = $true
            break
        }
        if ($line.StartsWith("#") -or $line.Trim().Length -eq 0) {
            continue
        }
        $rows.Add($line)
    }

    $sp.Close()

    if ($rawLinesSeen -eq 0) {
        Write-Warning "Received NOTHING from $Port in $TimeoutSec s. Most likely cause: the board isn't running IMU_Data_Logger.ino (wrong/old sketch uploaded), or the Arduino IDE Serial Monitor still has the port open. Check both and try again."
    } elseif (-not $gotEnd) {
        Write-Warning "Saw $rawLinesSeen line(s) but never the END marker before the $TimeoutSec s timeout -- trial may be incomplete."
    }

    # Always write the file (even if empty), via -Value so an empty $rows
    # doesn't silently skip file creation -- "nothing happened" should be
    # visible as an empty file, not a missing one.
    Set-Content -Path $OutFile -Value $rows -Encoding utf8
    $dataRowCount = [Math]::Max(0, $rows.Count - 1)  # first row is the CSV header
    Write-Host "  -> wrote $dataRowCount data rows to $OutFile"
    return $gotEnd
}
