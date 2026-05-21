$logFile = "C:\Users\CAU\AppData\Local\Temp\claude\d---RGB-LWIR---ver----\596bceb2-6671-4237-a339-d129fa6e6dab\tasks\brmf0qw1u.output"
$lastCount = 9  # 현재까지 완료된 epoch 수
$epochNum = 9

Add-Type -AssemblyName System.Windows.Forms

function Show-Notification($title, $message) {
    $notify = New-Object System.Windows.Forms.NotifyIcon
    $notify.Icon = [System.Drawing.SystemIcons]::Information
    $notify.Visible = $true
    $notify.ShowBalloonTip(8000, $title, $message, [System.Windows.Forms.ToolTipIcon]::Info)
    Start-Sleep -Milliseconds 8000
    $notify.Dispose()
}

Write-Host "Epoch 알림 모니터 시작 (현재 $lastCount epochs 완료)"

while ($true) {
    $lines = Get-Content $logFile -ErrorAction SilentlyContinue | Select-String "^\s+all\s"
    $currentCount = ($lines | Measure-Object).Count

    if ($currentCount -gt $lastCount) {
        $newLines = $lines | Select-Object -Last ($currentCount - $lastCount)
        foreach ($line in $newLines) {
            $epochNum++
            $vals = ($line.ToString().Trim() -split '\s+')
            $mAP05   = [math]::Round([double]$vals[6] * 100, 1)
            $mAP5095 = [math]::Round([double]$vals[8] * 100, 1)
            $P       = [math]::Round([double]$vals[4] * 100, 1)
            $R       = [math]::Round([double]$vals[5] * 100, 1)

            $title = "Epoch $epochNum/20 완료"
            $msg   = "mAP@0.5: $mAP05%  mAP@.5:.95: $mAP5095%`nP: $P%  R: $R%"
            Write-Host "$title | $msg"
            Show-Notification $title $msg
        }
        $lastCount = $currentCount
    }

    # 학습 종료 감지
    $finished = Get-Content $logFile -ErrorAction SilentlyContinue | Select-String "Results saved"
    if ($finished) {
        Show-Notification "학습 완료!" "전체 20 epoch 학습이 완료되었습니다."
        Write-Host "학습 완료 감지 — 모니터 종료"
        break
    }

    Start-Sleep -Seconds 30
}
