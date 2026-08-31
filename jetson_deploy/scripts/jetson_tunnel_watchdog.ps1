param(
    [int]$LocalPort = 18502,
    [int]$RemotePort = 8502,
    [string]$RemoteHost = "your-jetson-hostname",
    [string]$RemoteUser = "your-jetson-username"
)

$ErrorActionPreference = "SilentlyContinue"
$sshPath = "C:\Windows\System32\OpenSSH\ssh.exe"
$workspace = Split-Path -Parent $PSScriptRoot
$errorLog = Join-Path $workspace "ssh_tunnel_watchdog.log"

while ($true) {
    $address = $null
    $useIpv6 = $false

    $ipv6PingText = (& ping.exe -6 -n 1 -w 1500 $RemoteHost 2>$null) -join " "
    $ipv6Match = [regex]::Match($ipv6PingText, "\[([0-9a-fA-F:]+)\]")
    if ($ipv6PingText -match "Reply from" -and $ipv6Match.Success) {
        $address = $ipv6Match.Groups[1].Value
        $useIpv6 = $true
    }

    if (-not $address) {
        $pingText = (& ping.exe -4 -n 1 -w 1500 $RemoteHost 2>$null) -join " "
        $match = [regex]::Match($pingText, "\[(\d{1,3}(?:\.\d{1,3}){3})\]")
        if ($match.Success) {
            $address = $match.Groups[1].Value
        }
    }

    if (-not $address) {
        $address = Resolve-DnsName -Name $RemoteHost -Type A -ErrorAction SilentlyContinue |
            Where-Object { $_.IPAddress -match "^\d{1,3}(?:\.\d{1,3}){3}$" } |
            Select-Object -ExpandProperty IPAddress -First 1
    }

    if ($address) {
        $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $route = if ($useIpv6) { "IPv6" } else { "IPv4" }
        Add-Content -LiteralPath $errorLog -Value "$stamp connecting via $route to $RemoteUser@$address"
        $sshArguments = @(
            "-N",
            "-L", "127.0.0.1:${LocalPort}:127.0.0.1:${RemotePort}",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=8",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=5",
            "-o", "ServerAliveCountMax=2",
            "-o", "StrictHostKeyChecking=accept-new",
            "${RemoteUser}@${address}"
        )
        & $sshPath @sshArguments 2>> $errorLog
    }

    Start-Sleep -Seconds 2
}
