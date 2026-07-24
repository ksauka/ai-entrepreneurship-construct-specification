param(
    [string]$Distro = "",
    [string]$LinuxUser = "suvh",
    [string]$LauncherName = "Start ETV Theory Elaboration Platform.cmd",
    [string]$LegacyTaskName = "ETV Theory Elaboration Platform"
)

$ErrorActionPreference = "Stop"

if (-not $Distro) {
    $defaultLine = wsl.exe --list --verbose |
        Where-Object { $_ -match "^\s*\*" } |
        Select-Object -First 1
    if ($defaultLine) {
        $withoutMarker = $defaultLine -replace "^\s*\*\s*", ""
        $Distro = ($withoutMarker -split "\s{2,}")[0].Trim()
    } else {
        $Distro = wsl.exe --list --quiet |
            ForEach-Object { ($_ -replace "`0", "").Trim() } |
            Where-Object { $_ } |
            Select-Object -First 1
    }
    if (-not $Distro) {
        throw "No WSL distribution was found. Supply -Distro explicitly."
    }
}
if ($Distro -notmatch "^[A-Za-z0-9._-]+$") {
    throw "Distro contains unsupported characters: $Distro"
}
if ($LinuxUser -notmatch "^[A-Za-z0-9._-]+$") {
    throw "LinuxUser contains unsupported characters: $LinuxUser"
}

$startupDirectory = [Environment]::GetFolderPath("Startup")
if (-not $startupDirectory) {
    throw "The current Windows account has no Startup folder."
}
$launcherPath = Join-Path $startupDirectory $LauncherName
$wslPath = Join-Path $env:WINDIR "System32\wsl.exe"
$launcher = @"
@echo off
"$wslPath" -d $Distro -u $LinuxUser --exec /bin/systemctl --user start etv-dashboard.service etv-dashboard-tunnel-named.service
exit /b %ERRORLEVEL%
"@
Set-Content -LiteralPath $launcherPath -Value $launcher -Encoding Ascii

# Remove the earlier Task Scheduler implementation when upgrading an existing
# host. WSL can return 0xFFFFFFFF from a background scheduled task even though
# the same command succeeds in an interactive Windows logon session.
$legacyTask = Get-ScheduledTask -TaskName $LegacyTaskName -ErrorAction SilentlyContinue
if ($legacyTask) {
    Unregister-ScheduledTask -TaskName $LegacyTaskName -Confirm:$false
}

$process = Start-Process `
    -FilePath (Join-Path $env:WINDIR "System32\cmd.exe") `
    -ArgumentList "/d", "/c", "`"$launcherPath`"" `
    -Wait `
    -PassThru
if ($process.ExitCode -ne 0) {
    throw "The Windows Startup launcher returned exit code $($process.ExitCode)."
}

[PSCustomObject]@{
    StartupLauncher = $launcherPath
    TestExitCode = $process.ExitCode
    Distro = $Distro
    LinuxUser = $LinuxUser
}
