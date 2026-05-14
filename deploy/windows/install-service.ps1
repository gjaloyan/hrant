# Install Hrant as a Windows scheduled task — runs at user logon,
# restarts on failure. Lighter than NSSM and doesn't need extra
# dependencies; if you want a proper service (auto-start before
# login), see install-nssm.md instead.
#
# Render the template via `hrant gateway install --platform windows`
# which substitutes __PLACEHOLDERS__ with your actual paths. To use
# this file directly, replace the placeholders below.

param(
    [string]$TaskName = "HrantAgent",
    [string]$WorkDir  = "__WORKDIR__",
    [string]$PythonBin = "__PYTHON_BIN__",
    [string]$AgentHost = "__HOST__",
    [string]$AgentPort = "__PORT__"
)

$ErrorActionPreference = "Stop"

# Action: run `python -m backend.cli run` from WorkDir.
$action = New-ScheduledTaskAction `
    -Execute $PythonBin `
    -Argument "-m backend.cli run --host $AgentHost --port $AgentPort" `
    -WorkingDirectory $WorkDir

# Trigger: at user logon. Add `-AtStartup` here if you want it before
# login (requires SYSTEM-mode service via NSSM — see install-nssm.md).
$trigger = New-ScheduledTaskTrigger -AtLogOn

# Settings: restart on failure with 1-min backoff, no fixed end.
$settings = New-ScheduledTaskSettingsSet `
    -RestartCount 6 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([System.TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew

# Register under the current user (no SYSTEM/admin needed).
Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Hrant — self-learning AI agent (FastAPI + Telegram)" `
    -Force

Write-Host ""
Write-Host "Installed scheduled task `"$TaskName`". Manage with:"
Write-Host "  Start-ScheduledTask -TaskName $TaskName"
Write-Host "  Stop-ScheduledTask  -TaskName $TaskName"
Write-Host "  Get-ScheduledTask   -TaskName $TaskName"
Write-Host "  Unregister-ScheduledTask -TaskName $TaskName -Confirm:`$false"
