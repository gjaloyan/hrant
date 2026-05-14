# Hrant as a Windows Service via NSSM (boot-time start)

The PowerShell scheduled task in `install-service.ps1` is the lightest
path: it starts at user logon, no admin required. If you want the
agent to start **before login** (proper service), use NSSM. This adds
one dependency but lets the agent run on a headless / unattended box.

## Install NSSM

```powershell
winget install NSSM.NSSM
```

## Create the service

`hrant gateway install --platform windows-nssm` renders the right
command for your paths. The manual equivalent:

```powershell
# Replace __WORKDIR__ and __PYTHON_BIN__ with your absolute paths
# (forward slashes are fine in PowerShell).
nssm install Hrant `
    "__PYTHON_BIN__" `
    "-m backend.cli run --host __HOST__ --port __PORT__"

nssm set Hrant AppDirectory   "__WORKDIR__"
nssm set Hrant DisplayName    "Hrant AI Agent"
nssm set Hrant Description    "Self-learning AI agent (FastAPI + Telegram)"
nssm set Hrant Start          SERVICE_AUTO_START
nssm set Hrant AppStdout      "__WORKDIR__\logs\hrant.out.log"
nssm set Hrant AppStderr      "__WORKDIR__\logs\hrant.err.log"
nssm set Hrant AppRotateFiles 1
nssm set Hrant AppRotateBytes 10485760

# Start it
nssm start Hrant
```

## Manage

```powershell
nssm status   Hrant
nssm restart  Hrant
nssm stop     Hrant
nssm remove   Hrant confirm
```

Logs:

```powershell
Get-Content -Wait "__WORKDIR__\logs\hrant.out.log"
```

## Trade-off summary

| Approach           | Admin? | Pre-login start | Service Manager UI |
|--------------------|--------|-----------------|--------------------|
| Scheduled Task     | no     | no              | Task Scheduler     |
| NSSM service       | yes    | yes             | services.msc       |
