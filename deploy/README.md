# Running Hrant in the background

Three platforms supported, one CLI command for each:

```bash
hrant service install         # auto-detects OS, renders + installs
hrant service status          # what the OS service manager reports
hrant service uninstall       # removes the unit, leaves the venv intact
```

`hrant service install` doesn't run privileged commands itself — it
writes the unit/plist/task into the user-mode location for that
platform and prints the exact next step you need to copy-paste. That
way the install is transparent and reversible.

## Linux (systemd, user mode)

```bash
hrant service install --platform linux
# Then activate:
systemctl --user daemon-reload
systemctl --user enable --now hrant.service
# (Optional) survive logout:
sudo loginctl enable-linger $USER
# Live logs:
journalctl --user -u hrant -f
```

## macOS (launchd LaunchAgent)

```bash
hrant service install --platform macos
# Then activate:
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.hrant.agent.plist
launchctl enable    gui/$(id -u)/ai.hrant.agent
launchctl kickstart gui/$(id -u)/ai.hrant.agent
# Logs:
tail -f $PWD/logs/hrant.out.log $PWD/logs/hrant.err.log
```

## Windows (Scheduled Task, no admin)

```powershell
hrant service install --platform windows
# Then activate:
Start-ScheduledTask -TaskName HrantAgent
# Logs (uvicorn writes to console — to capture, use NSSM):
Get-ScheduledTaskInfo -TaskName HrantAgent
```

For boot-time start (before login), see
[deploy/windows/install-nssm.md](windows/install-nssm.md).

## Platform-detection fallback

`hrant service install` without `--platform` auto-detects via
`platform.system()`:

| platform.system() | --platform default |
|-------------------|--------------------|
| Linux             | linux              |
| Darwin            | macos              |
| Windows           | windows            |
