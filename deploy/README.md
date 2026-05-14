# Running Hrant in the background

Everything for "run Hrant as a service" lives under `hrant gateway` —
mirrors `openclaw gateway <action>` so the verbs are familiar.

```bash
hrant gateway start           # install (idempotent) + enable + start
hrant gateway stop            # stop without removing the unit
hrant gateway restart         # restart (use after `hrant update`)
hrant gateway logs -f         # follow journalctl / log file
hrant gateway status          # what the OS service manager reports
hrant gateway install         # only render the unit; don't start
hrant gateway uninstall       # remove the unit (leaves the venv intact)
```

`hrant gateway start` is what 90% of users want — one command from a
fresh `pip install -e .` to a running background service. Under the
hood it calls `gateway install` (renders the platform unit file) and
then runs the platform-native enable + start commands. If you'd
rather inspect the unit file first before activating, use `gateway
install` and run the activation steps yourself (printed below).

`hrant gateway install` never runs privileged commands — it writes
the unit/plist/task into the user-mode location for that platform
and prints the exact next step. Transparent and reversible.

## Linux (systemd, user mode)

```bash
hrant gateway start --platform linux      # install + activate
# OR step-by-step:
hrant gateway install --platform linux
systemctl --user daemon-reload
systemctl --user enable --now hrant.service
# (Optional) survive logout — `gateway start` does this best-effort:
sudo loginctl enable-linger $USER
# Live logs:
hrant gateway logs -f                     # = journalctl --user -u hrant -f
```

## macOS (launchd LaunchAgent)

```bash
hrant gateway start --platform macos      # install + activate
# OR step-by-step:
hrant gateway install --platform macos
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.hrant.agent.plist
launchctl enable    gui/$(id -u)/ai.hrant.agent
launchctl kickstart gui/$(id -u)/ai.hrant.agent
# Logs:
hrant gateway logs -f                     # = tail -f logs/hrant.out.log
```

## Windows (Scheduled Task, no admin)

```powershell
hrant gateway start --platform windows    # install + activate
# OR step-by-step:
hrant gateway install --platform windows
Start-ScheduledTask -TaskName HrantAgent
# Logs (uvicorn writes to console — to capture, use NSSM):
hrant gateway logs                        # = Get-ScheduledTaskInfo
```

For boot-time start (before login), see
[deploy/windows/install-nssm.md](windows/install-nssm.md).

## Platform-detection fallback

`hrant gateway <action>` without `--platform` auto-detects via
`platform.system()`:

| platform.system() | --platform default |
|-------------------|--------------------|
| Linux             | linux              |
| Darwin            | macos              |
| Windows           | windows            |

## Binding beyond loopback

`hrant gateway start` defaults to `--host 127.0.0.1` (loopback only).
To make the agent reachable from other devices on your LAN or
Tailnet:

```bash
hrant gateway start --gateway             # shorthand for --host 0.0.0.0
hrant gateway start --host 100.64.0.5     # bind a specific Tailscale IP
```
