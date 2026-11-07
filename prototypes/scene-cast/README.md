# Scene cast — prototype

[Français](README.fr.md)

A public, read-only live video of the 3D Observatory scene: the user sends a link
and anyone can watch what they see, with no YouTube or Twitch account. Set up on
2026-10-18. The user's condition was « Assure-toi que ce n'est pas dangereux au
niveau sécurité », so the rules below are part of the design.

```
OBS (scene window + scene process audio) --RTMP, loopback--> MediaMTX --HLS, loopback-->
  [front door: player page + Cast button] --> cloudflared quick tunnel --> viewers
```

## What may go out, and what may not

- **Picture**: only the scene's own window, captured by Windows Graphics Capture
  and matched by title. Never the desktop, so no other window can appear.
- **Sound**: only the scene process's audio (OBS Application Audio Capture).
  Never the desktop mix or the microphone, so the household and the local voice
  agents stay off the stream.
- **Server**: MediaMTX listens on 127.0.0.1 only. Publishing needs a secret and
  must come from this machine; reading is anonymous and limited to the `scene`
  path. Its API, metrics, RTSP, WebRTC and SRT are off.
- **Link**: a Cloudflare quick tunnel, started with an empty `--config` so it
  never touches a named tunnel. Its address is random, changes on every start,
  and points at the HLS side alone.
- **In the scene**: its "stream" cast mode hides the private overlays: the bot's
  captions, the private IX terminal and the HUD help. The bot speaks on the
  stream only through a public channel: one line of JSON `{en, fr, es}`, each
  text at most 400 characters, which IX writes atomically and the scene says in
  its own language.

## Setting it up (Windows)

1. MediaMTX (winget `bluenviron.mediamtx`): copy `mediamtx.yml.example` to
   `%LOCALAPPDATA%\scene-cast\mediamtx.yml`, put a random secret in it and in
   `publish.secret`, and start it there.
2. OBS (winget `OBSProject.OBSStudio`), profile `SceneCast`:
   - output: Advanced, encoder `obs_nvenc_h264_tex`, 1920×1080 at 30 fps;
   - `streamEncoder.json`: CBR 3000 kbps, `keyint_sec` 1, preset p4, tune `ll`,
     no B-frames, no lookahead;
   - audio: mono, 48 kHz, 64 kbps;
   - stream service: custom RTMP, server `rtmp://127.0.0.1:1935`, key
     `scene?user=obs&pass=<secret>`.
3. OBS scene collection `SceneCast`, two sources on the scene window: a
   `window_capture` (method WGC, priority "title must match") and a
   `wasapi_process_output_capture`.
4. In OBS's `global.ini` and `user.ini`, turn off the confirmations that would
   block an unattended start: `ConfirmOnExit=false`,
   `WarnBeforeStartingStream=false` and `WarnBeforeStoppingStream=false`.
5. Start OBS streaming:
   `obs64.exe --profile SceneCast --collection SceneCast --scene Scene --startstreaming --minimize-to-tray --disable-updater`,
   from OBS's `bin\64bit` folder.
6. Start the tunnel:
   `cloudflared tunnel --config <an empty file> --no-autoupdate --url http://127.0.0.1:8888`.
   The address it prints, plus `/scene/`, is the link to send.

Stop everything with `stop.ps1`, from the folder that holds the pid files. It
stops the tunnel first, so the public link dies at once. It asks OBS to close
with `taskkill` without `/F` and forces it only if it is still running 3 s
later: a forced kill puts OBS in safe mode at the next start, and safe mode
does not stream.

## Measured on 2026-10-18

- Encoding is on the GPU: OBS used about 0.5% of the CPU.
- The bot's voice was heard on the stream at about -6 dB peaks once the scene's
  speech player was trimmed by 6 dB; it had reached -0.6 dB before.
- OBS hooks the window again by its title after the scene restarts.
- Not explained: at 15:50 the stream was silent for 30 s while the scene said it
  played a line, then sound came back with no change.

The player page starts muted: browsers block autoplay with sound, so viewers
have to unmute it.
