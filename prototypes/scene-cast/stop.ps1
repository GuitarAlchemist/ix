# Stops the scene cast: the public tunnel first, then OBS, then the local video server.
# The public link dies with the tunnel; the next start gets a new random link.
$d = $PSScriptRoot
foreach ($name in 'cloudflared', 'obs', 'mediamtx') {
    $file = Join-Path $d "$name.pid"
    if (-not (Test-Path $file)) { continue }
    $id = [int](Get-Content $file)
    if ($name -eq 'obs') { taskkill /PID $id 2>&1 | Out-Null; [System.Threading.Thread]::Sleep(3000) }
    Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
    Remove-Item $file
    "stopped $name ($id)"
}
Remove-Item (Join-Path $d 'public-url.txt') -ErrorAction SilentlyContinue
