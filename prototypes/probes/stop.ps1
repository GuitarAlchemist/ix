# Stop the probe host through its stop file, so it stops its probes too, then the request watcher.
$out = Join-Path $env:LOCALAPPDATA 'ix-probes'

$file = Join-Path $out 'host.pid'
if (Test-Path $file) {
    $id = [int](Get-Content $file)
    New-Item -ItemType File -Force (Join-Path $out 'stop') | Out-Null
    $p = Get-Process -Id $id -ErrorAction SilentlyContinue
    if ($p -and -not $p.WaitForExit(10000)) {
        Stop-Process -Id $id -Force
        "host killed after 10 s ($id)"
    } else { "host stopped ($id)" }
    Remove-Item $file
}

$file = Join-Path $out 'requests.pid'
if (Test-Path $file) {
    Stop-Process -Id ([int](Get-Content $file)) -Force -ErrorAction SilentlyContinue
    Remove-Item $file
    'request watcher stopped'
}
