# Start the probe host and the request watcher, unless they already run. Samples, events and
# pid files go to %LOCALAPPDATA%\ix-probes. stop.ps1 stops both.
param(
    [string]$Python = (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python314\python.exe'),
    [string]$Interval = '5s'
)
$here = $PSScriptRoot
$out = Join-Path $env:LOCALAPPDATA 'ix-probes'
$go = Join-Path $env:LOCALAPPDATA 'Programs\go\bin'
if (Test-Path $go) { $env:PATH = "$go;$env:PATH" }  # the host runs go build for each probe
New-Item -ItemType Directory -Force $out | Out-Null

function Running([string]$name) {
    $file = Join-Path $out "$name.pid"
    (Test-Path $file) -and [bool](Get-Process -Id ([int](Get-Content $file)) -ErrorAction SilentlyContinue)
}

$exe = Join-Path $here 'out\bin\host.exe'
if (-not (Test-Path $exe)) {
    Push-Location $here
    go build -o $exe ./host
    Pop-Location
}
if (Running 'host') { 'host already running' } else {
    $p = Start-Process $exe -ArgumentList '-probes', 'probes', '-out', "`"$out`"", '-interval', $Interval `
        -WorkingDirectory $here -WindowStyle Hidden -PassThru
    Set-Content (Join-Path $out 'host.pid') $p.Id
    "host started ($($p.Id))"
}
if (Running 'requests') { 'request watcher already running' } else {
    $p = Start-Process $Python -ArgumentList '-B', "`"$(Join-Path $here 'probe_requests.py')`"", '--every', '2' `
        -WorkingDirectory $here -WindowStyle Hidden -RedirectStandardOutput (Join-Path $out 'requests.log') -PassThru
    Set-Content (Join-Path $out 'requests.pid') $p.Id
    "request watcher started ($($p.Id))"
}
