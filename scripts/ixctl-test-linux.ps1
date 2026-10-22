# Runs the ixctl Go suite on a Linux kernel from Windows, through WSL, the way CI's ixctl-go
# (ubuntu-latest) row does: `go vet`, then `go test -race` (cgo on, gcc in the distro).
#
# Why: the suite passes on Windows and can still fail on Linux. The first CI run of PR #423 lost two
# flood tests that count on a session queue filling behind a WebSocket writer: Linux loopback
# buffers absorb a flood the Windows ones do not. Run this before a push that touches go/ixctl.
#
# Needs: a WSL distro (default `Ubuntu`) with Go at /usr/local/go and gcc. One-time install of Go:
#   curl -fsSLO https://go.dev/dl/go1.27.1.linux-amd64.tar.gz && sudo tar -C /usr/local -xzf go1.27.1.linux-amd64.tar.gz
#
# Usage:
#   pwsh scripts/ixctl-test-linux.ps1                 # vet + race over ./go/ixctl/...
#   pwsh scripts/ixctl-test-linux.ps1 -Run 'TestAJump' # one test, race on
#   pwsh scripts/ixctl-test-linux.ps1 -NoRace          # plain go test
#   pwsh scripts/ixctl-test-linux.ps1 -Count 3         # the suite three times: a flake hunt
#   pwsh scripts/ixctl-test-linux.ps1 -Distro Debian
param(
    [string]$Distro = 'Ubuntu',
    [string]$Run = '',
    [switch]$NoRace,
    [string]$Timeout = '600s',
    [int]$Count = 1                                    # -Count 3 shakes a flake out of a timing-bound test
)
if ($Count -lt 1) { throw "-Count must be at least 1, got $Count" }
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$pkg = Join-Path $repo 'go\ixctl'
if (-not (Test-Path (Join-Path $pkg 'go.mod'))) { throw "no go.mod under $pkg" }
if (-not (Get-Command wsl -ErrorAction SilentlyContinue)) { throw 'wsl is not on PATH: install WSL or run the suite in CI' }

# The Windows path as the distro mounts it (C:\x\y -> /mnt/c/x/y). Computed here, not with
# `wslpath`: backslashes passed through `wsl --` reach bash as escapes and vanish.
if ($pkg -notmatch '^([A-Za-z]):\\') { throw "a drive-letter path is needed, got $pkg" }
$linux = '/mnt/' + $Matches[1].ToLower() + $pkg.Substring(2).Replace('\', '/')

$race = if ($NoRace) { '' } else { '-race' }
$cgo = if ($NoRace) { '0' } else { '1' }
# Values reach bash inside single quotes: an apostrophe in a path or a pattern closes the quote,
# so each one becomes '\'' (close, escaped apostrophe, reopen).
$q = { param($s) "'" + $s.Replace("'", "'\''") + "'" }
$runFlag = if ($Run) { "-run " + (& $q $Run) } else { '' }
# Go is called by its path: the distro's PATH holds the Windows one too, with spaces and parentheses
# bash will not take unquoted. The module cache lives in the distro, the sources on /mnt/c.
# `./...` as CI's ixctl-go job (ci.yml), so a subpackage added later is covered here too.
$go = '/usr/local/go/bin/go'
$cmd = "export CGO_ENABLED=$cgo; cd " + (& $q $linux) + " && $go version && $go vet ./... && $go test $race -timeout $Timeout -count=$Count $runFlag ./..."
Write-Host "wsl -d $Distro -- bash -lc `"$cmd`""
$out = wsl -d $Distro -- bash -lc $cmd 2>&1 | ForEach-Object { ($_ -join '') -replace "`0", '' }
$code = $LASTEXITCODE
$out | ForEach-Object { Write-Host $_ }
if ($code -ne 0) { throw "ixctl suite on Linux failed (exit $code)" }
# `go test -run` with a pattern that matches nothing exits 0 and prints "[no tests to run]": a
# misspelled -Run is not a green run.
if ($Run -and ($out -match 'no tests to run')) { throw "no test matched -Run '$Run'" }
Write-Host 'ixctl suite on Linux: ok'
