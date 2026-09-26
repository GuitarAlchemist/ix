# Score the Spike A learned head (state/router/learned-head.json) on any corpus
# with the same schema as heldout-test.json — local only (Ollama nomic-embed-text,
# the router's lowercase+trim normalization, L2 norm, softmax, max-prob decline at
# the head's own tau). Used to compare head vs Jev on the Stage 2 `fresh` corpus.
# Usage (from the ix root): pwsh state/router-spike/jev/head-on-corpus.ps1 -Corpus <path> -Out <json>
param([Parameter(Mandatory)][string]$Corpus, [Parameter(Mandatory)][string]$Out)
$ErrorActionPreference = "Stop"
$head = Get-Content state/router/learned-head.json -Raw | ConvertFrom-Json
if ($head.pca) { throw "PCA heads are not supported" }
$labels = @($head.labels); $k = $labels.Count; $tau = [double]$head.tau
$prompts = (Get-Content $Corpus -Raw | ConvertFrom-Json).prompts
$inTot = 0; $inOk = 0; $oosTot = 0; $oosOk = 0; $preds = [ordered]@{}
foreach ($p in $prompts) {
  $body = @{ model = "nomic-embed-text"; prompt = $p.prompt.Trim().ToLowerInvariant() } | ConvertTo-Json -Compress
  $v = (Invoke-RestMethod -Uri "http://localhost:11434/api/embeddings" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 60).embedding
  $norm = [math]::Sqrt(($v | ForEach-Object { $_ * $_ } | Measure-Object -Sum).Sum)
  $z = [double[]]::new($k)
  for ($j = 0; $j -lt $k; $j++) { $z[$j] = [double]$head.bias[$j] }
  for ($i = 0; $i -lt $v.Count; $i++) {
    $x = $v[$i] / $norm; $row = $head.weights[$i]
    for ($j = 0; $j -lt $k; $j++) { $z[$j] += $x * $row[$j] }
  }
  $m = ($z | Measure-Object -Maximum).Maximum
  $e = $z | ForEach-Object { [math]::Exp($_ - $m) }; $s = ($e | Measure-Object -Sum).Sum
  $best = 0; for ($j = 1; $j -lt $k; $j++) { if ($e[$j] -gt $e[$best]) { $best = $j } }
  $pred = if ($e[$best] / $s -lt $tau) { "__none__" } else { $labels[$best] }
  $preds[$p.id] = $pred
  if ($p.expectedIntentId -eq "__none__") { $oosTot++; if ($pred -eq "__none__") { $oosOk++ } }
  else { $inTot++; if ($pred -eq $p.expectedIntentId) { $inOk++ } }
}
$r = [ordered]@{ corpus = $Corpus; head = "state/router/learned-head.json"; tau = $tau; inscope_correct = $inOk; inscope_total = $inTot; oos_declined = $oosOk; oos_total = $oosTot; predictions = $preds }
$r | ConvertTo-Json -Depth 4 | Set-Content $Out -Encoding utf8
"head: in-scope $inOk/$inTot, OOS declined $oosOk/$oosTot (tau $tau)"
