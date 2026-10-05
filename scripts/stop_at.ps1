# Stop GPU jobs (eval/train/ollama/BDS) at the given time (HH:mm). ASCII only.
#   Start-Process powershell -ArgumentList "-NoProfile","-File","scripts/stop_at.ps1","-At","14:45" -WindowStyle Hidden
param([Parameter(Mandatory)][string]$At, [string]$Pattern = "llm\.(eval_user|train_lora|eval_fim|eval_gen|eval_repair|backtranslate|migrate_user|distill)")
$deadline = [datetime]::ParseExact($At, "HH:mm", $null)
if ($deadline -lt (Get-Date)) { $deadline = $deadline.AddDays(1) }
while ((Get-Date) -lt $deadline) {
  $alive = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match $Pattern }
  if (-not $alive) { exit 0 }
  Start-Sleep -Seconds 30
}
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match $Pattern } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Get-Process bedrock_server -ErrorAction SilentlyContinue | Stop-Process -Force
& ollama ps | Select-Object -Skip 1 | ForEach-Object { ($_ -split "\s+")[0] } | Where-Object { $_ } | ForEach-Object { & ollama stop $_ }
Add-Content -Path (Join-Path (Split-Path $PSScriptRoot -Parent) "out/stop_at.log") -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm') stopped jobs at deadline $At"
