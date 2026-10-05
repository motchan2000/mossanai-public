# mossanai-v3 pipeline (unattended): back-translation -> migration -> dataset -> LoRA training -> evaluation
#   powershell -File scripts/v3_pipeline.ps1 [-MigrateLimit 50] [-SkipWait]
# ASCII only (Windows PowerShell 5.1 misreads UTF-8 without BOM). Logs: out/v3_pipeline.log
param([int]$MigrateLimit = 50, [switch]$SkipWait)
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
function Log($m) { $line = "$(Get-Date -Format 'HH:mm:ss') $m"; $line; Add-Content -Path out/v3_pipeline.log -Value $line }

if (-not $SkipWait) {
  Log "waiting for back-translation to finish"
  while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match "llm.backtranslate" }) { Start-Sleep -Seconds 30 }
}
Log "back-translation done: $((Get-Content data/user/backtrans.jsonl).Count) requests"

Log "[1/4] migration (limit $MigrateLimit)"
uv run --no-sync python -m llm.migrate_user --limit $MigrateLimit *> out/migrate.log
Log "migration done: $(if (Test-Path data/user/migrated.jsonl) { (Get-Content data/user/migrated.jsonl).Count } else { 0 }) units passed"

Log "[2/4] build v3 datasets"
uv run --no-sync python -m llm.build_v3_data *> out/build_v3.log
Get-Content out/build_v3.log | Where-Object { $_ -match "jsonl|script_repair" } | ForEach-Object { Log "  $_" }

Log "[3/4] train v3"
ollama stop qwen3-coder:30b 2>$null
$train = @("data/sft/fim_train.jsonl", "data/sft/repair_train.jsonl", "data/sft/gen_train.jsonl") + (Get-ChildItem data/sft/v3_*_train.jsonl | ForEach-Object { "data/sft/$($_.Name)" })
$val = @("data/sft/fim_val.jsonl", "data/sft/repair_val.jsonl", "data/sft/gen_val.jsonl") + (Get-ChildItem data/sft/v3_*_val.jsonl | Where-Object { $_.Length -gt 0 } | ForEach-Object { "data/sft/$($_.Name)" })
uv run --no-sync python -m llm.train_lora --data $train --val $val --name v3 --max_len 6144 --tokens_per_batch 6144 --grad_accum 4 --eval_every 50 *> out/train_v3.log
Log "training done: $((Select-String -Path out/train_v3.log -Pattern '\[eval\]' | Select-Object -Last 1).Line)"

Log "[4/4] evaluate v3"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/eval_all.ps1 -Run v3 -Name mossanai-v3 *> out/eval_v3.log
uv run --no-sync python -m llm.eval_user --models mossanai-v3 *> out/eval/user_v3.log
uv run --no-sync python -m llm.eval_user --models mossanai-v3 --rag --compact *> out/eval/user_v3_rag.log
Log "all done"
