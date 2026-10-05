# Evaluate a trained LoRA run and register it in Ollama.
#   powershell -File scripts/eval_all.ps1 -Run v2 -Name mossanai-v2
# (ASCII only: Windows PowerShell 5.1 misreads UTF-8 files without BOM)
param([Parameter(Mandatory)][string]$Run, [Parameter(Mandatory)][string]$Name)
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = "utf-8"
$adapter = "out/lora/$Run/best"

"[1/4] FIM eval (transformers)"
uv run --no-sync python -m llm.eval_fim --adapter $adapter --out "out/eval/${Name}_fim.jsonl" *> "out/eval/fim_$Run.log"
"[2/4] export to Ollama"
uv run --no-sync python -m llm.export --adapter $adapter --name $Name --create *> "out/eval/export_$Run.log"
"[3/4] generation eval (Ollama)"
uv run --no-sync python -m llm.eval_gen --ollama $Name --out "out/eval/${Name}_gen.jsonl" *> "out/eval/gen_$Run.log"
"[4/4] repair eval (Ollama)"
uv run --no-sync python -m llm.eval_repair --ollama $Name --out "out/eval/${Name}_repair.jsonl" *> "out/eval/repair_$Run.log"
"done"
