param(
  [int]$Seed = 2,
  [int]$Timesteps = 80000,
  [int[]]$QueryBudgets = @(100, 200),
  [string[]]$Modes = @("feedback", "naive", "delta"),
  [string]$LogRoot = "",
  [switch]$UseProvided800kSchedule,
  [int]$PolicyStepsPerRound = 0,
  [int]$FinalPolicyTimesteps = 0,
  [int]$Rounds = 5,
  [int]$FragmentLength = 25,
  [int]$Oversampling = 2,
  [int]$NEnvs = 8,
  [int]$EvalFreq = 50000,
  [int]$NEvalEpisodes = 10,
  [int]$FinalEvalEpisodes = 100,
  [int]$RewardModelEpochs = 100,
  [int]$RewardModelPatience = 10,
  [double]$RewardModelLr = 0.001,
  [int]$RewardModelBatchSize = 32,
  [int]$RewardModelEnsembleSize = 5,
  [switch]$SkipRuns,
  [switch]$ForceRerun
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$mainRoot = Join-Path $repoRoot "baselines\main_5615e65"

if (-not (Test-Path -LiteralPath $mainRoot)) {
  throw "Missing main snapshot: $mainRoot"
}

if ($UseProvided800kSchedule) {
  $Timesteps = 800000
  if ($PolicyStepsPerRound -eq 0) {
    $PolicyStepsPerRound = 110000
  }
  if ($FinalPolicyTimesteps -eq 0) {
    $FinalPolicyTimesteps = 250000
  }
}

if ([string]::IsNullOrWhiteSpace($LogRoot)) {
  $LogRoot = Join-Path $repoRoot ("logs\reacher_main_compare_{0}_seed{1}" -f $Timesteps, $Seed)
} elseif (-not [System.IO.Path]::IsPathRooted($LogRoot)) {
  $LogRoot = Join-Path $repoRoot $LogRoot
}

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $venvPython) {
  $python = $venvPython
} else {
  $python = "python"
}

$versions = @(
  @{ Name = "current"; Root = $repoRoot; Partial = "partials/reacher_distance_partial.py" },
  @{ Name = "main"; Root = $mainRoot; Partial = "../../partials/reacher_distance_partial.py" }
)

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

function Find-CompletedRunMetadata {
  param(
    [string]$VersionLogRoot,
    [string]$Mode,
    [int]$QueryBudget
  )

  if (-not (Test-Path -LiteralPath $VersionLogRoot)) {
    return $null
  }

  foreach ($metadata in Get-ChildItem -Recurse -Filter metadata.json -Path $VersionLogRoot -ErrorAction SilentlyContinue) {
    try {
      $run = Get-Content -Raw $metadata.FullName | ConvertFrom-Json
      if (
        $run.mode -eq $Mode -and
        [int]$run.query_budget -eq $QueryBudget -and
        [int]$run.requested_timesteps -eq $Timesteps -and
        [int]$run.seed -eq $Seed
      ) {
        return $metadata.FullName
      }
    } catch {
      Write-Warning "Could not inspect metadata $($metadata.FullName): $_"
    }
  }

  return $null
}

function Get-CleanRunName {
  param(
    [string]$VersionLogRoot,
    [string]$BaseRunName
  )

  $candidate = $BaseRunName
  $index = 1
  while ((Test-Path -LiteralPath (Join-Path $VersionLogRoot $candidate)) -and
         (-not (Test-Path -LiteralPath (Join-Path (Join-Path $VersionLogRoot $candidate) "metadata.json")))) {
    $candidate = "${BaseRunName}_retry${index}"
    $index += 1
  }
  return $candidate
}

function Invoke-TrainRun {
  param(
    [hashtable]$Version,
    [string]$Mode,
    [int]$QueryBudget
  )

  $versionLogRoot = Join-Path $LogRoot $Version.Name
  New-Item -ItemType Directory -Force -Path $versionLogRoot | Out-Null

  if (-not $ForceRerun) {
    $completedMetadata = Find-CompletedRunMetadata -VersionLogRoot $versionLogRoot -Mode $Mode -QueryBudget $QueryBudget
    if ($null -ne $completedMetadata) {
      Write-Host "[$($Version.Name)] skipping completed mode=$Mode q=$QueryBudget ($completedMetadata)"
      return
    }
  }

  $collectionTimesteps = [Math]::Max(
    2000,
    [int][Math]::Ceiling($QueryBudget * $FragmentLength * $Oversampling / $Rounds)
  )
  $runName = "reacher_${Mode}_q${QueryBudget}_${Timesteps}_seed${Seed}"
  if (-not $ForceRerun) {
    $runName = Get-CleanRunName -VersionLogRoot $versionLogRoot -BaseRunName $runName
  }
  $variantName = "${Mode}_q${QueryBudget}"
  $runDir = Join-Path $versionLogRoot $runName

  $pythonArgs = @(
    "-m", "reward_composition_api", "train",
    "--suite", "mujoco",
    "--env-id", "Reacher-v5",
    "--mode", $Mode,
    "--variant-name", $variantName,
    "--run-name", $runName,
    "--log-dir", $versionLogRoot,
    "--seed", $Seed,
    "--timesteps", $Timesteps,
    "--n-envs", $NEnvs,
    "--eval-freq", $EvalFreq,
    "--n-eval-episodes", $NEvalEpisodes,
    "--final-eval-episodes", $FinalEvalEpisodes,
    "--rlhf-rounds", $Rounds,
    "--query-budget", $QueryBudget,
    "--collection-timesteps", $collectionTimesteps,
    "--fragment-length", $FragmentLength,
    "--reward-model-epochs", $RewardModelEpochs,
    "--reward-model-patience", $RewardModelPatience,
    "--reward-model-lr", $RewardModelLr,
    "--reward-model-batch-size", $RewardModelBatchSize,
    "--reward-model-ensemble-size", $RewardModelEnsembleSize,
    "--active-learning",
    "--active-query-strategy", "auto"
  )

  if ($PolicyStepsPerRound -gt 0) {
    $pythonArgs += @("--policy-timesteps-per-round", $PolicyStepsPerRound)
  }
  if ($FinalPolicyTimesteps -gt 0) {
    $pythonArgs += @("--final-policy-timesteps", $FinalPolicyTimesteps)
  }
  if ($Mode -in @("naive", "delta")) {
    $pythonArgs += @("--partial", $Version.Partial, "--include-partial-feature")
  }

  Write-Host ""
  Write-Host "[$($Version.Name)] mode=$Mode q=$QueryBudget seed=$Seed timesteps=$Timesteps collect=$collectionTimesteps"
  Push-Location $Version.Root
  try {
    & $python @pythonArgs
    if ($LASTEXITCODE -ne 0) {
      throw "Run failed with exit code $LASTEXITCODE"
    }
  } finally {
    Pop-Location
  }
}

$oldPythonPath = $env:PYTHONPATH
try {
  $env:PYTHONPATH = $null

  if (-not $SkipRuns) {
    foreach ($version in $versions) {
      foreach ($queryBudget in $QueryBudgets) {
        foreach ($mode in $Modes) {
          Invoke-TrainRun -Version $version -Mode $mode -QueryBudget $queryBudget
        }
      }
    }
  }

  $comparisonCsv = Join-Path $LogRoot "comparison.csv"
  & $python (Join-Path $repoRoot "scripts\compare_reacher_runs.py") --root $LogRoot --out $comparisonCsv
  if ($LASTEXITCODE -ne 0) {
    throw "Comparison failed with exit code $LASTEXITCODE"
  }
} finally {
  $env:PYTHONPATH = $oldPythonPath
}

Write-Host ""
Write-Host "Saved logs under $LogRoot"
Write-Host "Saved comparison to $(Join-Path $LogRoot 'comparison.csv')"
