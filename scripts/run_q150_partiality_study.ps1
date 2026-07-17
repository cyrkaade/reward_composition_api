[CmdletBinding()]
param(
    [ValidateSet("both", "legacy", "rcomp")]
    [string]$Implementation = "both",

    [int]$Seed = 7,
    [int]$Timesteps = 1000000,
    [int]$QueryBudget = 150,
    [int]$RlhfRounds = 5,
    [int]$NEnvs = 8,
    [int]$EvalFreq = 100000,

    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",

    [string]$RcompRoot = "",

    [switch]$PlanOnly,
    [switch]$ValidateOnly,
    [switch]$RerunCompleted
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$legacyApiRoot = Join-Path $repoRoot "api"
$siblingApiRoot = Join-Path (Split-Path $repoRoot -Parent) "reconstructing\api"
$apiRoot = if ($RcompRoot) {
    (Resolve-Path $RcompRoot).Path
} elseif (Test-Path -LiteralPath (Join-Path $legacyApiRoot "pyproject.toml")) {
    $legacyApiRoot
} else {
    $siblingApiRoot
}
$partialFile = Join-Path $repoRoot "partials\scaled_true_reward_levels.py"
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$python = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { (Get-Command python).Source }
$studyRoot = Join-Path $repoRoot "logs\q${QueryBudget}_partiality_seed${Seed}"

if ($Timesteps -le 0) { throw "Timesteps must be greater than zero." }
if ($QueryBudget -le 0) { throw "QueryBudget must be greater than zero." }
if ($RlhfRounds -le 0) { throw "RlhfRounds must be greater than zero." }
if ($NEnvs -le 0) { throw "NEnvs must be greater than zero." }
if (-not (Test-Path -LiteralPath $partialFile)) { throw "Missing partial file: $partialFile" }

$implementations = switch ($Implementation) {
    "both" { @("legacy", "rcomp") }
    default { @($Implementation) }
}

$suites = @(
    [pscustomobject]@{
        Name = "mujoco"
        EnvId = "Reacher-v5"
        Slug = "reacher"
        CollectionTimesteps = 1500
        FragmentLength = 1
        ExtraArgs = @("--preset", "reacher")
    },
    [pscustomobject]@{
        Name = "atari"
        EnvId = "ALE/SpaceInvaders-v5"
        Slug = "spaceinvaders"
        CollectionTimesteps = 50000
        FragmentLength = 64
        ExtraArgs = @()
    }
)

$partials = @(
    [pscustomobject]@{ Label = "p04"; Name = "scaled_true_04" },
    [pscustomobject]@{ Label = "p06"; Name = "scaled_true_06" },
    [pscustomobject]@{ Label = "p08"; Name = "scaled_true_08" }
)

function Get-ImplementationContext([string]$Name) {
    if ($Name -eq "legacy") {
        return [pscustomobject]@{ WorkDir = $repoRoot; Module = "reward_composition_api" }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $apiRoot "pyproject.toml"))) {
        throw "rcomp API not found at $apiRoot. Pass -RcompRoot with the path to reconstructing/api."
    }
    return [pscustomobject]@{ WorkDir = $apiRoot; Module = "rcomp" }
}

function Format-Command([string]$Executable, [string[]]$Arguments) {
    $formatted = foreach ($item in @($Executable) + $Arguments) {
        if ($item -match "[\s']") { "'" + $item.Replace("'", "''") + "'" } else { $item }
    }
    return $formatted -join " "
}

function Invoke-Python([string]$WorkDir, [string[]]$Arguments) {
    if ($PlanOnly) {
        Write-Host "[$WorkDir] $(Format-Command $python $Arguments)"
        return
    }

    Push-Location $WorkDir
    try {
        & $python @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code $LASTEXITCODE`: $(Format-Command $python $Arguments)"
        }
    }
    finally {
        Pop-Location
    }
}

function Test-Partials([string]$ImplementationName) {
    $context = Get-ImplementationContext $ImplementationName
    foreach ($suite in $suites) {
        foreach ($partial in $partials) {
            $partialRef = "${partialFile}:$($partial.Name)"
            $arguments = @(
                "-m", $context.Module, "validate-partial",
                "--suite", $suite.Name,
                "--env-id", $suite.EnvId,
                "--partial", $partialRef
            )
            Invoke-Python $context.WorkDir $arguments
        }
    }
}

function Invoke-Condition(
    [string]$ImplementationName,
    [pscustomobject]$Suite,
    [string]$Mode,
    [string]$Variant,
    [string]$PartialName = ""
) {
    $context = Get-ImplementationContext $ImplementationName
    $logDir = Join-Path $studyRoot "$ImplementationName\$($Suite.Name)"
    $stepsLabel = if ($Timesteps -ge 1000000 -and $Timesteps % 1000000 -eq 0) {
        "$($Timesteps / 1000000)m"
    } else {
        "$Timesteps"
    }
    $runName = "$($Suite.Slug)_${Variant}_q${QueryBudget}_${stepsLabel}_seed${Seed}"
    $metadata = Join-Path $logDir "$runName\metadata.json"

    if ((Test-Path -LiteralPath $metadata) -and -not $RerunCompleted) {
        Write-Host "Skipping completed run: $metadata"
        return
    }

    $arguments = @(
        "-m", $context.Module, "train",
        "--suite", $Suite.Name,
        "--env-id", $Suite.EnvId,
        "--mode", $Mode,
        "--variant-name", $Variant,
        "--run-name", $runName,
        "--log-dir", $logDir,
        "--timesteps", "$Timesteps",
        "--seed", "$Seed",
        "--n-envs", "$NEnvs",
        "--device", $Device,
        "--eval-freq", "$EvalFreq",
        "--n-eval-episodes", "10",
        "--final-eval-episodes", "50",
        "--query-budget", "$QueryBudget",
        "--rlhf-rounds", "$RlhfRounds",
        "--collection-timesteps", "$($Suite.CollectionTimesteps)",
        "--fragment-length", "$($Suite.FragmentLength)",
        "--reward-hidden-sizes", "200",
        "--reward-model-lr", "0.01",
        "--reward-model-epochs", "100",
        "--reward-model-patience", "10",
        "--reward-model-batch-size", "32",
        "--reward-model-ensemble-size", "1",
        "--no-active-learning"
    ) + $Suite.ExtraArgs

    if ($PartialName) {
        $arguments += @(
            "--partial", "${partialFile}:$PartialName",
            "--include-partial-feature"
        )
    }

    Write-Host "`n[$ImplementationName/$($Suite.Name)] $Variant"
    Invoke-Python $context.WorkDir $arguments
}

foreach ($implementationName in $implementations) {
    if (-not $PlanOnly) {
        Write-Host "`nValidating partials with $implementationName CLI..."
    }
    Test-Partials $implementationName
}

if ($ValidateOnly) {
    Write-Host "`nPartial validation succeeded for: $($implementations -join ', ')"
    exit 0
}

foreach ($implementationName in $implementations) {
    foreach ($suite in $suites) {
        Invoke-Condition $implementationName $suite "feedback" "feedback"
        foreach ($mode in @("naive", "delta")) {
            foreach ($partial in $partials) {
                Invoke-Condition $implementationName $suite $mode "$mode`_$($partial.Label)" $partial.Name
            }
        }
    }
}

if ($PlanOnly) {
    Write-Host "`nPlan complete. Remove -PlanOnly to execute it."
} else {
    Write-Host "`nStudy complete. Results: $studyRoot"
}
