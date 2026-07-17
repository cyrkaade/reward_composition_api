param(
  [int]$Seed = 2,
  [int]$Timesteps = 30000,
  [int[]]$Queries = @(100, 200, 300, 400),
  [string]$LogRoot = "logs/walker2d_partiality_grid_30k_seed2"
)

$ErrorActionPreference = "Stop"

$suite = "mujoco"
$envId = "Walker2d-v5"
$fragmentLength = 25
$rounds = 4
$partials = @(
  @{ Name = "full"; Ref = "walker2d_partiality_examples:walker2d_example_full" },
  @{ Name = "medium"; Ref = "walker2d_partiality_examples:walker2d_example_medium" },
  @{ Name = "weak"; Ref = "walker2d_partiality_examples:walker2d_example_weak" },
  @{ Name = "low"; Ref = "walker2d_partiality_examples:walker2d_example_low" }
)

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

foreach ($partial in $partials) {
  python -m reward_composition_api partiality `
    --suite $suite `
    --env-id $envId `
    --partial $partial.Ref `
    --timesteps 60000 `
    --fragment-length $fragmentLength `
    --seed $Seed | Out-Host
}

foreach ($partial in $partials) {
  foreach ($queryBudget in $Queries) {
    $collectionTimesteps = [Math]::Max(2000, [int][Math]::Ceiling($queryBudget * $fragmentLength * 2 / $rounds))
    $runName = "walker2d_grid_$($partial.Name)_q${queryBudget}_${Timesteps}_seed$Seed"

    python -m reward_composition_api train `
      --suite $suite `
      --env-id $envId `
      --mode delta `
      --partial $partial.Ref `
      --variant-name "grid_$($partial.Name)_q$queryBudget" `
      --run-name $runName `
      --log-dir $LogRoot `
      --seed $Seed `
      --timesteps $Timesteps `
      --n-envs 4 `
      --eval-freq 10000 `
      --n-eval-episodes 3 `
      --final-eval-episodes 10 `
      --rlhf-rounds $rounds `
      --query-budget $queryBudget `
      --collection-timesteps $collectionTimesteps `
      --fragment-length $fragmentLength `
      --reward-model-epochs 10 `
      --reward-model-patience 3 `
      --reward-model-lr 0.001 `
      --reward-model-batch-size 32 `
      --reward-model-ensemble-size 1 `
      --no-active-learning `
      --include-partial-feature
  }
}

python -m reward_composition_api plot-partiality-grid `
  --runs-root $LogRoot `
  --partiality-root logs/partiality `
  --env-id $envId `
  --output "$LogRoot/partiality_grid.png" `
  --title "Walker2d partiality grid"

Write-Host "Saved runs to $LogRoot"
Write-Host "Saved figure to $LogRoot/partiality_grid.png"
