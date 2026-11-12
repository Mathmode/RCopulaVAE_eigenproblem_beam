# R1.4, data budget with the proposed GC-LogN model (as GCLogN_06Nov-5modes_7D_Gamma1.0_Conv6000): eigenvalue solver vs
# frozen surrogate (Ref. 41) trained with N simulations, same reduced data, settings and seed. Local (Windows) version.
#   powershell -File MODULES/comparison_surrogate_decoder/run_data_budget_LogN_local.ps1 -N 1000
param([int]$N = 1000)
$env:PYTHONUNBUFFERED = "1"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root
$py = "C:\Users\anafd\miniconda3\envs\tf_prob_gpu\python.exe"
$B = Join-Path $root "Output\comparison_surrogate_decoder\data_budget"
$data = "$B\data_ntrain$N"; $sur = "$B\surrogate_ntrain$N"
$eig = "$B\LogN_eigen_ntrain$N"; $svae = "$B\LogN_surrogate_ntrain$N"
$common = @("--data-dir", $data, "--n-modes", "5", "--obs", "fv", "--seed", "1234", "--schedule", "plateau", "--epochs", "6000")
if (-not (Test-Path "$eig\final_model_weights.weights.h5")) {
    & $py MODULES/baseline_GaussCopula_GaussMarginals/main_train_GCLogN.py @common --run-name $eig }
if (-not (Test-Path "$svae\surrogateVAE_full_model.weights.h5")) {
    & $py MODULES/comparison_surrogate_decoder/main_train_GCTN_surrogate.py --family gclogn --surrogate-dir $sur @common --run-name $svae }
foreach ($r in @($eig, $svae)) {
    & $py MODULES/review29oct_fixed/postprocessing.py --run-folder $r
    & $py MODULES/model_comparison/evaluate_posterior_quality.py --run-folder $r --samples 500 }
& $py MODULES/comparison_surrogate_decoder/compare_eigen_vs_surrogate.py --eigen-run $eig --surrogate-run $svae --out-dir "$B\LogN_comparison_ntrain$N" --no-figure
& $py MODULES/comparison_surrogate_decoder/analyse_eigen_vs_surrogate.py --eigen-run $eig --surrogate-run $svae --surrogate-dir $sur --out-dir "$B\LogN_comparison_ntrain$N\analysis"
Write-Output "done: $B\LogN_comparison_ntrain$N"
