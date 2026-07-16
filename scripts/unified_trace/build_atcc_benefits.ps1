param(
    [Parameter(Mandatory = $true)]
    [string]$RawCsv,
    [Parameter(Mandatory = $true)]
    [string]$OutputCsv
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-Mean([object[]]$Values) {
    if ($Values.Count -eq 0) { return [double]::NaN }
    return [double](($Values | Measure-Object -Average).Average)
}

function Get-SampleStd([object[]]$Values) {
    if ($Values.Count -lt 2) { return 0.0 }
    $mean = Get-Mean $Values
    $sum = 0.0
    foreach ($value in $Values) {
        $sum += ([double]$value - $mean) * ([double]$value - $mean)
    }
    return [math]::Sqrt($sum / ($Values.Count - 1))
}

function Get-Values([object[]]$Rows, [string]$Field) {
    return @($Rows | ForEach-Object { [double]($_.$Field) })
}

function Get-Sum([object[]]$Rows, [string]$Field) {
    return [double]((Get-Values $Rows $Field | Measure-Object -Sum).Sum)
}

function Get-Ratio([double]$Numerator, [double]$Denominator) {
    if ($Denominator -eq 0.0) { return [double]::NaN }
    return $Numerator / $Denominator
}

function Round-Metric([double]$Value) {
    if ([double]::IsNaN($Value) -or [double]::IsInfinity($Value)) { return $null }
    return [math]::Round($Value, 6)
}

$raw = @(Import-Csv $RawCsv)
$badRows = @($raw | Where-Object { $_.status -ne "ok" })
if ($badRows.Count -ne 0) {
    throw "Raw input contains $($badRows.Count) non-ok rows"
}

$benefits = @()
$groups = $raw | Group-Object workload, workload_variant, clients | Sort-Object Name
foreach ($group in $groups) {
    $atcc = @(
        $group.Group |
            Where-Object { $_.cc -eq "paper-atcc" } |
            Sort-Object { [int]$_.seed }
    )
    $baseline = @(
        $group.Group |
            Where-Object { $_.cc -ne "paper-atcc" } |
            Sort-Object { [int]$_.seed }
    )
    if ($atcc.Count -eq 0 -or $baseline.Count -eq 0) { continue }

    $pairs = @()
    foreach ($atccRow in $atcc) {
        $baselineRow = @($baseline | Where-Object { $_.seed -eq $atccRow.seed })
        if ($baselineRow.Count -ne 1) {
            throw "Missing unique baseline pair for $($group.Name), seed $($atccRow.seed)"
        }
        $pairs += [pscustomobject]@{
            agent = Get-Ratio ([double]$atccRow.agent_tps) ([double]$baselineRow[0].agent_tps)
            total = Get-Ratio ([double]$atccRow.total_tps) ([double]$baselineRow[0].total_tps)
        }
    }

    $agentAtcc = Get-Values $atcc "agent_tps"
    $agentBase = Get-Values $baseline "agent_tps"
    $totalAtcc = Get-Values $atcc "total_tps"
    $totalBase = Get-Values $baseline "total_tps"
    $backgroundAtcc = Get-Values $atcc "background_tps"
    $backgroundBase = Get-Values $baseline "background_tps"
    $completionAtcc = Get-Values $atcc "agent_task_completion_rate"
    $completionBase = Get-Values $baseline "agent_task_completion_rate"
    $abortAtcc = Get-Values $atcc "agent_attempt_abort_rate"
    $abortBase = Get-Values $baseline "agent_attempt_abort_rate"
    $retryAtcc = Get-Values $atcc "agent_avg_retry_count"
    $retryBase = Get-Values $baseline "agent_avg_retry_count"
    $tokensAtcc = Get-Values $atcc "agent_avg_tokens"
    $tokensBase = Get-Values $baseline "agent_avg_tokens"
    $wastedAtcc = Get-Values $atcc "wasted_reasoning_ms"
    $wastedBase = Get-Values $baseline "wasted_reasoning_ms"
    $p99Atcc = Get-Values $atcc "agent_p99_latency_ms"
    $p99Base = Get-Values $baseline "agent_p99_latency_ms"
    $p999Atcc = Get-Values $atcc "agent_p999_latency_ms"
    $p999Base = Get-Values $baseline "agent_p999_latency_ms"

    $fastOk = Get-Sum $atcc "paper_background_fast_publishes"
    $fastFail = Get-Sum $atcc "paper_background_fast_publish_failures"
    $nativeAttempts = Get-Sum $atcc "paper_background_native_batch_attempts"
    $nativeCommits = Get-Sum $atcc "paper_background_native_batch_commits"
    $privatePrepares = Get-Sum $atcc "paper_version_private_prepares"
    $atomicPublishes = Get-Sum $atcc "paper_version_atomic_publishes"
    $atccAborts = Get-Sum $atcc "agent_aborts"
    $baseAborts = Get-Sum $baseline "agent_aborts"
    $atccValidation = Get-Sum $atcc "version_validation_abort_count"
    $baseValidation = Get-Sum $baseline "version_validation_abort_count"

    $benefits += [pscustomobject][ordered]@{
        workload = $atcc[0].workload
        workload_variant = $atcc[0].workload_variant
        contention = $atcc[0].level
        clients = [int]$atcc[0].clients
        agent_ratio = Round-Metric ([double]$atcc[0].agent_ratio)
        atcc_cc = "paper-atcc"
        baseline_cc = $baseline[0].cc
        n_seeds = $pairs.Count
        warmup_seconds = Round-Metric ([double]$atcc[0].warmup_seconds)
        measure_seconds = Round-Metric ([double]$atcc[0].measure_seconds)
        atcc_agent_tps_mean = Round-Metric (Get-Mean $agentAtcc)
        atcc_agent_tps_std = Round-Metric (Get-SampleStd $agentAtcc)
        baseline_agent_tps_mean = Round-Metric (Get-Mean $agentBase)
        baseline_agent_tps_std = Round-Metric (Get-SampleStd $agentBase)
        agent_tps_ratio_of_means = Round-Metric (Get-Ratio (Get-Mean $agentAtcc) (Get-Mean $agentBase))
        agent_tps_ratio_seed_mean = Round-Metric (Get-Mean (Get-Values $pairs "agent"))
        agent_tps_ratio_seed_std = Round-Metric (Get-SampleStd (Get-Values $pairs "agent"))
        atcc_total_tps_mean = Round-Metric (Get-Mean $totalAtcc)
        atcc_total_tps_std = Round-Metric (Get-SampleStd $totalAtcc)
        baseline_total_tps_mean = Round-Metric (Get-Mean $totalBase)
        baseline_total_tps_std = Round-Metric (Get-SampleStd $totalBase)
        total_tps_ratio_of_means = Round-Metric (Get-Ratio (Get-Mean $totalAtcc) (Get-Mean $totalBase))
        total_tps_ratio_seed_mean = Round-Metric (Get-Mean (Get-Values $pairs "total"))
        total_tps_ratio_seed_std = Round-Metric (Get-SampleStd (Get-Values $pairs "total"))
        atcc_background_tps_mean = Round-Metric (Get-Mean $backgroundAtcc)
        baseline_background_tps_mean = Round-Metric (Get-Mean $backgroundBase)
        background_tps_ratio = Round-Metric (Get-Ratio (Get-Mean $backgroundAtcc) (Get-Mean $backgroundBase))
        atcc_completion_rate_mean = Round-Metric (Get-Mean $completionAtcc)
        baseline_completion_rate_mean = Round-Metric (Get-Mean $completionBase)
        completion_rate_delta = Round-Metric ((Get-Mean $completionAtcc) - (Get-Mean $completionBase))
        atcc_abort_rate_mean = Round-Metric (Get-Mean $abortAtcc)
        baseline_abort_rate_mean = Round-Metric (Get-Mean $abortBase)
        abort_rate_delta = Round-Metric ((Get-Mean $abortAtcc) - (Get-Mean $abortBase))
        atcc_avg_retry_mean = Round-Metric (Get-Mean $retryAtcc)
        baseline_avg_retry_mean = Round-Metric (Get-Mean $retryBase)
        retry_reduction_ratio = Round-Metric (Get-Ratio (Get-Mean $retryBase) (Get-Mean $retryAtcc))
        atcc_avg_tokens_mean = Round-Metric (Get-Mean $tokensAtcc)
        baseline_avg_tokens_mean = Round-Metric (Get-Mean $tokensBase)
        per_task_token_reduction_ratio = Round-Metric (Get-Ratio (Get-Mean $tokensBase) (Get-Mean $tokensAtcc))
        atcc_wasted_reasoning_ms_mean = Round-Metric (Get-Mean $wastedAtcc)
        baseline_wasted_reasoning_ms_mean = Round-Metric (Get-Mean $wastedBase)
        wasted_reasoning_reduction_ratio = Round-Metric (Get-Ratio (Get-Mean $wastedBase) (Get-Mean $wastedAtcc))
        atcc_p99_latency_ms_mean = Round-Metric (Get-Mean $p99Atcc)
        baseline_p99_latency_ms_mean = Round-Metric (Get-Mean $p99Base)
        p99_latency_reduction = Round-Metric (1.0 - (Get-Ratio (Get-Mean $p99Atcc) (Get-Mean $p99Base)))
        atcc_p999_latency_ms_mean = Round-Metric (Get-Mean $p999Atcc)
        baseline_p999_latency_ms_mean = Round-Metric (Get-Mean $p999Base)
        p999_latency_reduction = Round-Metric (1.0 - (Get-Ratio (Get-Mean $p999Atcc) (Get-Mean $p999Base)))
        atcc_validation_abort_count_sum = Round-Metric $atccValidation
        baseline_validation_abort_count_sum = Round-Metric $baseValidation
        atcc_validation_abort_share = Round-Metric (Get-Ratio $atccValidation $atccAborts)
        baseline_validation_abort_share = Round-Metric (Get-Ratio $baseValidation $baseAborts)
        atcc_lock_wait_events_mean = Round-Metric (Get-Mean (Get-Values $atcc "paper_lock_wait_events"))
        atcc_lock_wait_ms_mean = Round-Metric (Get-Mean (Get-Values $atcc "paper_lock_wait_ms"))
        atcc_wounds_mean = Round-Metric (Get-Mean (Get-Values $atcc "paper_wounds"))
        atcc_background_commit_rate_mean = Round-Metric (Get-Mean (Get-Values $atcc "background_commit_rate"))
        baseline_background_commit_rate_mean = Round-Metric (Get-Mean (Get-Values $baseline "background_commit_rate"))
        atcc_private_fast_publish_rate = Round-Metric (Get-Ratio $fastOk ($fastOk + $fastFail))
        atcc_native_batch_commit_rate = Round-Metric (Get-Ratio $nativeCommits $nativeAttempts)
        atcc_atomic_publish_per_private_prepare = Round-Metric (Get-Ratio $atomicPublishes $privatePrepares)
        atcc_agent_admission_deferral_rate_mean = Round-Metric (Get-Mean (Get-Values $atcc "agent_admission_deferral_rate"))
        status = "ok"
    }
}

$benefits |
    Sort-Object workload, @{ Expression = { [int]$_.clients } } |
    Export-Csv $OutputCsv -NoTypeInformation -Encoding UTF8

Write-Output "rows=$($benefits.Count) output=$OutputCsv"
