[CmdletBinding()]
param(
    [string]$Executable,

    [string]$Snippet,

    [string]$Case,

    [string]$Joint,

    [string]$Solver = 'avbd',

    [string]$Execution = 'parallel',

    $Frames = 600,

    $Seed = 1,

    $DispatcherThreads = 2,

    $Dt = (1.0 / 60.0),

    [string]$Level = 'regression',

    $TimeoutSeconds = 0,

    [string]$WorkingDirectory,

    [string]$ArtifactRoot,

    [string]$ExpectedSha256,

    [string]$BuildLogPath,

    [string]$BuildCommand,

    [int]$BuildExitCode,

    [int]$BuildWarningCount,

    [int]$BuildErrorCount,

    [switch]$AllowMissingBuildEvidence,

    [hashtable]$ChildEnvironment = @{},

    [string[]]$ExtraArguments = @(),

    [string]$ExpectedCapability = 'SUPPORTED',

    [string]$ExpectedValidation = 'GATED',

    [hashtable]$ExpectedResultFields = @{},

    [switch]$AllowSkip,

    [switch]$AllowNonChecked,

    $ExpectedRequestedFrames = $null,

    [string[]]$RuntimeDependencyPaths = @(),

    [string[]]$RuntimeDependencyExpectedSha256 = @()
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

$script:runnerEvidenceState = [ordered]@{
    stage = 'initialization'
    runDirectory = $null
    manifestPath = $null
    manifest = $null
    executablePath = $null
    executableSha256Before = $null
    runtimeDependencies = @()
    processStarted = $false
    processCompleted = $false
    processId = $null
    timedOut = $null
    processExitCode = $null
    elapsedSeconds = $null
    stdoutFile = 'stdout.log'
    stderrFile = 'stderr.log'
}

function New-EvidenceFinalizationErrorRecord {
    param(
        [string]$Stage,
        [Exception]$Exception
    )

    return [ordered]@{
        utc = (Get-Date).ToUniversalTime().ToString('o')
        stage = $Stage
        exceptionType = $Exception.GetType().FullName
        message = ($Exception.Message -replace '[\r\n]+', ' ')
    }
}

trap {
    $originalException = $_.Exception
    $message = $originalException.Message -replace '[\r\n]+', ' '
    $evidenceState = $script:runnerEvidenceState
    $canWriteFallbackEvidence = $false
    try {
        $canWriteFallbackEvidence =
            $evidenceState.runDirectory -and
            (Test-Path -LiteralPath $evidenceState.runDirectory `
                -PathType Container)
    } catch {
        $fallbackProbeMessage = $_.Exception.Message -replace '[\r\n]+', ' '
        Write-Host ("[AVBD_RUNNER_EVIDENCE] " +
            "stage=fallback-directory-probe error=$fallbackProbeMessage")
    }
    if ($canWriteFallbackEvidence) {
        try {
        $originalErrorRecord = New-EvidenceFinalizationErrorRecord `
            -Stage $evidenceState.stage -Exception $originalException
        $evidenceFinalizationErrors = @($originalErrorRecord)
        $errorTextPath =
            Join-Path $evidenceState.runDirectory 'evidence-finalization-error.txt'
        $errorDetailsPath =
            Join-Path $evidenceState.runDirectory 'evidence-finalization-details.json'
        $utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)

        # Preserve the original exception before attempting any secondary
        # hashing or JSON serialization. A later fallback failure must not
        # erase the exception that caused infrastructure finalization to fail.
        try {
            $initialErrorText = @(
                'AVBD runner evidence finalization failed.',
                "utc=$($originalErrorRecord.utc)",
                "stage=$($originalErrorRecord.stage)",
                "exceptionType=$($originalErrorRecord.exceptionType)",
                "message=$($originalErrorRecord.message)",
                "processStarted=$($evidenceState.processStarted)",
                "processCompleted=$($evidenceState.processCompleted)"
            ) -join [Environment]::NewLine
            [IO.File]::WriteAllText(
                $errorTextPath,
                $initialErrorText + [Environment]::NewLine,
                $utf8WithoutBom)
        } catch {
            $fallbackError = New-EvidenceFinalizationErrorRecord `
                -Stage 'fallback-error-text-write' -Exception $_.Exception
            $evidenceFinalizationErrors += $fallbackError
            Write-Host ("[AVBD_RUNNER_EVIDENCE] stage=" +
                "$($fallbackError.stage) error=$($fallbackError.message)")
        }

        $processEvidence = [ordered]@{
            started = $evidenceState.processStarted
            completed = $evidenceState.processCompleted
            processId = $evidenceState.processId
            timedOut = $evidenceState.timedOut
            exitCode = $evidenceState.processExitCode
            elapsedSeconds = $evidenceState.elapsedSeconds
            stdoutFile = $evidenceState.stdoutFile
            stderrFile = $evidenceState.stderrFile
            evidenceFinalizationStage = $evidenceState.stage
        }

        $fallbackManifest = $evidenceState.manifest
        if (-not $fallbackManifest) {
            $fallbackManifest = [ordered]@{
                schema = 1
                createdUtc = (Get-Date).ToUniversalTime().ToString('o')
                executable = [ordered]@{
                    path = $evidenceState.executablePath
                    sha256Before = $evidenceState.executableSha256Before
                }
                runtimeDependencies = @($evidenceState.runtimeDependencies)
            }
            $evidenceState.manifest = $fallbackManifest
        }
        if (-not $evidenceState.manifestPath) {
            $evidenceState.manifestPath =
                Join-Path $evidenceState.runDirectory 'manifest.json'
        }
        $fallbackManifest['process'] = $processEvidence

        if (-not $fallbackManifest.Contains('executable')) {
            $fallbackManifest['executable'] = [ordered]@{
                path = $evidenceState.executablePath
                sha256Before = $evidenceState.executableSha256Before
            }
        }
        $fallbackExecutable = $fallbackManifest.executable
        $fallbackExecutable['sha256After'] = $null
        $fallbackExecutable['afterStatus'] = 'NOT_AVAILABLE'
        $fallbackExecutable['afterError'] = $null
        if ($evidenceState.executablePath) {
            try {
                if (Test-Path -LiteralPath $evidenceState.executablePath `
                        -PathType Leaf) {
                    $fallbackExecutable.sha256After =
                        (Get-FileHash -Algorithm SHA256 `
                            -LiteralPath $evidenceState.executablePath).Hash.ToUpperInvariant()
                    $fallbackExecutable.afterStatus =
                        if ($fallbackExecutable.sha256After -ceq
                            $evidenceState.executableSha256Before) {
                            'UNCHANGED'
                        } else {
                            'CHANGED'
                        }
                } else {
                    $fallbackExecutable.afterStatus = 'MISSING'
                    $fallbackExecutable.afterError =
                        'Executable was not present during fallback verification.'
                }
            } catch {
                $fallbackExecutable.afterStatus = 'ERROR'
                $fallbackExecutable.afterError =
                    ($_.Exception.Message -replace '[\r\n]+', ' ')
                $fallbackError = New-EvidenceFinalizationErrorRecord `
                    -Stage 'fallback-executable-after-verification' `
                    -Exception $_.Exception
                $evidenceFinalizationErrors += $fallbackError
            }
        }

        foreach ($runtimeDependencyRecord in
                 @($evidenceState.runtimeDependencies)) {
            $runtimeDependencyRecord['sha256After'] = $null
            $runtimeDependencyRecord['afterStatus'] = 'NOT_AVAILABLE'
            $runtimeDependencyRecord['afterError'] = $null
            try {
                if (Test-Path -LiteralPath $runtimeDependencyRecord.originalPath `
                        -PathType Leaf) {
                    $runtimeDependencyRecord.sha256After =
                        (Get-FileHash -Algorithm SHA256 `
                            -LiteralPath $runtimeDependencyRecord.originalPath).Hash.ToUpperInvariant()
                    $runtimeDependencyRecord.afterStatus =
                        if ($runtimeDependencyRecord.sha256After -ceq
                            $runtimeDependencyRecord.sha256Before) {
                            'UNCHANGED'
                        } else {
                            'CHANGED'
                        }
                } else {
                    $runtimeDependencyRecord.afterStatus = 'MISSING'
                    $runtimeDependencyRecord.afterError =
                        'Original dependency was not present during fallback verification.'
                }
            } catch {
                $runtimeDependencyRecord.afterStatus = 'ERROR'
                $runtimeDependencyRecord.afterError =
                    ($_.Exception.Message -replace '[\r\n]+', ' ')
                $fallbackError = New-EvidenceFinalizationErrorRecord `
                    -Stage 'fallback-runtime-dependency-after-verification' `
                    -Exception $_.Exception
                $evidenceFinalizationErrors += $fallbackError
            }

            $runtimeDependencyRecord['snapshotSha256After'] = $null
            $runtimeDependencyRecord['snapshotAfterStatus'] = 'NOT_AVAILABLE'
            $runtimeDependencyRecord['snapshotAfterError'] = $null
            if ($runtimeDependencyRecord.snapshotPath) {
                $fallbackSnapshotPath = Join-Path $evidenceState.runDirectory (
                    $runtimeDependencyRecord.snapshotPath.Replace('/', '\'))
                try {
                    if (Test-Path -LiteralPath $fallbackSnapshotPath -PathType Leaf) {
                        $runtimeDependencyRecord.snapshotSha256After =
                            (Get-FileHash -Algorithm SHA256 `
                                -LiteralPath $fallbackSnapshotPath).Hash.ToUpperInvariant()
                        $runtimeDependencyRecord.snapshotAfterStatus =
                            if ($runtimeDependencyRecord.snapshotSha256After -ceq
                                $runtimeDependencyRecord.snapshotSha256Before) {
                                'UNCHANGED'
                            } else {
                                'CHANGED'
                            }
                    } else {
                        $runtimeDependencyRecord.snapshotAfterStatus = 'MISSING'
                        $runtimeDependencyRecord.snapshotAfterError =
                            'Dependency snapshot was not present during fallback verification.'
                    }
                } catch {
                    $runtimeDependencyRecord.snapshotAfterStatus = 'ERROR'
                    $runtimeDependencyRecord.snapshotAfterError =
                        ($_.Exception.Message -replace '[\r\n]+', ' ')
                    $fallbackError = New-EvidenceFinalizationErrorRecord `
                        -Stage 'fallback-runtime-snapshot-after-verification' `
                        -Exception $_.Exception
                    $evidenceFinalizationErrors += $fallbackError
                }
            }
        }
        $fallbackManifest['runtimeDependencies'] =
            @($evidenceState.runtimeDependencies)
        $fallbackManifest['evidenceFinalizationStatus'] = 'ERROR'
        $fallbackManifest['evidenceFinalizationErrorFile'] =
            'evidence-finalization-error.txt'
        $fallbackManifest['evidenceFinalizationDetailsFile'] = $null

        $fallbackDetailsWritten = $false
        try {
            $fallbackDetails = [ordered]@{
                schema = 1
                classification = 'INFRASTRUCTURE_ERROR'
                process = $processEvidence
                executable = $fallbackExecutable
                runtimeDependencies = @($evidenceState.runtimeDependencies)
                evidenceFinalizationErrors = @($evidenceFinalizationErrors)
            }
            [IO.File]::WriteAllText(
                $errorDetailsPath,
                (($fallbackDetails | ConvertTo-Json -Depth 10) +
                    [Environment]::NewLine),
                $utf8WithoutBom)
            $fallbackDetailsWritten = $true
        } catch {
            $fallbackError = New-EvidenceFinalizationErrorRecord `
                -Stage 'fallback-details-write' -Exception $_.Exception
            $evidenceFinalizationErrors += $fallbackError
            Write-Host ("[AVBD_RUNNER_EVIDENCE] stage=" +
                "$($fallbackError.stage) error=$($fallbackError.message)")
        }
        if ($fallbackDetailsWritten) {
            $fallbackManifest.evidenceFinalizationDetailsFile =
                'evidence-finalization-details.json'
        }
        $fallbackManifest['evidenceFinalizationErrors'] =
            @($evidenceFinalizationErrors)

        try {
            [IO.File]::WriteAllText(
                $evidenceState.manifestPath,
                (($fallbackManifest | ConvertTo-Json -Depth 10) +
                    [Environment]::NewLine),
                $utf8WithoutBom)
        } catch {
            $fallbackError = New-EvidenceFinalizationErrorRecord `
                -Stage 'fallback-manifest-write' -Exception $_.Exception
            $evidenceFinalizationErrors += $fallbackError
            $fallbackManifest['evidenceFinalizationErrors'] =
                @($evidenceFinalizationErrors)
            Write-Host ("[AVBD_RUNNER_EVIDENCE] stage=" +
                "$($fallbackError.stage) error=$($fallbackError.message)")
            try {
                [IO.File]::WriteAllText(
                    $evidenceState.manifestPath,
                    (($fallbackManifest | ConvertTo-Json -Depth 10) +
                        [Environment]::NewLine),
                    $utf8WithoutBom)
            } catch {
                $fallbackRetryError = New-EvidenceFinalizationErrorRecord `
                    -Stage 'fallback-manifest-retry' -Exception $_.Exception
                Write-Host ("[AVBD_RUNNER_EVIDENCE] stage=" +
                    "$($fallbackRetryError.stage) " +
                    "error=$($fallbackRetryError.message)")
            }
        }
        } catch {
            $fallbackUnhandledMessage =
                $_.Exception.Message -replace '[\r\n]+', ' '
            Write-Host ("[AVBD_RUNNER_EVIDENCE] " +
                "stage=fallback-unhandled error=$fallbackUnhandledMessage")
        }
    }
    Write-Host "[AVBD_RUNNER] classification=INFRASTRUCTURE_ERROR error=$message"
    exit 4
}

if ([string]::IsNullOrWhiteSpace($Executable)) {
    throw '-Executable is required.'
}
if ([string]::IsNullOrWhiteSpace($Snippet)) {
    throw '-Snippet is required.'
}
if ([string]::IsNullOrWhiteSpace($Case)) {
    throw '-Case is required.'
}

$runtimeDependencyPathList = @($RuntimeDependencyPaths)
$runtimeDependencyExpectedHashList = @($RuntimeDependencyExpectedSha256)
if ($runtimeDependencyPathList.Count -ne
    $runtimeDependencyExpectedHashList.Count) {
    throw ('-RuntimeDependencyPaths and ' +
           '-RuntimeDependencyExpectedSha256 must have identical counts.')
}
for ($runtimeDependencyIndex = 0;
     $runtimeDependencyIndex -lt $runtimeDependencyPathList.Count;
     $runtimeDependencyIndex++) {
    if ([string]::IsNullOrWhiteSpace(
            [string]$runtimeDependencyPathList[$runtimeDependencyIndex])) {
        throw "Runtime dependency path at index $runtimeDependencyIndex is empty."
    }
    $expectedRuntimeDependencyHash =
        [string]$runtimeDependencyExpectedHashList[$runtimeDependencyIndex]
    if ([string]::IsNullOrWhiteSpace($expectedRuntimeDependencyHash) -or
        $expectedRuntimeDependencyHash.Trim() -cnotmatch '^[A-Fa-f0-9]{64}$') {
        throw ("Runtime dependency expected SHA-256 at index " +
               "$runtimeDependencyIndex must contain exactly 64 hex digits.")
    }
    $runtimeDependencyExpectedHashList[$runtimeDependencyIndex] =
        $expectedRuntimeDependencyHash.Trim().ToUpperInvariant()
}

$Solver = $Solver.ToLowerInvariant()
$Execution = $Execution.ToLowerInvariant()
$Level = $Level.ToLowerInvariant()
$ExpectedCapability = $ExpectedCapability.ToUpperInvariant()
$ExpectedValidation = $ExpectedValidation.ToUpperInvariant()
if ($Joint) {
    $Joint = $Joint.ToLowerInvariant()
}

if (@('avbd', 'tgs', 'pgs') -notcontains $Solver) {
    throw "Invalid -Solver value: $Solver"
}
if (@('parallel', 'sequential') -notcontains $Execution) {
    throw "Invalid -Execution value: $Execution"
}
if (@('smoke', 'regression', 'soak') -notcontains $Level) {
    throw "Invalid -Level value: $Level"
}
if ($Joint -and
    @('spherical', 'fixed', 'd6', 'prismatic', 'revolute') -notcontains $Joint) {
    throw "Invalid -Joint value: $Joint"
}
if (@('SUPPORTED', 'PARTIAL', 'UNKNOWN', 'UNSUPPORTED', 'NOT_APPLICABLE') -notcontains
    $ExpectedCapability) {
    throw "Invalid -ExpectedCapability value: $ExpectedCapability"
}
if (@('NO_HEADLESS', 'HEADLESS_NO_ORACLE', 'PROBE', 'GATED', 'ACCEPTED',
      'REGRESSED', 'BLOCKED') -notcontains $ExpectedValidation) {
    throw "Invalid -ExpectedValidation value: $ExpectedValidation"
}

$normalizedExpectedResultFields = [ordered]@{}
foreach ($expectedResultKey in @($ExpectedResultFields.Keys)) {
    $expectedResultValue = [string]$ExpectedResultFields[$expectedResultKey]
    if ($expectedResultKey -cnotmatch '^[A-Za-z][A-Za-z0-9_]*$') {
        throw "Invalid -ExpectedResultFields key: $expectedResultKey"
    }
    if ([string]::IsNullOrWhiteSpace($expectedResultValue) -or
        $expectedResultValue -notmatch '^\S+$' -or
        @($expectedResultValue.ToCharArray() |
            Where-Object { [int]$_ -gt 127 }).Count) {
        throw "Invalid -ExpectedResultFields value for: $expectedResultKey"
    }
    $normalizedExpectedResultFields[$expectedResultKey] = $expectedResultValue
}

$buildEvidenceParameterNames = @(
    'BuildLogPath', 'BuildCommand', 'BuildExitCode', 'BuildWarningCount',
    'BuildErrorCount'
)
$providedBuildEvidenceParameters = @(
    $buildEvidenceParameterNames | Where-Object {
        $PSBoundParameters.ContainsKey($_)
    }
)
$buildEvidenceComplete =
    $providedBuildEvidenceParameters.Count -eq $buildEvidenceParameterNames.Count
if ($providedBuildEvidenceParameters.Count -ne 0 -and -not $buildEvidenceComplete) {
    throw ('Build evidence must be supplied as a complete set: ' +
           ($buildEvidenceParameterNames -join ', '))
}
if (-not $buildEvidenceComplete -and -not $AllowMissingBuildEvidence) {
    throw ('Complete build evidence is required. Use ' +
           '-AllowMissingBuildEvidence only for non-acceptance diagnostics.')
}
if ($buildEvidenceComplete) {
    if ([string]::IsNullOrWhiteSpace($BuildLogPath) -or
        [string]::IsNullOrWhiteSpace($BuildCommand)) {
        throw '-BuildLogPath and -BuildCommand must be non-empty.'
    }
    if ($BuildCommand -match '[\r\n]') {
        throw '-BuildCommand must be a single line.'
    }
    if ($BuildExitCode -lt 0 -or $BuildWarningCount -lt 0 -or
        $BuildErrorCount -lt 0) {
        throw 'Build exit/warning/error values must be non-negative integers.'
    }
    if ($BuildExitCode -ne 0 -or $BuildErrorCount -ne 0) {
        throw 'Refusing to run a snippet from failed build evidence.'
    }
}

$allowedChildEnvironmentKeys = @(
    'PHYSX_SNIPPET_HEADLESS', 'PHYSX_SNIPPET_SOLVER',
    'PHYSX_SNIPPET_FRAME_COUNT'
)
$normalizedChildEnvironment = [ordered]@{}
foreach ($childEnvironmentKey in @($ChildEnvironment.Keys)) {
    if ($allowedChildEnvironmentKeys -cnotcontains $childEnvironmentKey) {
        throw "Unsupported -ChildEnvironment key: $childEnvironmentKey"
    }
    $childEnvironmentValue = [string]$ChildEnvironment[$childEnvironmentKey]
    if ($childEnvironmentValue -match '[\r\n]') {
        throw "Child environment value must be one line: $childEnvironmentKey"
    }
    $normalizedChildEnvironment[$childEnvironmentKey] = $childEnvironmentValue
}

$parsedFrames = [uint32]0
if (-not [uint32]::TryParse([string]$Frames, [ref]$parsedFrames) -or
    $parsedFrames -lt 1 -or $parsedFrames -gt 100000000) {
    throw '-Frames must be an integer in [1, 100000000].'
}
$Frames = $parsedFrames

$expectedRequestedFramesExplicit =
    $PSBoundParameters.ContainsKey('ExpectedRequestedFrames') -and
    $null -ne $ExpectedRequestedFrames -and
    -not [string]::IsNullOrWhiteSpace([string]$ExpectedRequestedFrames)
$parsedExpectedRequestedFrames = [uint64]0
if (-not $expectedRequestedFramesExplicit) {
    $ExpectedRequestedFrames = [uint64]$Frames
} elseif (-not [uint64]::TryParse([string]$ExpectedRequestedFrames,
                                  [ref]$parsedExpectedRequestedFrames) -or
          $parsedExpectedRequestedFrames -lt 1 -or
          $parsedExpectedRequestedFrames -gt 1000000000000) {
    throw '-ExpectedRequestedFrames must be an integer in [1, 1000000000000].'
} else {
    $ExpectedRequestedFrames = $parsedExpectedRequestedFrames
}

if ($ExpectedRequestedFrames -ne [uint64]$Frames) {
    if (-not $normalizedExpectedResultFields.Contains('cycles') -or
        -not $normalizedExpectedResultFields.Contains('framesPerCycle')) {
        throw ('A differing -ExpectedRequestedFrames requires exact ' +
               '-ExpectedResultFields entries for cycles and framesPerCycle.')
    }

    $expectedCycles = [uint64]0
    $expectedFramesPerCycle = [uint64]0
    if (-not [uint64]::TryParse(
            [string]$normalizedExpectedResultFields['cycles'],
            [ref]$expectedCycles) -or $expectedCycles -lt 1 -or
        -not [uint64]::TryParse(
            [string]$normalizedExpectedResultFields['framesPerCycle'],
            [ref]$expectedFramesPerCycle) -or
        $expectedFramesPerCycle -ne [uint64]$Frames -or
        $expectedCycles -gt
            ([uint64]::MaxValue / $expectedFramesPerCycle) -or
        ($expectedCycles * $expectedFramesPerCycle) -ne
            $ExpectedRequestedFrames) {
        throw ('cycles * framesPerCycle must equal ' +
               '-ExpectedRequestedFrames, and framesPerCycle must equal -Frames.')
    }
}

$parsedSeed = [uint32]0
if (-not [uint32]::TryParse([string]$Seed, [ref]$parsedSeed)) {
    throw '-Seed must be an unsigned 32-bit integer.'
}
$Seed = $parsedSeed

$parsedDispatcherThreads = [uint32]0
if (-not [uint32]::TryParse([string]$DispatcherThreads,
                            [ref]$parsedDispatcherThreads) -or
    $parsedDispatcherThreads -lt 1 -or $parsedDispatcherThreads -gt 256) {
    throw '-DispatcherThreads must be an integer in [1, 256].'
}
$DispatcherThreads = $parsedDispatcherThreads

$parsedDt = [double]0
if (-not [double]::TryParse([string]$Dt,
                            [Globalization.NumberStyles]::Float,
                            [Globalization.CultureInfo]::InvariantCulture,
                            [ref]$parsedDt) -or
    [double]::IsNaN($parsedDt) -or [double]::IsInfinity($parsedDt) -or
    $parsedDt -lt 1.0e-6 -or $parsedDt -gt 1.0) {
    throw '-Dt must be a finite number in [1e-6, 1.0] using invariant format.'
}
$Dt = $parsedDt

$parsedTimeoutSeconds = [int]0
if (-not [int]::TryParse([string]$TimeoutSeconds,
                         [ref]$parsedTimeoutSeconds) -or
    $parsedTimeoutSeconds -lt 0 -or $parsedTimeoutSeconds -gt 86400) {
    throw '-TimeoutSeconds must be an integer in [0, 86400].'
}
$TimeoutSeconds = $parsedTimeoutSeconds

function ConvertTo-WindowsCommandLineArgument {
    param([AllowEmptyString()][string]$Value)

    if ($Value.Length -eq 0) {
        return '""'
    }
    if ($Value -notmatch '[\s"]') {
        return $Value
    }

    $builder = New-Object System.Text.StringBuilder
    [void]$builder.Append('"')
    $backslashes = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq '\') {
            $backslashes++
            continue
        }
        if ($character -eq '"') {
            [void]$builder.Append(('\' * (2 * $backslashes + 1)))
            [void]$builder.Append('"')
            $backslashes = 0
            continue
        }
        if ($backslashes) {
            [void]$builder.Append(('\' * $backslashes))
            $backslashes = 0
        }
        [void]$builder.Append($character)
    }
    if ($backslashes) {
        [void]$builder.Append(('\' * (2 * $backslashes)))
    }
    [void]$builder.Append('"')
    return $builder.ToString()
}

function Write-Utf8File {
    param([string]$Path, [AllowEmptyString()][string]$Text)
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Text, $encoding)
}

function Get-SafeName {
    param([string]$Value)
    return [regex]::Replace($Value, '[^A-Za-z0-9_.-]', '_')
}

$repoRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
if (-not $WorkingDirectory) {
    $WorkingDirectory = $repoRoot
}
$WorkingDirectory = (Resolve-Path -LiteralPath $WorkingDirectory).Path
$Executable = (Resolve-Path -LiteralPath $Executable).Path
$script:runnerEvidenceState.executablePath = $Executable

$runtimeDependencyEvidence = @()
$seenRuntimeDependencyPaths = @{}
for ($runtimeDependencyIndex = 0;
     $runtimeDependencyIndex -lt $runtimeDependencyPathList.Count;
     $runtimeDependencyIndex++) {
    $runtimeDependencyInputPath =
        [string]$runtimeDependencyPathList[$runtimeDependencyIndex]
    if (-not (Test-Path -LiteralPath $runtimeDependencyInputPath -PathType Leaf)) {
        throw ("Runtime dependency at index $runtimeDependencyIndex does not " +
               "exist or is not a file: $runtimeDependencyInputPath")
    }
    $resolvedRuntimeDependencyPath =
        (Resolve-Path -LiteralPath $runtimeDependencyInputPath).Path
    $runtimeDependencyPathKey =
        $resolvedRuntimeDependencyPath.ToUpperInvariant()
    if ($seenRuntimeDependencyPaths.ContainsKey($runtimeDependencyPathKey)) {
        throw ("Duplicate runtime dependency path at index " +
               "${runtimeDependencyIndex}: $resolvedRuntimeDependencyPath")
    }
    $seenRuntimeDependencyPaths[$runtimeDependencyPathKey] = $true

    $runtimeDependencyInfo =
        Get-Item -LiteralPath $resolvedRuntimeDependencyPath
    $runtimeDependencyHashBefore =
        (Get-FileHash -Algorithm SHA256 `
            -LiteralPath $resolvedRuntimeDependencyPath).Hash.ToUpperInvariant()
    $expectedRuntimeDependencyHash =
        $runtimeDependencyExpectedHashList[$runtimeDependencyIndex]
    if ($runtimeDependencyHashBefore -cne $expectedRuntimeDependencyHash) {
        throw ("Runtime dependency SHA-256 mismatch at index " +
               "$runtimeDependencyIndex. Expected " +
               "$expectedRuntimeDependencyHash, found " +
               "${runtimeDependencyHashBefore}: $resolvedRuntimeDependencyPath")
    }

    $runtimeDependencyEvidence += [ordered]@{
        index = $runtimeDependencyIndex
        originalPath = $resolvedRuntimeDependencyPath
        snapshotPath = $null
        sizeBytes = $runtimeDependencyInfo.Length
        expectedSha256 = $expectedRuntimeDependencyHash
        sha256Before = $runtimeDependencyHashBefore
        sha256After = $null
        snapshotSizeBytes = $null
        snapshotSha256Before = $null
        snapshotSha256After = $null
    }
}

if (-not $ArtifactRoot) {
    if (-not $env:TEMP) {
        throw 'TEMP is not defined; specify -ArtifactRoot.'
    }
    $ArtifactRoot = Join-Path $env:TEMP 'PhysX_AVBD_validation'
}
[void](New-Item -ItemType Directory -Force -Path $ArtifactRoot)
$ArtifactRoot = (Resolve-Path -LiteralPath $ArtifactRoot).Path

if (-not $TimeoutSeconds) {
    $TimeoutSeconds = switch ($Level) {
        'smoke' { 120 }
        'regression' { 600 }
        'soak' { 1800 }
    }
}

if (-not $AllowNonChecked -and
    $Executable -notmatch '[\\/]checked[\\/]') {
    Write-Error 'The executable is not from a checked directory. Use -AllowNonChecked to override.'
    exit 4
}

$commit = (& git -C $repoRoot rev-parse HEAD 2>$null | Select-Object -First 1).Trim()
if (-not $commit) {
    throw 'Unable to resolve the repository commit.'
}
$statusText = (& git -C $repoRoot status --porcelain=v1 --untracked-files=all 2>&1 | Out-String).TrimEnd()
$trackedDiff = (& git -C $repoRoot diff HEAD --binary --no-ext-diff 2>&1 | Out-String).TrimEnd()
$branch = (& git -C $repoRoot rev-parse --abbrev-ref HEAD 2>$null | Select-Object -First 1).Trim()
if (-not $branch) {
    throw 'Unable to resolve the repository branch.'
}
$dirty = $statusText.Length -ne 0

$hashBefore = (Get-FileHash -Algorithm SHA256 -LiteralPath $Executable).Hash.ToUpperInvariant()
$script:runnerEvidenceState.executableSha256Before = $hashBefore
if ($ExpectedSha256 -and
    $hashBefore -ne $ExpectedSha256.Trim().ToUpperInvariant()) {
    Write-Error "Executable SHA-256 mismatch. Expected $ExpectedSha256, found $hashBefore."
    exit 4
}

$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$runNonce = [Guid]::NewGuid().ToString('N').Substring(0, 8)
$runName = '{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}' -f $timestamp, $runNonce,
    $commit.Substring(0, 12), $hashBefore.Substring(0, 12),
    (Get-SafeName $Snippet), (Get-SafeName $Case), $Solver, $Execution
$runDirectory = Join-Path $ArtifactRoot $runName
[void](New-Item -ItemType Directory -Path $runDirectory)
$script:runnerEvidenceState.runDirectory = $runDirectory
$script:runnerEvidenceState.stage = 'artifact-initialization'

for ($runtimeDependencyIndex = 0;
     $runtimeDependencyIndex -lt $runtimeDependencyEvidence.Count;
     $runtimeDependencyIndex++) {
    $runtimeDependencyRecord =
        $runtimeDependencyEvidence[$runtimeDependencyIndex]
    $runtimeDependencySnapshotRelativePath =
        'runtime-dependencies/{0:D4}/{1}' -f $runtimeDependencyIndex,
            (Split-Path -Leaf $runtimeDependencyRecord.originalPath)
    $runtimeDependencySnapshotPath = Join-Path $runDirectory (
        $runtimeDependencySnapshotRelativePath.Replace('/', '\'))
    [void](New-Item -ItemType Directory -Force `
        -Path (Split-Path -Parent $runtimeDependencySnapshotPath))
    Copy-Item -LiteralPath $runtimeDependencyRecord.originalPath `
        -Destination $runtimeDependencySnapshotPath
    $runtimeDependencySnapshotInfo =
        Get-Item -LiteralPath $runtimeDependencySnapshotPath
    $runtimeDependencySnapshotHash =
        (Get-FileHash -Algorithm SHA256 `
            -LiteralPath $runtimeDependencySnapshotPath).Hash.ToUpperInvariant()
    if ($runtimeDependencySnapshotHash -cne
        $runtimeDependencyRecord.sha256Before) {
        throw ("Runtime dependency snapshot SHA-256 mismatch at index " +
               "${runtimeDependencyIndex}: " +
               "$runtimeDependencySnapshotRelativePath")
    }
    $runtimeDependencyRecord.snapshotPath =
        $runtimeDependencySnapshotRelativePath
    $runtimeDependencyRecord.snapshotSizeBytes =
        $runtimeDependencySnapshotInfo.Length
    $runtimeDependencyRecord.snapshotSha256Before =
        $runtimeDependencySnapshotHash
}
$script:runnerEvidenceState.runtimeDependencies =
    @($runtimeDependencyEvidence)

$statusPath = Join-Path $runDirectory 'git-status.txt'
$diffPath = Join-Path $runDirectory 'tracked.diff'
Write-Utf8File $statusPath ($statusText + [Environment]::NewLine)
Write-Utf8File $diffPath ($trackedDiff + [Environment]::NewLine)

# A tracked diff cannot contain newly created source files. Snapshot the
# selected snippet, the shared headless contract, and this runner so a dirty
# worktree run remains auditable even before those files are staged.
$sourceCandidates = @($PSCommandPath)
$snippetSourceDirectory = Join-Path $repoRoot (
    'physx\snippets\' + $Snippet.ToLowerInvariant())
if (Test-Path -LiteralPath $snippetSourceDirectory -PathType Container) {
    $sourceCandidates += Get-ChildItem -LiteralPath $snippetSourceDirectory `
        -File -Recurse | Where-Object {
            @('.c', '.cc', '.cpp', '.h', '.hpp', '.inl') -contains $_.Extension
        } | ForEach-Object { $_.FullName }
}
$commonHeadlessHeader =
    Join-Path $repoRoot 'physx\snippets\snippetcommon\SnippetHeadless.h'
if (Test-Path -LiteralPath $commonHeadlessHeader -PathType Leaf) {
    $sourceCandidates += $commonHeadlessHeader
}

$sourceSnapshotDirectory = Join-Path $runDirectory 'source-snapshot'
$sourceSnapshot = @()
foreach ($sourceCandidate in @($sourceCandidates | Sort-Object -Unique)) {
    $resolvedSource = (Resolve-Path -LiteralPath $sourceCandidate).Path
    if (-not $resolvedSource.StartsWith(
            $repoRoot + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase)) {
        throw "Source snapshot path is outside the repository: $resolvedSource"
    }
    $relativeSource = $resolvedSource.Substring($repoRoot.Length).TrimStart(
        [char[]]@('\', '/'))
    $snapshotPath = Join-Path $sourceSnapshotDirectory $relativeSource
    [void](New-Item -ItemType Directory -Force `
        -Path (Split-Path -Parent $snapshotPath))
    Copy-Item -LiteralPath $resolvedSource -Destination $snapshotPath
    $sourceInfo = Get-Item -LiteralPath $resolvedSource
    $sourceSnapshot += [ordered]@{
        path = $relativeSource.Replace('\', '/')
        snapshotPath = ('source-snapshot/' + $relativeSource.Replace('\', '/'))
        sha256 = (Get-FileHash -Algorithm SHA256 `
            -LiteralPath $resolvedSource).Hash.ToUpperInvariant()
        sizeBytes = $sourceInfo.Length
    }
}

$copiedBuildLog = $null
$copiedBuildCommand = $null
$copiedBuildResult = $null
$buildEvidenceFiles = @()
if ($buildEvidenceComplete) {
    $resolvedBuildLog = (Resolve-Path -LiteralPath $BuildLogPath).Path
    $copiedBuildLog = Join-Path $runDirectory 'build.log'
    Copy-Item -LiteralPath $resolvedBuildLog -Destination $copiedBuildLog
    $copiedBuildCommand = Join-Path $runDirectory 'build-command.txt'
    Write-Utf8File $copiedBuildCommand (
        $BuildCommand + [Environment]::NewLine)
    $copiedBuildResult = Join-Path $runDirectory 'build-result.json'
    $buildResult = [ordered]@{
        schema = 1
        exitCode = $BuildExitCode
        warningCount = $BuildWarningCount
        errorCount = $BuildErrorCount
    }
    Write-Utf8File $copiedBuildResult (
        ($buildResult | ConvertTo-Json -Depth 4) + [Environment]::NewLine)
    foreach ($buildEvidenceFile in @(
            $copiedBuildLog, $copiedBuildCommand, $copiedBuildResult)) {
        $buildEvidenceInfo = Get-Item -LiteralPath $buildEvidenceFile
        $buildEvidenceFiles += [ordered]@{
            path = $buildEvidenceInfo.Name
            sha256 = (Get-FileHash -Algorithm SHA256 `
                -LiteralPath $buildEvidenceFile).Hash.ToUpperInvariant()
            sizeBytes = $buildEvidenceInfo.Length
        }
    }
}

$arguments = @(
    '--headless',
    "--solver=$Solver",
    "--case=$Case",
    "--execution=$Execution",
    "--frames=$Frames",
    "--seed=$Seed",
    "--dispatcher-threads=$DispatcherThreads",
    ('--dt=' + $Dt.ToString('R', [Globalization.CultureInfo]::InvariantCulture))
)
if ($Joint) {
    $arguments += "--joint=$Joint"
}
foreach ($extraArgument in $ExtraArguments) {
    if ([string]$extraArgument -match '^--joint(?:=|$)') {
        throw 'Pass the joint selector through -Joint, not -ExtraArguments.'
    }
}
$arguments += $ExtraArguments
$argumentString = (($arguments | ForEach-Object {
            ConvertTo-WindowsCommandLineArgument ([string]$_)
        }) -join ' ')
$commandText = (ConvertTo-WindowsCommandLineArgument $Executable) + ' ' + $argumentString
Write-Utf8File (Join-Path $runDirectory 'command.txt') ($commandText + [Environment]::NewLine)

$relevantEnvironment = @{}
Get-ChildItem Env: | Where-Object {
    $_.Name -like 'PHYSX_SNIPPET_*' -or $_.Name -like 'PHYSX_AVBD_*' -or
    $_.Name -like 'AVBD_*'
} | ForEach-Object {
    $relevantEnvironment[$_.Name] = $_.Value
}
$environmentRecord = [ordered]@{
    inheritedRelevantEnvironment = $relevantEnvironment
    injectedChildEnvironment = $normalizedChildEnvironment
}
Write-Utf8File (Join-Path $runDirectory 'environment.json') (
    ($environmentRecord | ConvertTo-Json -Depth 4) + [Environment]::NewLine)

$executableInfo = Get-Item -LiteralPath $Executable
$manifest = [ordered]@{
    schema = 1
    createdUtc = (Get-Date).ToUniversalTime().ToString('o')
    repository = [ordered]@{
        root = $repoRoot
        commit = $commit
        branch = $branch
        dirty = $dirty
        statusFile = 'git-status.txt'
        trackedDiffFile = 'tracked.diff'
        sourceSnapshotDirectory = 'source-snapshot'
        sourceFiles = $sourceSnapshot
    }
    build = [ordered]@{
        configuration = if ($Executable -match '[\\/]checked[\\/]') { 'checked' } else { 'unknown' }
        platform = if ($Executable -match 'x86_64|x64') { 'x64' } else { 'unknown' }
        crtLayout = if ($Executable -match '\.vc\d+\.md[\\/]') { 'md' } elseif ($Executable -match '\.vc\d+\.mt[\\/]') { 'mt' } else { 'unknown' }
        evidenceComplete = $buildEvidenceComplete
        buildLog = if ($copiedBuildLog) { 'build.log' } else { $null }
        buildCommand = if ($copiedBuildCommand) { 'build-command.txt' } else { $null }
        buildResult = if ($copiedBuildResult) { 'build-result.json' } else { $null }
        exitCode = if ($buildEvidenceComplete) { $BuildExitCode } else { $null }
        warningCount = if ($buildEvidenceComplete) { $BuildWarningCount } else { $null }
        errorCount = if ($buildEvidenceComplete) { $BuildErrorCount } else { $null }
        files = $buildEvidenceFiles
    }
    executable = [ordered]@{
        path = $Executable
        sha256Before = $hashBefore
        sizeBytes = $executableInfo.Length
        lastWriteUtc = $executableInfo.LastWriteTimeUtc.ToString('o')
    }
    runtimeDependencies = @($runtimeDependencyEvidence)
    invocation = [ordered]@{
        command = $commandText
        workingDirectory = $WorkingDirectory
        timeoutSeconds = $TimeoutSeconds
        level = $Level
        snippet = $Snippet
        case = $Case
        joint = $Joint
        solver = $Solver
        execution = $Execution
        frames = $Frames
        expectedRequestedFrames = $ExpectedRequestedFrames
        expectedRequestedFramesExplicit = $expectedRequestedFramesExplicit
        seed = $Seed
        dispatcherThreads = $DispatcherThreads
        dt = $Dt
        expectedCapability = $ExpectedCapability
        expectedValidation = $ExpectedValidation
        expectedResultFields = $normalizedExpectedResultFields
        inheritedRelevantEnvironment = $relevantEnvironment
        injectedChildEnvironment = $normalizedChildEnvironment
    }
}
$manifestPath = Join-Path $runDirectory 'manifest.json'
$script:runnerEvidenceState.manifest = $manifest
$script:runnerEvidenceState.manifestPath = $manifestPath
$script:runnerEvidenceState.stage = 'prelaunch-manifest-write'
Write-Utf8File $manifestPath (($manifest | ConvertTo-Json -Depth 8) + [Environment]::NewLine)

$startInfo = New-Object System.Diagnostics.ProcessStartInfo
$startInfo.FileName = $Executable
$startInfo.Arguments = $argumentString
$startInfo.WorkingDirectory = $WorkingDirectory
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true

$environmentKeys = @($startInfo.EnvironmentVariables.Keys)
foreach ($environmentKey in $environmentKeys) {
    if ($environmentKey -like 'PHYSX_SNIPPET_*' -or
        $environmentKey -like 'PHYSX_AVBD_*' -or
        $environmentKey -like 'AVBD_*') {
        $startInfo.EnvironmentVariables.Remove($environmentKey)
    }
}
foreach ($childEnvironmentEntry in $normalizedChildEnvironment.GetEnumerator()) {
    $startInfo.EnvironmentVariables[$childEnvironmentEntry.Key] =
        [string]$childEnvironmentEntry.Value
}

$process = New-Object System.Diagnostics.Process
$process.StartInfo = $startInfo
$stopwatch = [Diagnostics.Stopwatch]::StartNew()
$script:runnerEvidenceState.stage = 'child-launch'
if (-not $process.Start()) {
    throw 'Failed to start the snippet process.'
}
$script:runnerEvidenceState.processStarted = $true
$script:runnerEvidenceState.processId = $process.Id
$script:runnerEvidenceState.stage = 'child-running'
$stdoutTask = $process.StandardOutput.ReadToEndAsync()
$stderrTask = $process.StandardError.ReadToEndAsync()
$completed = $process.WaitForExit($TimeoutSeconds * 1000)
$timedOut = -not $completed
if ($timedOut) {
    try {
        $process.Kill()
    } catch {
        throw "Timed-out snippet could not be terminated: $($_.Exception.Message)"
    }
    if (-not $process.WaitForExit(5000)) {
        throw 'Timed-out snippet did not terminate within 5 seconds after Kill().'
    }
}
$script:runnerEvidenceState.processCompleted = $true
$script:runnerEvidenceState.timedOut = $timedOut
$script:runnerEvidenceState.stage = 'child-completed'
$stopwatch.Stop()
$script:runnerEvidenceState.elapsedSeconds =
    [Math]::Round($stopwatch.Elapsed.TotalSeconds, 6)
$script:runnerEvidenceState.stage = 'process-exit-code-read'
$processExitCode = if ($timedOut) { $null } else { $process.ExitCode }
$script:runnerEvidenceState.processExitCode = $processExitCode
$script:runnerEvidenceState.stage = 'stdout-async-read'
$stdout = $stdoutTask.Result
$script:runnerEvidenceState.stage = 'stderr-async-read'
$stderr = $stderrTask.Result
$script:runnerEvidenceState.stage = 'process-dispose'
$process.Dispose()

$stdoutPath = Join-Path $runDirectory 'stdout.log'
$stderrPath = Join-Path $runDirectory 'stderr.log'
$script:runnerEvidenceState.stage = 'stdout-log-write'
Write-Utf8File $stdoutPath $stdout
$script:runnerEvidenceState.stage = 'stderr-log-write'
Write-Utf8File $stderrPath $stderr

$script:runnerEvidenceState.stage = 'executable-after-hash'
$hashAfter = (Get-FileHash -Algorithm SHA256 -LiteralPath $Executable).Hash.ToUpperInvariant()
$manifest.executable['sha256After'] = $hashAfter
$runtimeDependencyPostRunErrors =
    New-Object 'System.Collections.Generic.List[string]'
$script:runnerEvidenceState.stage = 'runtime-dependency-after-verification'
foreach ($runtimeDependencyRecord in $runtimeDependencyEvidence) {
    $runtimeDependencyIndex = $runtimeDependencyRecord.index
    if (-not (Test-Path -LiteralPath $runtimeDependencyRecord.originalPath `
            -PathType Leaf)) {
        $runtimeDependencyPostRunErrors.Add(
            "runtime dependency disappeared during the run: index=$runtimeDependencyIndex")
    } else {
        try {
            $runtimeDependencyRecord.sha256After =
                (Get-FileHash -Algorithm SHA256 `
                    -LiteralPath $runtimeDependencyRecord.originalPath).Hash.ToUpperInvariant()
            if ($runtimeDependencyRecord.sha256After -cne
                $runtimeDependencyRecord.sha256Before) {
                $runtimeDependencyPostRunErrors.Add(
                    "runtime dependency SHA-256 changed during the run: index=$runtimeDependencyIndex")
            }
        } catch {
            $runtimeDependencyPostRunErrors.Add(
                "runtime dependency could not be verified after the run: index=$runtimeDependencyIndex error=$($_.Exception.Message)")
        }
    }

    $runtimeDependencySnapshotPath = Join-Path $runDirectory (
        $runtimeDependencyRecord.snapshotPath.Replace('/', '\'))
    if (-not (Test-Path -LiteralPath $runtimeDependencySnapshotPath `
            -PathType Leaf)) {
        $runtimeDependencyPostRunErrors.Add(
            "runtime dependency snapshot disappeared during the run: index=$runtimeDependencyIndex")
    } else {
        try {
            $runtimeDependencyRecord.snapshotSha256After =
                (Get-FileHash -Algorithm SHA256 `
                    -LiteralPath $runtimeDependencySnapshotPath).Hash.ToUpperInvariant()
            if ($runtimeDependencyRecord.snapshotSha256After -cne
                $runtimeDependencyRecord.snapshotSha256Before) {
                $runtimeDependencyPostRunErrors.Add(
                    "runtime dependency snapshot SHA-256 changed during the run: index=$runtimeDependencyIndex")
            }
        } catch {
            $runtimeDependencyPostRunErrors.Add(
                "runtime dependency snapshot could not be verified after the run: index=$runtimeDependencyIndex error=$($_.Exception.Message)")
        }
    }
}
$manifest['process'] = [ordered]@{
    timedOut = $timedOut
    exitCode = $processExitCode
    elapsedSeconds = [Math]::Round($stopwatch.Elapsed.TotalSeconds, 6)
    stdoutFile = 'stdout.log'
    stderrFile = 'stderr.log'
}
$script:runnerEvidenceState.stage = 'final-manifest-write'
Write-Utf8File $manifestPath (($manifest | ConvertTo-Json -Depth 8) + [Environment]::NewLine)
$script:runnerEvidenceState.stage = 'contract-validation'

$contractErrors = New-Object 'System.Collections.Generic.List[string]'
foreach ($runtimeDependencyPostRunError in $runtimeDependencyPostRunErrors) {
    $contractErrors.Add($runtimeDependencyPostRunError)
}
$fields = @{}
$resultLine = $null
$runnerClassification = 'INFRASTRUCTURE_ERROR'

if ($timedOut) {
    $runnerClassification = 'TIMEOUT'
} else {
    $resultLines = @(
        (($stdout -split "`r?`n") + ($stderr -split "`r?`n")) |
            Where-Object { $_ -cmatch '^\[AVBD_GATE\]\s+' }
    )
    if ($resultLines.Count -ne 1) {
        $contractErrors.Add("expected exactly one AVBD_GATE line, found $($resultLines.Count)")
    } else {
        $resultLine = $resultLines[0]
        if (@($resultLine.ToCharArray() | Where-Object { [int]$_ -gt 127 }).Count) {
            $contractErrors.Add('AVBD_GATE line contains non-ASCII characters')
        }
        $tokens = $resultLine.Substring('[AVBD_GATE]'.Length).Trim() -split '\s+'
        foreach ($token in $tokens) {
            if ($token -cnotmatch '^([A-Za-z][A-Za-z0-9_]*)=(\S+)$') {
                $contractErrors.Add("malformed result token: $token")
                continue
            }
            $key = $matches[1]
            $value = $matches[2]
            if ($fields.ContainsKey($key)) {
                $contractErrors.Add("duplicate result key: $key")
            } else {
                $fields[$key] = $value
            }
        }
    }

    $requiredFields = @(
        'schema', 'snippet', 'case', 'solver', 'execution', 'requestedFrames',
        'completedFrames', 'dt', 'seed', 'dispatcherThreads', 'capability',
        'validation', 'status', 'reason', 'nonFinite', 'physicsErrors',
        'physicsWarnings'
    )
    foreach ($requiredField in $requiredFields) {
        if (-not $fields.ContainsKey($requiredField)) {
            $contractErrors.Add("missing result key: $requiredField")
        } else {
            $actualKey = @($fields.Keys | Where-Object { $_ -ieq $requiredField })[0]
            if ($actualKey -cne $requiredField) {
                $contractErrors.Add("non-canonical result key: $actualKey")
            }
        }
    }

    if ($fields.Count) {
        if ($fields.ContainsKey('schema') -and $fields.schema -cne '1') {
            $contractErrors.Add('unsupported result schema')
        }
        foreach ($expected in @{
                snippet = $Snippet
                solver = $Solver
                execution = $Execution
                requestedFrames = [string]$ExpectedRequestedFrames
                seed = [string]$Seed
                dispatcherThreads = [string]$DispatcherThreads
            }.GetEnumerator()) {
            if ($fields.ContainsKey($expected.Key) -and
                $fields[$expected.Key] -cne $expected.Value) {
                $contractErrors.Add("result $($expected.Key) mismatch")
            }
        }

        $isReportedConfigError = $false
        if ($fields.ContainsKey('case')) {
            $isReportedConfigError =
                $fields.ContainsKey('status') -and
                $fields.status -ceq 'ERROR' -and
                $fields.case -ceq 'config-error'
            if (-not $isReportedConfigError -and $fields.case -cne $Case) {
                $contractErrors.Add('result case mismatch')
            }
        }

        $expectedJoint = $Joint
        if (-not $expectedJoint -and $Snippet -ceq 'SnippetJoint') {
            $expectedJoint = switch ($Case) {
                'passive' { 'all' }
                'impact-all' { 'all' }
                'fixed-no-break' { 'fixed' }
                'fixed-break' { 'fixed' }
                default { $null }
            }
        }
        if ($expectedJoint -and -not $isReportedConfigError) {
            if (-not $fields.ContainsKey('joint')) {
                $contractErrors.Add('missing result key: joint')
            } else {
                $actualJointKey =
                    @($fields.Keys | Where-Object { $_ -ieq 'joint' })[0]
                if ($actualJointKey -cne 'joint') {
                    $contractErrors.Add("non-canonical result key: $actualJointKey")
                }
                if ($fields.joint -cne $expectedJoint) {
                    $contractErrors.Add('result joint mismatch')
                }
            }
        }

        $allowedCapabilities = @(
            'SUPPORTED', 'PARTIAL', 'UNKNOWN', 'UNSUPPORTED', 'NOT_APPLICABLE')
        $allowedValidations = @(
            'NO_HEADLESS', 'HEADLESS_NO_ORACLE', 'PROBE', 'GATED',
            'ACCEPTED', 'REGRESSED', 'BLOCKED')
        if ($fields.ContainsKey('capability')) {
            if ($allowedCapabilities -cnotcontains $fields.capability) {
                $contractErrors.Add('invalid capability value')
            } elseif ($fields.capability -cne $ExpectedCapability) {
                $contractErrors.Add('result capability mismatch')
            }
        }
        if ($fields.ContainsKey('validation')) {
            if ($allowedValidations -cnotcontains $fields.validation) {
                $contractErrors.Add('invalid validation value')
            } elseif ($fields.validation -cne $ExpectedValidation) {
                $contractErrors.Add('result validation mismatch')
            }
        }

        foreach ($expectedResult in
                 $normalizedExpectedResultFields.GetEnumerator()) {
            if (-not $fields.ContainsKey($expectedResult.Key)) {
                $contractErrors.Add(
                    "missing expected result key: $($expectedResult.Key)")
            } elseif ($fields[$expectedResult.Key] -cne
                      $expectedResult.Value) {
                $contractErrors.Add(
                    "result $($expectedResult.Key) expectation mismatch")
            }
        }

        if ($fields.ContainsKey('dt')) {
            $parsedDt = 0.0
            $dtParsed = [double]::TryParse(
                $fields.dt,
                [Globalization.NumberStyles]::Float,
                [Globalization.CultureInfo]::InvariantCulture,
                [ref]$parsedDt)
            $expectedFloatDt = [double][single]$Dt
            $dtTolerance = [Math]::Max(1.0e-9,
                                       [Math]::Abs($expectedFloatDt) * 1.0e-6)
            if (-not $dtParsed -or
                [Math]::Abs($parsedDt - $expectedFloatDt) -gt $dtTolerance) {
                $contractErrors.Add('result dt mismatch')
            }
        }

        foreach ($integerField in @('requestedFrames', 'completedFrames',
                                     'seed', 'dispatcherThreads', 'nonFinite',
                                     'physicsErrors', 'physicsWarnings')) {
            if ($fields.ContainsKey($integerField)) {
                $parsedInteger = [uint64]0
                if (-not [uint64]::TryParse($fields[$integerField],
                                            [ref]$parsedInteger)) {
                    $contractErrors.Add("invalid integer result value: $integerField")
                }
            }
        }

        foreach ($entry in $fields.GetEnumerator()) {
            if ($entry.Value -match
                '^(?i:[+-]?(?:nan(?:\([^)]*\))?|inf(?:inity)?|1\.\#(?:INF|QNAN|SNAN|IND)))$') {
                $contractErrors.Add("non-finite result value: $($entry.Key)")
            }
        }

        $allowedStatuses = @('PASS', 'FAIL', 'ERROR', 'SKIP')
        $expectedExitCodes = @{ PASS = 0; FAIL = 1; ERROR = 2; SKIP = 3 }
        if ($fields.ContainsKey('status') -and
            $allowedStatuses -cnotcontains $fields.status) {
            $contractErrors.Add("unknown status: $($fields.status)")
        } elseif ($fields.ContainsKey('status') -and
                  $processExitCode -ne $expectedExitCodes[$fields.status]) {
            $contractErrors.Add('process exit code does not match result status')
        }

        if ($fields.ContainsKey('status') -and $fields.status -ceq 'PASS') {
            if ($fields.completedFrames -cne $fields.requestedFrames) {
                $contractErrors.Add('PASS has incomplete frame count')
            }
            if ($fields.nonFinite -cne '0') {
                $contractErrors.Add('PASS has nonFinite != 0')
            }
            if ($fields.physicsErrors -cne '0') {
                $contractErrors.Add('PASS has physicsErrors != 0')
            }
        }

        if ($fields.ContainsKey('status') -and $fields.status -ceq 'SKIP') {
            if (-not $AllowSkip) {
                $contractErrors.Add('SKIP was not allowed by the runner invocation')
            }
            if ($fields.capability -cne 'UNSUPPORTED' -and
                $fields.capability -cne 'NOT_APPLICABLE') {
                $contractErrors.Add('SKIP has an invalid capability')
            }
        }
    }

    if ($hashAfter -cne $hashBefore) {
        $contractErrors.Add('executable SHA-256 changed during the run')
    }

    if ($contractErrors.Count -eq 0) {
        $runnerClassification = $fields.status
    }
}
if ($runtimeDependencyPostRunErrors.Count) {
    $runnerClassification = 'INFRASTRUCTURE_ERROR'
}

$resultObject = [ordered]@{
    schema = 1
    classification = $runnerClassification
    timedOut = $timedOut
    processExitCode = $processExitCode
    resultLine = $resultLine
    fields = $fields
    contractErrors = @($contractErrors)
    artifactDirectory = $runDirectory
}
Write-Utf8File (Join-Path $runDirectory 'result.json') (
    ($resultObject | ConvertTo-Json -Depth 8) + [Environment]::NewLine)

$summaryLines = @(
    '# AVBD snippet validation run',
    '',
    "- Classification: $runnerClassification",
    "- Process exit: $processExitCode",
    "- Timeout: $timedOut",
    "- Commit: $commit",
    "- Dirty: $dirty",
    "- Executable SHA-256: $hashBefore",
    "- Build evidence complete: $buildEvidenceComplete",
    "- Build exit/warnings/errors: $(if ($buildEvidenceComplete) { "$BuildExitCode/$BuildWarningCount/$BuildErrorCount" } else { 'NOT_SUPPLIED' })",
    "- Elapsed seconds: $([Math]::Round($stopwatch.Elapsed.TotalSeconds, 6))",
    "- Command: ``$commandText``"
)
if ($contractErrors.Count) {
    $summaryLines += ''
    $summaryLines += 'Contract errors:'
    foreach ($contractError in $contractErrors) {
        $summaryLines += "- $contractError"
    }
}
Write-Utf8File (Join-Path $runDirectory 'summary.md') (
    ($summaryLines -join [Environment]::NewLine) + [Environment]::NewLine)

Write-Host "[AVBD_RUNNER] classification=$runnerClassification artifact=$runDirectory"
if ($runtimeDependencyPostRunErrors.Count) {
    exit 4
}
if ($timedOut) {
    exit 124
}
if ($processExitCode -lt 0 -or $processExitCode -gt 3) {
    exit 5
}
if ($contractErrors.Count) {
    exit 4
}
exit $processExitCode
