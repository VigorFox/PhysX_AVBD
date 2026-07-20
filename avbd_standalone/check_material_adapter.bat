@echo off
REM Structural + symbol gate: AVBD PhysX material adaptation layer (ingest + consume).
setlocal
set ROOT=%~dp0..\physx\source\lowleveldynamics\src
set FAIL=0

echo === AVBD material adapter structure check ===

findstr /C:"constraint.restitution = patch->restitution" "%ROOT%\DyAvbdDynamicsPrep.cpp" >nul
if errorlevel 1 ( echo FAIL: prep does not copy patch restitution & set FAIL=1 ) else ( echo OK: prep restitution )

findstr /C:"constraint.friction = patch->dynamicFriction" "%ROOT%\DyAvbdDynamicsPrep.cpp" >nul
if errorlevel 1 ( echo FAIL: prep does not copy dynamicFriction & set FAIL=1 ) else ( echo OK: prep mu_d )

findstr /C:"constraint.staticFriction = patch->staticFriction" "%ROOT%\DyAvbdDynamicsPrep.cpp" >nul
if errorlevel 1 ( echo FAIL: prep does not copy staticFriction & set FAIL=1 ) else ( echo OK: prep mu_s )

findstr /C:"applyAvbdMaterialNormalVelocity" "%ROOT%\DyAvbdSolver.cpp" >nul
if errorlevel 1 ( echo FAIL: missing applyAvbdMaterialNormalVelocity & set FAIL=1 ) else ( echo OK: material normal response )

findstr /C:"bounceThresholdVelocity" "%ROOT%\DyAvbdTypes.h" >nul
if errorlevel 1 ( echo FAIL: missing bounceThresholdVelocity config & set FAIL=1 ) else ( echo OK: bounce config field )

findstr /C:"getBounceThreshold" "%ROOT%\DyAvbdDynamics.cpp" >nul
if errorlevel 1 ( echo FAIL: scene bounce threshold not wired & set FAIL=1 ) else ( echo OK: scene bounce wire )

findstr /C:"contactCoulombMu" "%ROOT%\DyAvbdSolver.cpp" >nul
if errorlevel 1 ( echo FAIL: friction path missing contactCoulombMu & set FAIL=1 ) else ( echo OK: friction consumes mu )

findstr /C:"hasDeformableStaticAnchor" "%ROOT%\DyAvbdSolver.cpp" >nul
if errorlevel 1 ( echo FAIL: deformable branch missing & set FAIL=1 ) else ( echo OK: deformable branch present )

REM Material response must run for pure dyn-dyn islands (not only body-static).
findstr /C:"// Gate on numContacts (not hasBodyStaticContact)" "%ROOT%\DyAvbdSolver.cpp" >nul
if errorlevel 1 (
  findstr /C:"if (contacts && numContacts > 0)" "%ROOT%\DyAvbdSolver.cpp" >nul
  if errorlevel 1 ( echo FAIL: material path may still gate on hasBodyStaticContact only & set FAIL=1 ) else ( echo OK: material path uses numContacts gate )
) else ( echo OK: material path documents dyn-dyn numContacts gate )

if %FAIL%==1 (
  echo MATERIAL ADAPTER STRUCTURE CHECK FAILED
  exit /b 1
)
echo MATERIAL ADAPTER STRUCTURE CHECK PASSED
exit /b 0
