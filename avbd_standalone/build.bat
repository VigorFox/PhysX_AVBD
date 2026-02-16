@echo off
setlocal
cd /d "%~dp0"

echo Setting up environment...
if exist "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

echo Building AVBD...
cl /EHsc /O2 /std:c++17 avbd_solver.cpp avbd_tests_stability.cpp avbd_tests_collision.cpp avbd_tests_joints.cpp avbd_main.cpp /Fe:avbd_test.exe

if %errorlevel% neq 0 (
    echo Build FAILED!
    exit /b %errorlevel%
)

echo Build SUCCESS. Running tests...
avbd_test.exe

echo Cleaning up...
del /q *.obj 2>nul
