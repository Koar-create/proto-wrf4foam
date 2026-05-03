@echo off
setlocal

:: Define base paths (Converted from Linux /e/ to Windows E:\)
set "BASE_PATH_0=E:\WRF-OpenFOAM-Coupling"
set "BASE_PATH=E:\WRF-OpenFOAM-Coupling\steady_experiments_finer_ABL"
set "SRC_EXP=20250901_0000_two_boundaries_as_outlet"

:: Change directory to BASE_PATH (using /D to handle drive letter changes)
cd /d "%BASE_PATH%"

:: Loop through all directories starting with "20"
for /d %%I in (20*) do (
    :: Check if the current folder is NOT the source experiment folder
    if /I not "%%I"=="%SRC_EXP%" (
        echo =========================================================
        echo Processing directory: %%I
        echo =========================================================
        
        :: 1. Handle the 'constant' directory and its subdirectories
        if not exist "%BASE_PATH%\%%I\constant\" mkdir "%BASE_PATH%\%%I\constant\"
        
        :: Loop through directory items (extendedFeatureEdgeMesh, polyMesh)
        for %%J in (extendedFeatureEdgeMesh polyMesh) do (
            :: rmdir is used to remove directories or directory symlinks
            if exist "%BASE_PATH%\%%I\constant\%%J\" rmdir /s /q "%BASE_PATH%\%%I\constant\%%J"
            
            :: mklink /D is required for directory symbolic links
            mklink /D "%BASE_PATH%\%%I\constant\%%J" "%BASE_PATH%\%SRC_EXP%\constant\%%J"
        )
        
        :: 2. Handle the '0' directory and its files
        if not exist "%BASE_PATH%\%%I\0\" mkdir "%BASE_PATH%\%%I\0\"
        
        :: Loop through file items (C, Cx, Cy, Cz)
        for %%K in (C Cx Cy Cz) do (
            :: del is used to remove files or file symlinks
            if exist "%BASE_PATH%\%%I\0\%%K" del /f /q "%BASE_PATH%\%%I\0\%%K"
            
            :: mklink (without /D) is used for file symbolic links
            mklink "%BASE_PATH%\%%I\0\%%K" "%BASE_PATH%\%SRC_EXP%\0\%%K"
        )
        
        :: 3. Handle the 'VTK' directory
        if exist "%BASE_PATH%\%%I\VTK\" rmdir /s /q "%BASE_PATH%\%%I\VTK"
        mklink /D "%BASE_PATH%\%%I\VTK" "%BASE_PATH%\%SRC_EXP%\VTK"
        
        :: 4. Handle the 'triSurface' directory (pointing to BASE_PATH_0)
        if exist "%BASE_PATH%\%%I\constant\triSurface\" rmdir /s /q "%BASE_PATH%\%%I\constant\triSurface"
        mklink /D "%BASE_PATH%\%%I\constant\triSurface" "%BASE_PATH_0%\constant\triSurface"
        
        echo.
        echo Finished processing: %%I
        :: echo Press any key to continue to the next directory...
        :: pause >nul
    )
)

echo All operations completed successfully.
pause