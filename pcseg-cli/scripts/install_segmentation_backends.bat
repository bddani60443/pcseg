@echo off
REM Thin wrapper so Windows users can just double-click (or run without any
REM extra flags) instead of hitting PowerShell's script-execution-policy
REM prompt ("...is not digitally signed. You cannot run this script...").
REM .bat/.cmd files aren't subject to that policy at all - only .ps1 is - so
REM this sidesteps the friction entirely rather than asking the user to
REM change machine-wide settings or remember an -ExecutionPolicy flag every
REM time. Forwards any arguments straight through, e.g.:
REM   install_segmentation_backends.bat -DinoSize base -SamSize large
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_segmentation_backends.ps1" %*
