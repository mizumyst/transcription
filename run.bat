@echo off

python --version 3> nul
if errorlevel 1 goto python_install

:menu
cls

echo 1) Begin inference
echo 2) Export JSON to XML
echo 3) Export XML to Word
echo.
echo 8) Export XML to JSON
echo 9) Export JSON to Word
echo.
echo X) Exit

set /p o="> "

if %o% == 1 goto infer
if %o% == 2 goto to_elan
if %o% == 3 goto to_word

if %o% == 8 goto to_json_xml
if %o% == 9 goto to_word_json
if %o% == 0 goto requirements

if "%o%" == "X" goto eof
if "%o%" == "x" goto eof

goto menu

:infer

py src/main.py
echo Inference complete.

:to_elan

py src/export.py data xml
echo Exported to XML (ELAN).
pause

goto menu

:to_word

py src/export.py data docx xml
echo Exported to Word.
pause

goto menu

:to_json_xml

py src/export.py data json
echo Exported to JSON.
pause

goto menu

:to_word_json

py src/export.py data docx json
echo Exported to Word.
pause

goto menu

:requirements

pip install -r src/requirements.txt

cls
echo Installation of required libraries complete.
pause

goto menu

:python_install
echo Python 3 is not installed.
echo.
echo Once Python 3 is installed, restart this program and
echo enter 0 to complete installation.

python

pause

:eof
