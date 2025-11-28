@echo off
cls
echo %~n0%~x0   version 0.0.0
echo.

rem Created by Mechatronic Solutions LLC
rem Mark W Kiehl
rem
rem LICENSE: MIT


rem Batch files: https://steve-jansen.github.io/guides/windows-batch-scripting/
rem Batch files: https://tutorialreference.com/batch-scripting/batch-script-tutorial
rem Scripting Google CLI:  https://cloud.google.com/sdk/docs/scripting-gcloud

rem **********************************************************************************
rem INSTRUCTIONS
rem
rem Edit below:
rem		VENV_FOLDER_NAME
rem		PYTHON_VERSION
rem		PIP INSTALL list
rem 
rem **********************************************************************************

SETLOCAL

rem Define the name for the Python virtual environment folder below
SET VENV_FOLDER_NAME=medium_dspy_multi_agent_pipeline_search_docs_web
echo VENV_FOLDER_NAME: %VENV_FOLDER_NAME%
echo.

rem Define below the Python version to be used.  Recommend Python v3.12
rem py --list-paths will show what Python versions are installed.
SET PYTHON_VERSION=3.12



echo Python versions installed:
py --list-paths
echo.
echo Python version to be used for the new Python virtual environment: %PYTHON_VERSION%


rem The current working directory for this script should be the same as the Python virtual environment for this project.
SET PATH_SCRIPT=%~dp0


rem Define below the Python version GCP should use.  Recommend Python v3.12
rem py --list-paths will show what Python versions are installed.
SET PATH_VENV=%PATH_SCRIPT%%VENV_FOLDER_NAME%
rem echo PATH_VENV: %PATH_VENV%

echo.
echo This script should be run from a Command Prompt (not Windows File Explorer).
echo.
echo A Python virtual environment will be created for the folder:
echo  %PATH_VENV%
echo.
echo Press ENTER to continue, or CTRL-C to abort so you can edit this file '%~n0%~x0'.
pause
echo.


IF EXIST "%PATH_VENV%\." (
	echo ERROR: PATH_VENV already exists!  %PATH_VENV%
	EXIT /B
)

rem Create a virtual envrionment for the folder
echo Creating a Python virtual environment in folder '%PATH_VENV%'..
@echo on
CALL py -%PYTHON_VERSION% -m venv %VENV_FOLDER_NAME%
@echo off
IF %ERRORLEVEL% NEQ 0 (
	echo ERROR %ERRORLEVEL%:
	EXIT /B
)

rem Change to the new virtual environment folder
@echo on
cd %PATH_VENV%
@echo off

rem Activate the virtual environment
echo Activating the Python virtual environment..
@echo on
CALL scripts\activate
@echo off
IF %ERRORLEVEL% NEQ 0 (
	echo ERROR %ERRORLEVEL%:
	EXIT /B
)

@echo on
CALL py -V
@echo off

rem echo.
rem echo If the Python virtual environment has been activated, press ENTER, otherwise CTRL-C to abort.
rem echo.
rem pause


rem EDIT BELOW THE PYTHON PACKAGES TO BE INSTALLED BY PIP
@echo on
CALL py -m pip install --upgrade pip
rem
call py -m pip install dspy
call py -m pip install dotenv
call py -m pip install openai
call py -m pip install langchain-core
call py -m pip install langchain-tavily
call py -m pip install langchain-community
call py -m pip install rank_bm25
call py -m pip install langchain-chroma
call py -m pip install langchain-openai
call py -m pip install langchain-classic
call py -m pip install rich
@echo off

rem Show the currently installed Python packages
echo.
echo Installed Python packages:
@echo on
CALL py -m pip list
@echo off

rem Write the requirements.txt file
@echo on
CALL py -m pip freeze > requirements.txt
@echo off

rem Dectivate the virtual environment
echo Deactivating the Python virtual environment..
@echo on
CALL scripts\deactivate
@echo off
IF %ERRORLEVEL% NEQ 0 (
	echo ERROR %ERRORLEVEL%:
	EXIT /B
)

echo.
echo Normal end of the script. 
pause

ENDLOCAL
