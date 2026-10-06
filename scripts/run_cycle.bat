@echo off
rem Sanity run on the toy QA task with the mock backend.
cd /d "%~dp0.."
python -m macd.main cycle --task-config configs\tasks\qa.yaml --model-config configs\models\local.yaml --cycles 2 %*
