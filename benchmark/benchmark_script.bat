@echo off
REM Sequentially running Python script with different arguments

python benchmark.py --model vit_base_places365 
python benchmark.py --model densenet161


echo All executions finished!
pause

