@echo off
REM Sequentially running Python script with different arguments

python benchmark.py --model resnet50_at 
python benchmark.py --model vit_base_at 
python benchmark.py --model resnet50_moco 
python benchmark.py --model vit_base_moco 
python benchmark.py --model resnet50_dino 
python benchmark.py --model vit_base_dino 


echo All executions finished!
pause

