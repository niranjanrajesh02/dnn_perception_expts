@echo off
REM Sequentially running Python script with different arguments

python benchmark.py --model vgg16
python benchmark.py --model convnext_large
python benchmark.py --model vit_base
python benchmark.py --model vit_large
python benchmark.py --model swin_base
python benchmark.py --model swin_large
python benchmark.py --model deit_base
python benchmark.py --model deit_large


echo All executions finished!
pause