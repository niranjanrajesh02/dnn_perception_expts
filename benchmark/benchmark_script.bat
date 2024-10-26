@echo off
REM Sequentially running Python script with different arguments

python benchmark.py --model facenet_casia 
python benchmark.py --model facenet_vggface2 
python benchmark.py --model face_vit 
python benchmark.py --model resnet50_places365 
python benchmark.py --model densenet161_places365 
python benchmark.py --model vit_base_places365 
python benchmark.py --model densenet161


echo All executions finished!
pause

