### 1. Dependancy

Requirement.txt 참고

'''
pip install -r requirement.txt
'''

### 2. Config 설정

./config.py

`--batch_size` : 배치 사이즈 조정 (`default` : 8) 

`--image_path` : input dataset의 path

`--model_save_path` : 모델 저장 디렉토리 (`default` : `./checkpoint`)

`--severity` : Masking의 강도 조절

그 외 configuration은 해당 파일 참고

### 3.  Training

예시) 

'''
severity 1
$python run.py --model_save_path ./checkpoint1

severity 4
$python run.py --severity 4 --model_save_path ./checkpoint4

contextual learning (severity increased every 10 epoch)
$python run_CL.py --model_save_path ./checkpoint
'''

### 4.  Test

예시) 

'''
$ python test.py --model_save_path checkpoint_CL --pretrained_model_G 5_G_L1_1.pth --pretrained_model_D 5_D_L1_1.pth --severity 3 --save_fig
'''
