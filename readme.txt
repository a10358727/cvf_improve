########## step 1 - environment setup #########
   	python 3.8.13
   	pytorch 1.12.1 py3.8_cuda11.3_cudnn8_0
	other require pkgs are list in requirements.txt
########## step 2 - self-defined testing variables setup #########

----testing mode------ 

	python main.py --noise_dataroot "你要測試的圖片位置" --name "選擇results中的名字" 	--model_name "使用的模型名稱"

----train mode------ 

	python main.py --noise_dataroot "濕指紋圖片位置" --clean_dataroot  "乾指紋圖片位置"  --batch_size 64	--train 