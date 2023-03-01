curPath=$(readlink -f "$(dirname "$0")")
# python train.py --dataset_path /workspace/volume/sz-008-chengjiankai/SDK/LED/11-05 --label_name ok shaojiao --batch_size 128 --epochs 100 --save_model ./runs/best.pth --device mlu
python train.py --dataset_path /workspace/volume/guojun/Train/Classification/dataset/catsdogs/train/label_info.txt --label_name Cat Dog --batch_size 128 --epochs 200 --save_model /workspace/volume/guojun/Train/Classification/outputs/CatDog.pth --device mlu
# python train.py --dataset_path /workspace/volume/sz-008-chengjiankai/SDK/LED/dataset/label_info.txt --label_name ok error --batch_size 128 --epochs 100 --save_model ./runs/best.pth --device mlu
# python train.py --dataset_path /home/cjkai/workspace/datasets/LED/11-05 --label_name ok shaojiao --batch_size 128 --epochs 100 --save_model ./runs/best.pth --device gpu
