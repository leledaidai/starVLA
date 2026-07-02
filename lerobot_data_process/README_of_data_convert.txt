python -m lerobot_data_process.build_bridge_lerobot_cot_index \
    --dataset-dir /inspire/hdd/global_user/daizihao-CZXS25110035/zhdai/datasets/bridge_orig_lerobot \
    --output-dir /inspire/hdd/global_user/daizihao-CZXS25110035/zhdai/datasets/bridge_orig_lerobot_cot_index \
    --splits train \
    --validate-against-dataset \
    --overwrite




python -m lerobot_data_process.build_bridge_lerobot_cot_index \
    --dataset-dir /inspire/hdd/global_user/daizihao-CZXS25110035/zhdai/datasets/bridge_orig_train_valid_split_lerobot \
    --output-dir /inspire/hdd/global_user/daizihao-CZXS25110035/zhdai/datasets/bridge_orig_train_valid_split_lerobot_cot_index \
    --splits train val \
    --validate-against-dataset \
    --overwrite




cd /inspire/hdd/global_user/daizihao-CZXS25110035/zhdai/any4lerobot/openx2lerobot

python openx_rlds.py \
    --raw-dir /inspire/hdd/global_user/daizihao-CZXS25110035/zhdai/openpi_dataset_bridge/bridge_orig/1.0.0 \
    --local-dir /inspire/hdd/global_user/daizihao-CZXS25110035/zhdai/datasets \
    --splits train val \
    --use-videos \
    --image-writer-process 0 \
    --image-writer-threads 20 \
    --tfds-parallel-calls 20 \
    --tfds-prefetch 2 \
    --tfds-decode-parallel-calls 20 \
    --tfds-interleave-parallel-calls 20 \
    --tfds-interleave-cycle-length 32 \
    --tfds-buffer-size-mb 64 \
    --tfds-private-threadpool-size 48 \
    --encoder-threads 6 \
    --metadata-buffer-size 100 \
    --batch-encoding-size 1


python openx_rlds.py \
    --raw-dir /inspire/hdd/global_user/daizihao-CZXS25110035/zhdai/openpi_dataset_bridge/bridge_orig/1.0.0 \
    --local-dir /inspire/hdd/global_user/daizihao-CZXS25110035/zhdai/datasets \
    --splits train val \
    --use-videos \
    --image-writer-process 0 \
    --image-writer-threads 20 \
    --tfds-parallel-calls 20 \
    --tfds-prefetch 2 \
    --tfds-decode-parallel-calls 20 \
    --tfds-interleave-parallel-calls 20 \
    --tfds-interleave-cycle-length 32 \
    --tfds-buffer-size-mb 64 \
    --tfds-private-threadpool-size 48 \
    --encoder-threads 6 \
    --metadata-buffer-size 100 \
    --batch-encoding-size 1 \
    --resume \
    --log-every-episodes 100


python openx_rlds.py     --raw-dir /inspire/hdd/global_user/daizihao-CZXS25110035/zhdai/datasets/bridge_orig/1.0.0     --local-dir /inspire/hdd/global_user/daizihao-CZXS25110035/zhdai/datasets     --splits train val     --use-videos     --image-writer-process 0     --image-writer-threads 20     --tfds-parallel-calls 20     --tfds-prefetch 2     --tfds-decode-parallel-calls 20     --tfds-interleave-parallel-calls 20     --tfds-interleave-cycle-length 32     --tfds-buffer-size-mb 64     --tfds-private-threadpool-size 48     --encoder-threads 6     --metadata-buffer-size 100     --batch-encoding-size 1


  python openx_rlds.py \
      --raw-dir /inspire/hdd/global_user/daizihao-CZXS25110035/zhdai/datasets/bridge_orig/1.0.0 \
      --local-dir /inspire/hdd/project/embodied-multimodality/gongjingjing-25039/zhdai \
      --splits train val \
      --use-videos \
      --image-writer-process 0 \
      --image-writer-threads 20 \
      --tfds-parallel-calls 20 \
      --tfds-prefetch 2 \
      --tfds-decode-parallel-calls 20 \
      --tfds-interleave-parallel-calls 20 \
      --tfds-interleave-cycle-length 32 \
      --tfds-buffer-size-mb 64 \
      --tfds-private-threadpool-size 48 \
      --encoder-threads 6 \
      --metadata-buffer-size 100 \
      --batch-encoding-size 1