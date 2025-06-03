# !/usr/bin/env sh

### searched + fapn + newcrfs
python ./tools/test.py \
        ./configs/evomde_test_kitti.py \
        --checkpoint ./weights/kitti_best.pth \
        --dataset kitti \
        --input_height 352 \
        --input_width 1120 \
        --max_depth 80.0 \
        --max_depth_eval 80 \
        --batch_size 1 \
        --data_path_eval /data/dataset/KITTI/ \
        --gt_path_eval /data/dataset/KITTI/data_depth_annotated/ \
        --filenames_file_eval data_splits/eigen_test_files_with_gt.txt \
        --do_kb_crop \
        --num_threads 1 \
        --garg_crop \
        --with_fapn
