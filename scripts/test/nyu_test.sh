# !/usr/bin/env sh

### searched + fapn + newcrfs
python ./tools/test.py \
        ./configs/evomde_test_nyu.py \
        --checkpoint ./weights/nyu_best.pth \
        --dataset nyu \
        --input_height 480 \
        --input_width 640 \
        --max_depth 10.0 \
        --max_depth_eval 10 \
        --batch_size 1 \
        --data_path_eval /data/dataset/NYU_Depth_V2/test/ \
        --gt_path_eval /data/dataset/NYU_Depth_V2/test/ \
        --filenames_file_eval data_splits/nyudepthv2_test_files_with_gt.txt \
        --num_threads 1 \
        --eigen_crop \
        --with_fapn 
