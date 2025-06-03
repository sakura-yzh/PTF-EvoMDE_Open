# !/usr/bin/env sh

### searched + fapn + newcrfs
python ./tools/test.py \
        ./configs/evomde_test_med.py \
        --checkpoint ./weights/colon_best.pth \
        --dataset colon \
        --input_height 256 \
        --input_width 256 \
        --max_depth 20.0 \
        --max_depth_eval 20 \
        --batch_size 16 \
        --num_threads 1 \
        --with_fapn
