#!/bin/bash

cd ./mmdetection-0.6.0 || exit
pip install -v -e .

cd mmdet/ops || exit
for mod in dcn nms roi_align roi_pool sigmoid_focal_loss; do
  cd "$mod" && python setup.py build_ext --inplace && cd ..
done

echo "✅ Build complete."
