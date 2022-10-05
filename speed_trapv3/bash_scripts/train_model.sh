set +e
echo taining the model
speed-trapv3 keypoints train-model "/root/.darwin/datasets/sparrow-computing/kj_speedtrap/releases/backtirev1.0/annotations" "/root/code/speed-trapv3/data/datasets/annotations"
echo Launching tensorboard...