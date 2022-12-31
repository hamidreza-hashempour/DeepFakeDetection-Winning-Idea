python ./training/pipelines/train_classifier.py --data-dir "./data_root" \
 --config "./configs/b7.json"\
 --output-dir "./data_root/weights"\
 --resume "./data_root/weights/classifier_DeepFakeClassifier_tf_efficientnet_b7_ns_1_last" \
 --logdir "./logs"
