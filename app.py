import argparse
import os
import re
import time
import cv2

import torch
import pandas as pd
from kernel_utils import VideoReader, FaceExtractor, confident_strategy, predict_on_video_set
from training.zoo.classifiers import DeepFakeClassifier
import gradio as gr


weights_dir = "C:/Users/Green/Desktop/VESSL/dfdc_deepfake_challenge/Trained_model"
models_dir = ["final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23"]
test_dir = "C:/Users/Green/Desktop/VESSL/dfdc_deepfake_challenge/test_videos"


def deepfakeclassifier(potential_test_video, option):
    if option == 'Original':
        weights_dir = "C:/Users/Green/Desktop/VESSL/dfdc_deepfake_challenge/Trained_model"
        models_dir = ["final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23"]
    else:
        weights_dir = "D:/University And Papers/VESSL/dfdc_deepfake_challenge/data_root/weights"
        models_dir = ["classifier_DeepFakeClassifier_tf_efficientnet_b7_ns_1_last"]

    parts = potential_test_video.split("\\")
    test_videos = [parts[-1]]
    parts[0] += "\\"
    test_dir = parts[:-1]
    test_dir = os.path.join(*test_dir)


    models = []
    model_paths = [os.path.join(weights_dir, model) for model in models_dir]
    for path in model_paths:
        model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to("cuda")
        print("loading state dict {}".format(path))
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=True)
        model.eval()
        del checkpoint
        models.append(model.half())

    frames_per_video = 32
    video_reader = VideoReader()
    video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn)
    input_size = 380
    strategy = confident_strategy
    stime = time.time()

    print("Predicting {} videos".format(len(test_videos)))
    predictions = predict_on_video_set(face_extractor=face_extractor, input_size=input_size, models=models,
                                       strategy=strategy, frames_per_video=frames_per_video, videos=test_videos,
                                       num_workers=6, test_dir=test_dir)

    print("Elapsed:", time.time() - stime)
    return "This video is FAKE with {} probability!".format(predictions[0])

demo = gr.Interface(fn=deepfakeclassifier, inputs=[gr.Video(), 
                    gr.Radio(["Original", "Custom"])] ,outputs="text", description="Original option uses the trained weights of the winning idea. Custom is my trained \
                        network. Original optional performs better as it uses much more data for training!")

demo.launch()