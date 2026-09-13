import argparse
from ultralytics import YOLO
import os
import datetime

def predict_video(model_path: str, input_path: str, save_dir: str, **kwargs):
    '''Predicts objects in the video using the YOLO model and saves the output to the specified directory'''

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} is required.")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path {input_path} is required.")

    os.makedirs(save_dir, exist_ok=True)
    model = YOLO(model_path)

    if 'output_file' in kwargs and kwargs['output_file']:
        dynamic_name = kwargs['output_file']
    else:   
        input_name = os.path.basename(input_path)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dynamic_name = f"{input_name}_{timestamp}"

    results=model.predict(
        source=input_path, 
        save=True, 
        save_dir=os.path.abspath(save_dir),
        name=dynamic_name,
        show_conf=True,
        show_labels=True,
        verbose=False,
        stream=True,
        conf=kwargs.get('confidence_threshold', 0.5),
        iou=kwargs.get('iou_threshold', 0.7),
        vid_stride=kwargs.get('video_stride', 1)
    )

    for _ in results:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO video tracking with ByteTrack and count unique birds.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--output_dir", type=str, default="examples", help="Directory to save output video (default: 'examples')")
    parser.add_argument("--output_file", type=str, default=None, help="Name of the output video file (default: None, will use input name with timestamp)")
    parser.add_argument("--model", type=str, default="weights/best.pt", help="Path to trained YOLO model weights")
    parser.add_argument("--stride", type=int, default=1, help="Process every nth frame (default: 1)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--iou", type=float, default=0.7, help="IOU threshold for tracking (default: 0.7)")
    args = parser.parse_args()

    predict_video(
        model_path=args.model,
        input_path=args.input_file,
        save_dir=args.output_dir,
        stride=args.stride,
        output_file=args.output_file,
        conf=args.conf,
    )

