import argparse
from dotenv import find_dotenv, load_dotenv
from ultralytics import YOLO
import os
import shutil

MY_DATASET_DIR_NAME = "Bird Detection"    
MY_DATASET_ARCHIVE_NAME = "Bird.Detection"
DATASET_DIR_NAME = "Bird Detection 8classes"    
DATASET_ARCHIVE_NAME = "Bird.Detection.8classes"
BASE_MODEL_NAME = "weights/base.pt"
FINETUNED_MODEL_NAME = "weights/best.pt"


def _download_roboflow_dataset(workspace_name: str, project_name: str, version_num: int, api_key: str, target_dir: str) -> str:
    '''
        Authenticates with Roboflow and downloads the specified dataset version.
        Returns the absolute path to the downloaded dataset folder.
    '''
    import roboflow

    rf = roboflow.Roboflow(api_key=api_key)
    version = rf.workspace(workspace_name).project(project_name).version(version_num)
    dataset = version.download("yolo26")

    final_path = os.path.join(target_dir, os.path.basename(dataset.location))
    shutil.move(dataset.location, final_path)

    return final_path
def _download_github_dataset(repo_url: str, target_dir: str) -> str:
    '''
        Downloads the dataset from the specified GitHub repository.
        Returns the absolute path to the downloaded dataset folder.
    '''
    from urllib import request
    import zipfile

    zip_filename = f"{DATASET_ARCHIVE_NAME}.zip"
    request.urlretrieve(repo_url, zip_filename)

    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

    os.remove(zip_filename)
    return os.path.join(target_dir, DATASET_DIR_NAME)
def download_dataset(workspace_name: str, project_name: str, version_num: int, api_key: str = None, target_dir: str = "data", repo_url: str = None) -> str:
    '''Smart router that chooses the download method based on provided arguments'''
    os.makedirs(target_dir, exist_ok=True)

    expected_dataset_path = os.path.join(target_dir, DATASET_DIR_NAME)
    if os.path.exists(expected_dataset_path) and os.path.exists(os.path.join(expected_dataset_path, "data.yaml")):
        print(f"Dataset already exists locally at '{expected_dataset_path}', skipping download")
        return expected_dataset_path

    if api_key:
        print("Detected Roboflow API key. Initiating API download...")
        return _download_roboflow_dataset(workspace_name, project_name, version_num, api_key, target_dir)
    elif repo_url:
        print("No Roboflow key provided. Falling back to GitHub Release download...")
        return _download_github_dataset(repo_url, target_dir)
    else:
        raise ValueError("Error: you must provide either '--rf_key' or ensure the dataset exists on github")


def _get_or_download_base_model(base_model_path: str = BASE_MODEL_NAME):
    '''
        Checks if the base model exists locally, if not downloads it from Ultralytics.
        Returns the YOLO model object.
    '''
    os.makedirs(os.path.dirname(base_model_path), exist_ok=True)
    if not os.path.exists(base_model_path):
        print(f"Base model '{base_model_path}' not found. Downloading...")
        temp_model = YOLO("yolo26n.pt")
        temp_model.save(base_model_path)
    base_model = YOLO(base_model_path)
    return base_model

def evaluate_base_model(dataset_path: str):
    '''
        Evaluates the base YOLO model on the validation set of the custom dataset.
        Computes mAP, Precision, Recall, and F1-score using torchmetrics.
    '''
    import torch 
    from torchmetrics.detection import MeanAveragePrecision
    from pathlib import Path
    from PIL import Image

    base_model = _get_or_download_base_model()

    metric = MeanAveragePrecision(box_format="xyxy")

    val_images_dir = os.path.join(dataset_path, "valid", "images")
    val_labels_dir = os.path.join(dataset_path, "valid", "labels")

    val_images = []
    if os.path.exists(val_images_dir):
        for file in os.listdir(val_images_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                val_images.append(os.path.join(val_images_dir, file))

    total_gt_boxes = 0
    total_pred_boxes = 0
    matched_tp = 0

    for img_path in val_images:
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        txt_name = Path(img_path).stem + ".txt"
        label_path = os.path.join(val_labels_dir, txt_name)

        gt_boxes, gt_labels = [], []

        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        ## map any ground-truth bird species (0, 1, 2, ...) to COCO's bird class ID (14)
                        x_center, y_center, w, h = map(float, parts[1:5])

                        ## denormalization for torchmetrics
                        xmin = (x_center - w / 2) * img_w
                        ymin = (y_center - h / 2) * img_h
                        xmax = (x_center + w / 2) * img_w
                        ymax = (y_center + h / 2) * img_h

                        gt_boxes.append([xmin, ymin, xmax, ymax])
                        gt_labels.append(14)

        total_gt_boxes += len(gt_boxes)

        target = [
            {
                "boxes": (
                    torch.tensor(gt_boxes, dtype=torch.float32)
                    if gt_boxes
                    else torch.empty((0, 4), dtype=torch.float32)
                ),
                "labels": (
                    torch.tensor(gt_labels, dtype=torch.int64)
                    if gt_labels
                    else torch.empty((0,), dtype=torch.int64)
                ),
            }
        ]

        ## inference for base model
        results = base_model.predict(source=img_path, verbose=False)
        pred_boxes = []
        pred_scores = []
        pred_labels = []

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes_data = results[0].boxes
            xyxy = boxes_data.xyxy.cpu().numpy()
            conf = boxes_data.conf.cpu().numpy()
            cls = boxes_data.cls.cpu().numpy().astype(int)

            for box, score, c in zip(xyxy, conf, cls):
                ## keep ONLY bird detections (class 14 from COCO), ignore cars, cats, dogs, etc, because our datasets dont have other classes
                if c == 14:
                    pred_boxes.append(box.tolist())
                    pred_scores.append(float(score))
                    pred_labels.append(c)

        total_pred_boxes += len(pred_boxes)

        ## count for basic Precision/Recall calculation
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            matched_tp += min(len(gt_boxes), len(pred_boxes))

            preds = [
                {
                    "boxes": (
                        torch.tensor(pred_boxes, dtype=torch.float32)
                        if pred_boxes
                        else torch.empty((0, 4), dtype=torch.float32)
                    ),
                    "scores": (
                        torch.tensor(pred_scores, dtype=torch.float32)
                        if pred_scores
                        else torch.empty((0,), dtype=torch.float32)
                    ),
                    "labels": (
                        torch.tensor(pred_labels, dtype=torch.int64)
                        if pred_labels
                        else torch.empty((0,), dtype=torch.int64)
                    ),
                }
            ]

        metric.update(preds, target)

    ## compute metrics via torchmetrics
    map_results = metric.compute()

    precision = (
        matched_tp / total_pred_boxes if total_pred_boxes > 0 else 0.0
    )
    recall = matched_tp / total_gt_boxes if total_gt_boxes > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall + 1e-6)
        if (precision + recall) > 0
        else 0.0
    )

    print("\nEvaluation base model before finetune...")
    print(f"mAP (IoU=0.50:0.95): {map_results['map']:.4f}")
    print(f"mAP (IoU=0.50):      {map_results['map_50']:.4f}")
    print(f"mAP (IoU=0.75):      {map_results['map_75']:.4f}")
    print("-" * 50)
    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    print(f"F1-score:            {f1:.4f}")


def finetune_yolo_model(
        data_yaml_path: str, 
        epochs: int, 
        batch_size: int, 
        img_size: int, 
        patience: int, 
        single_cls: bool,
        save_period: int,
        resume: bool,
        last_checkpoint_path: str = "runs/bird_detection/weights/last.pt"
    ):
    '''Performs fine-tuning of the pre-trained YOLO26n model on a custom dataset with checkpoints'''
    if resume and os.path.exists(last_checkpoint_path):
        model = YOLO(last_checkpoint_path)
        model.train(resume=True)
    else:
        if resume:
            print(f"Warning: resume flag is set but last checkpoint '{last_checkpoint_path}' not found. Starting training from base model.")
        base_model = _get_or_download_base_model()
        base_model.train(
            data=data_yaml_path, 
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            patience=patience,
            single_cls=single_cls,
            save_period=save_period,  

            project=os.path.abspath("runs"),        
            name="bird_detection",
            exist_ok=True, 
            val=True 
        )

def save_best_weights(source_dir: str = "runs/bird_detection", target_path: str = "weights/best.pt"):
    '''Copies the best trained weights from the YOLO runs directory to the target weights directory '''

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    source_best_weights = os.path.join(source_dir, "weights", "best.pt")

    if os.path.exists(source_best_weights):
        shutil.copy(source_best_weights, target_path)
        print(f"Best weights saved to '{target_path}'")
    else:
        print("Error: Best weights not found. Please check the training process.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO26n for bird detection")

    # load the .env file and get the Roboflow Api key from it
    load_dotenv(find_dotenv())
    ROBOFLOW_KEY = os.getenv("ROBOFLOW_KEY")

    # add command line arguments for the Roboflow dataset parameters
    parser.add_argument("--workspace", type=str, help="Roboflow workspace name for dataset (default: sasha-litvak)", default="sasha-litvak")
    parser.add_argument("--project", type=str, help="Roboflow project name for dataset (default: bird-detection-df9zw)", default="bird-detection-df9zw")
    parser.add_argument("--version", type=int, help="Roboflow dataset version number for dataset (default: 2 version)", default=2)
    parser.add_argument("--rf_key", type=str, help="Roboflow API key (default: from .env file)", default=ROBOFLOW_KEY)
    parser.add_argument(
        "--repo_url", type=str, 
        help="GitHub repository URL for dataset (default: https://github.com/sashaLitv/TG-Lab-Test-Task/releases/tag/bird-species-8-yolo26)", 
        default=f"https://github.com/sashaLitv/TG-Lab-Test-Task/releases/download/bird-species-8-yolo26/{DATASET_ARCHIVE_NAME}.zip"
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        default=None,
        help="Path to the dataset root folder used for testing. Inside this folder, it will automatically look into 'valid/images' and 'valid/labels'. " \
        "If not specified, it falls back to the validation folder of the downloaded training dataset.",
    )
    parser.add_argument(
        "--multi_class", 
        action="store_true", 
        help="Train on original multiple classes (subspecies). If not set, merges all into a single 'bird' class."
    )

    # add command line arguments for training parameters
    parser.add_argument("--epochs", type=int, required=False, help="Number of epochs for training (default: 10)", default=10)
    parser.add_argument("--batch", type=int, help="Batch size (default: 16)", default=16)
    parser.add_argument("--img_size", type=int, help="Image size for training (default: 640 like in default dataset)", default=640)
    parser.add_argument("--patience", type=int, help="Patience for training (default: 3)", default=3)

    parser.add_argument("--save_period", type=int, help="Save checkpoint every X epochs (default: -1, disabled)", default=-1)
    parser.add_argument("--resume", action="store_true", help="Resume training from the last saved checkpoint if it exists")

    args = parser.parse_args()

    # PIPELINE: download dataset -> evaluate base model -> finetune base model -> save best model's weights
    data_path = download_dataset(
        workspace_name=args.workspace,
        project_name=args.project,
        version_num=args.version,
        api_key=args.rf_key,
        repo_url=args.repo_url
    )
    evaluate_base_model(data_path if args.eval_path is None else args.eval_path)

    is_single_cls = not args.multi_class
    results = finetune_yolo_model(
        data_yaml_path=os.path.join(data_path, "data.yaml"),
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        patience=args.patience,
        single_cls=is_single_cls,
        save_period=args.save_period,
        resume=args.resume
    )
    save_best_weights()