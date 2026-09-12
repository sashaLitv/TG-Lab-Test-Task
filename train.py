import argparse
from dotenv import find_dotenv, load_dotenv
from ultralytics import YOLO
import os
import shutil

DATASET_DIR_NAME = "Bird Detection-2"    
DATASET_ARCHIVE_NAME = "Bird.Detection-2"
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

def _prepare_coco_val_labels(dataset_path: str) -> str:
    '''Creates a parallel 'valid_coco' directory with its own images and mapped labels to bypass YOLO's rigid path logic'''
    import glob 
    import shutil

    val_images_dir = os.path.join(dataset_path, "valid", "images")
    val_labels_dir = os.path.join(dataset_path, "valid", "labels")
    
    coco_val_dir = os.path.join(dataset_path, "valid_coco")
    coco_images_dir = os.path.join(coco_val_dir, "images")
    coco_labels_dir = os.path.join(coco_val_dir, "labels")
    
    os.makedirs(coco_images_dir, exist_ok=True)
    os.makedirs(coco_labels_dir, exist_ok=True)

    val_images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        val_images.extend(glob.glob(os.path.join(val_images_dir, ext)))

    for img_path in val_images:
        shutil.copy(img_path, coco_images_dir)
        
        txt_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        orig_txt_path = os.path.join(val_labels_dir, txt_name)
        dest_txt_path = os.path.join(coco_labels_dir, txt_name)
        
        if not os.path.exists(orig_txt_path):
            continue
            
        with open(orig_txt_path, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts:
                parts[0] = "14" 
                new_lines.append(" ".join(parts) + "\n")
                
        with open(dest_txt_path, "w") as f:
            f.writelines(new_lines)
            
    return coco_val_dir
def _create_coco_yaml(dataset_path: str, base_model) -> str:
    '''
        Creates a data_coco.yaml by reading the original data.yaml, preserving its structure (paths, roboflow block), 
        but updating nc to 80, setting COCO names, and pointing labels to labels_coco.
    '''
    import yaml

    coco_yaml_path = os.path.join(dataset_path, "data_coco.yaml")
    orig_yaml_path = os.path.join(dataset_path, "data.yaml")

    config_data = {}
    if os.path.exists(orig_yaml_path):
        with open(orig_yaml_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

    coco_names = base_model.names.copy() if hasattr(base_model, "names") else {i: f"class_{i}" for i in range(80)}

    config_data["nc"] = 80
    config_data["names"] = coco_names
    
    orig_val = config_data.get("val", "../valid/images")
    config_data["val"] = orig_val.replace("valid", "valid_coco")
    
    if "labels" in config_data:
        del config_data["labels"]

    with open(coco_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, sort_keys=False, default_flow_style=False)

    return coco_yaml_path

def evaluate_base_model(dataset_path: str):
    '''
        Main function: checks if COCO-mapped data already exists to avoid redundant processing,
        runs built-in YOLO validation, and outputs real mAP50 and mAP50-95 metrics.
    '''

    base_model = _get_or_download_base_model()
    
    coco_labels_dir = os.path.join(dataset_path, "valid_coco", "labels")
    coco_yaml_path = os.path.join(dataset_path, "data_coco.yaml")

    # skip preparation if both labels_coco directory and data_coco.yaml already exist
    if os.path.exists(coco_labels_dir) and os.path.exists(coco_yaml_path):
        print("\nFound existing COCO-mapped labels and config, skipping preparation.")
    else:
        print("\nPreparing dataset and configuration for COCO evaluation.")
        _prepare_coco_val_labels(dataset_path)
        _create_coco_yaml(dataset_path, base_model)

    print("\nRunning base model evaluation")
    metrics = base_model.val(data=coco_yaml_path, split="val", verbose=True)

    mp = metrics.box.mp
    mr = metrics.box.mr
    
    f1 = 2 * (mp * mr) / (mp + mr + 1e-6)

    print(f"\nBase model Precision: {mp:.4f}")
    print(f"Base model Recall: {mr:.4f}")
    print(f"Base model F1-score: {f1:.4f}")
    print(f"Base model mAP50: {metrics.box.map50:.4f}")
    print(f"Base model mAP50-95: {metrics.box.map:.4f}")
        

def finetune_yolo_model(data_path: str, epochs: int, batch_size: int, img_size: int, patience: int, single_cls: bool):
    '''Performs fine-tuning of the pre-trained YOLO26n model on a custom dataset'''
    base_model = _get_or_download_base_model()

    base_model.train(
        data=data_path, 
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=patience,

        single_cls=single_cls,

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
        help="GitHub repository URL for dataset (default: https://github.com/sashaLitv/TG-Lab-Test-Task/releases/tag/v2.0-bird_dataset)", 
        default=f"https://github.com/sashaLitv/TG-Lab-Test-Task/releases/download/v2.0-bird_dataset/{DATASET_ARCHIVE_NAME}.zip"
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

    args = parser.parse_args()

    # PIPELINE: download dataset -> evaluate base model -> finetune base model (3 classes) -> save best model's weights
    data_path = download_dataset(
        workspace_name=args.workspace,
        project_name=args.project,
        version_num=args.version,
        api_key=args.rf_key,
        repo_url=args.repo_url
    )
    evaluate_base_model(data_path)

    is_single_cls = not args.multi_class
    finetune_yolo_model(
        data_path=os.path.join(data_path, "data.yaml"),
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        patience=args.patience,
        single_cls=is_single_cls
    )
    save_best_weights()