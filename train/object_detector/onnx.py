from ultralytics import YOLO

def load_model(model_path):
    """Load the YOLO model from a given path."""
    return YOLO(model_path)

def export_model_to_onnx(model):
    """
    Export the YOLO model to ONNX format.

    :param model: The YOLO model object to export.
    :param export_path: Path where the ONNX model will be saved.
    """
    model.export(format="onnx")

def evaluate_model(model_path, data_path, eval_name="onnx_eval", batch_size=16, device="cpu", img_size=320):
    """
    Evaluate the ONNX model on the validation dataset.

    :param model_path: Path to the ONNX model.
    :param data_path: Path to the dataset configuration file.
    :param eval_name: Name for the evaluation run.
    :param batch_size: Batch size for evaluation.
    :param device: Device to use for evaluation, e.g., "cpu" or "cuda".
    :param img_size: Image size to use during evaluation.
    :return: Results of the evaluation.
    """
    onnx_model = YOLO(model_path)
    results = onnx_model.val(
        name=eval_name,
        data=data_path,
        batch=batch_size,
        device=device,
        imgsz=img_size,
    )
    return results

if __name__ == "__main__":
    # Paths
    yolo_pt_model_path = "./runs/detect/yolo_middle_640/weights/best.pt"
    data_config_path = "./datasets/object_detector_training/config.yaml"

    # Load the YOLOv8 model
    model = load_model(yolo_pt_model_path)

    # Export the model to ONNX format
    export_model_to_onnx(model)

    # # Evaluate the exported ONNX model
    results = evaluate_model(
        model_path=yolo_pt_model_path.replace('.pt', '.onnx'),
        data_path=data_config_path,
        batch_size=16,
        device="cpu",
        img_size=640
    )

    # Print results
    print(results)

#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95):
#                    all       1561       3744      0.828       0.75      0.809      0.522