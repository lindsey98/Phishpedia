from logo_matching import load_model_weights
import torch
import onnxruntime
from memory_profiler import profile
import numpy as np
from train.siamese.data import GetLoader
import torchvision as tv
from tqdm import tqdm
from train.siamese import mobilenet
from collections import OrderedDict


def load_and_prepare_model(weights_path, num_classes=270, device='cuda'):
    """
    Load and prepare the model with the specified weights.

    :param weights_path: Path to the model weights file.
    :param num_classes: Number of output classes for the model.
    :param device: Device to load the model on ('cuda' or 'cpu').
    :return: Loaded model.
    """
    model = mobilenet.mobilenet_v2(num_classes=num_classes)

    # Load weights safely
    try:
        weights = torch.load(weights_path, map_location='cpu')
        state_dict = weights.get('model', weights)
        new_state_dict = OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items())
        model.load_state_dict(new_state_dict)
    except KeyError:
        raise KeyError("Failed to load state dict")

    model.to(device)
    model.eval()
    return model


def export_to_onnx(model, export_path, device, input_size=(1, 3, 64, 64), opset_version=10):
    """
    Export the PyTorch model to ONNX format.

    :param model: PyTorch model to be exported.
    :param export_path: Path to save the ONNX model.
    :param input_size: Size of the input tensor.
    :param opset_version: ONNX opset version.
    """
    x = torch.randn(*input_size, requires_grad=True).to(device)
    torch.onnx.export(
        model,
        x,
        export_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['classification', 'features'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model exported to {export_path}")


def create_onnx_session(onnx_model_path, use_cuda=True):
    """
    Create an ONNX runtime session.

    :param onnx_model_path: Path to the ONNX model file.
    :param use_cuda: Whether to use CUDA for inference.
    :return: ONNX runtime session.
    """
    sess_options = onnxruntime.SessionOptions()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    return onnxruntime.InferenceSession(onnx_model_path, sess_options, providers=providers)


def topk_numpy(output, target, ks=(1,)):
    """
    Returns one boolean vector for each k, whether the target is within the output's top-k.
    """
    topk_indices = np.argsort(output, axis=1)[:, -max(ks):][:, ::-1]
    correct = np.equal(topk_indices, np.array([target])[:, None])
    return [np.any(correct[:, :k], axis=1) for k in ks]


def run_eval(session, dataset):
    """
    Run evaluation using the ONNX model session on a given dataset.

    :param session: ONNX runtime session.
    :param dataset: Dataset to evaluate on.
    """
    all_top1, all_top5 = [], []
    for it in tqdm(range(len(dataset))):
        img, label = dataset[it]
        logits = session.run(['classification'], {'input': img.unsqueeze(0).numpy()})[0]
        top1, top5 = topk_numpy(logits, label, ks=(1, 5))
        all_top1.extend(top1)
        all_top5.extend(top5)

    print("Top-1 Accuracy: {:.2%}".format(np.mean(all_top1)))
    print("Top-5 Accuracy: {:.2%}".format(np.mean(all_top5)))


def prepare_validation_dataset(data_root, data_list, label_dict):
    """
    Prepare the validation dataset.

    :param data_root: Path to the data root directory.
    :param data_list: Path to the data list file.
    :param label_dict: Path to the label dictionary file.
    :return: Prepared validation dataset.
    """
    val_tx = tv.transforms.Compose([
        tv.transforms.Resize((64, 64)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return GetLoader(data_root=data_root, data_list=data_list, label_dict=label_dict, transform=val_tx)


if __name__ == '__main__':
    # Set paths
    weights_path = "./runs/targetlist_finetuned/bit.pth.tar"
    onnx_model_path = "./models/mobilenetv2_64.onnx"
    data_root = './models/expand_targetlist'
    data_list = './datasets/siamese_training/test_targets.txt'
    label_dict = './datasets/siamese_training/target_dict.pkl'

    # Load and prepare model
    model = load_and_prepare_model(weights_path, num_classes=270, device='cuda')

    # Export the model to ONNX
    export_to_onnx(model, onnx_model_path, device='cuda', input_size=(1, 3, 64, 64))

    # Create ONNX session
    session = create_onnx_session(onnx_model_path, use_cuda=True)

    # Prepare validation dataset
    valid_set = prepare_validation_dataset(data_root, data_list, label_dict)

    # Run evaluation
    run_eval(session, valid_set)

    ##
    # top1 90.37%,
    # top5 96.15%