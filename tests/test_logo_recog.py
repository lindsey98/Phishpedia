import pytest
from unittest.mock import patch, MagicMock
from logo_recog import pred_rcnn, config_rcnn, vis
import torch
import numpy as np


def test_pred_rcnn_image_not_found():
    with patch('phishpedia.cv2.imread', return_value=None) as mock_imread:
        result = pred_rcnn('path/to/nonexistent_image.png', 'dummy_predictor')
        mock_imread.assert_called_once_with('path/to/nonexistent_image.png')
        assert result is None


def test_pred_rcnn_image_with_alpha_channel():
    # 创建一个模拟的图像（带有 alpha 通道）
    mock_image = MagicMock()
    mock_image.shape = (100, 100, 4)  # 包含 alpha 通道

    with patch('logo_recog.cv2.imread', return_value=mock_image) as mock_imread:
        # 模拟 cv2.cvtColor
        with patch('logo_recog.cv2.cvtColor', return_value=MagicMock()) as mock_cvtColor:
            # 模拟 predictor 的输出
            mock_instances = MagicMock()
            mock_instances.pred_classes = MagicMock()
            mock_instances.pred_classes.__getitem__.return_value = MagicMock()
            mock_instances.pred_boxes = MagicMock()
            mock_instances.pred_boxes.__getitem__.return_value = MagicMock()
            mock_predictor = MagicMock(return_value={'instances': mock_instances})
            
            # 设定 pred_classes 为 1 和 0
            mock_instances.pred_classes == 1
            mock_instances.pred_classes.__eq__.return_value = MagicMock()
            mock_instances.pred_boxes.__getitem__.return_value = MagicMock()
            
            result = pred_rcnn('path/to/image_with_alpha.png', mock_predictor)

            mock_imread.assert_called_once_with('path/to/image_with_alpha.png')
            mock_cvtColor.assert_called_once()
            mock_predictor.assert_called_once()
            assert result
            # 根据 mock 的返回值，断言结果
            # 由于我们没有具体的返回值设置，这里仅断言返回值被调用
            # 在实际测试中，您应根据具体的 mock 返回值进行断言


def test_pred_rcnn_image_notwith_alpha_channel():
    # 创建一个模拟的图像（不带 alpha 通道）
    mock_image = MagicMock()
    mock_image.shape = (100, 100, 3)  # 包含 alpha 通道

    with patch('logo_recog.cv2.imread', return_value=mock_image) as mock_imread:
        # 模拟 predictor 的输出
        mock_instances = MagicMock()
        mock_instances.pred_classes = MagicMock()
        mock_instances.pred_classes.__getitem__.return_value = MagicMock()
        mock_instances.pred_boxes = MagicMock()
        mock_instances.pred_boxes.__getitem__.return_value = MagicMock()
        mock_predictor = MagicMock(return_value={'instances': mock_instances})
        
        # 设定 pred_classes 为 1 和 0
        mock_instances.pred_classes == 1
        mock_instances.pred_classes.__eq__.return_value = MagicMock()
        mock_instances.pred_boxes.__getitem__.return_value = MagicMock()
        
        result = pred_rcnn('path/to/image_with_alpha.png', mock_predictor)
        
        mock_imread.assert_called_once_with('path/to/image_with_alpha.png')
        mock_predictor.assert_called_once()
        assert result


def test_config_rcnn_cpu():
    """
    测试 config_rcnn 函数在 CUDA 不可用时的行为。
    """
    with patch('logo_recog.get_cfg') as mock_get_cfg:
        # 创建一个模拟的 cfg 对象
        mock_cfg = MagicMock()
        mock_get_cfg.return_value = mock_cfg

        with patch('logo_recog.DefaultPredictor') as mock_default_predictor:
            # 模拟 torch.cuda.is_available() 返回 False
            with patch('logo_recog.torch.cuda.is_available', return_value=False):
                # 调用 config_rcnn 函数
                predictor = config_rcnn('path/to/cfg.yaml', 'path/to/weights.pth', 0.5)

                # 断言 get_cfg 被调用一次
                mock_get_cfg.assert_called_once()

                # 断言 merge_from_file 被调用并传入正确的路径
                mock_cfg.merge_from_file.assert_called_once_with('path/to/cfg.yaml')

                # 断言模型权重和置信度阈值被正确设置
                assert mock_cfg.MODEL.WEIGHTS == 'path/to/weights.pth'
                assert mock_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST == 0.5

                # 断言设备被设置为 'cpu'
                assert mock_cfg.MODEL.DEVICE == 'cpu'

                # 断言 DefaultPredictor 被正确初始化
                mock_default_predictor.assert_called_once_with(mock_cfg)

                # 断言返回的 predictor 是 DefaultPredictor 的返回值
                assert predictor == mock_default_predictor.return_value


def test_config_rcnn_gpu():
    """
    测试 config_rcnn 函数在 CUDA 可用时的行为。
    """
    with patch('logo_recog.get_cfg') as mock_get_cfg:
        # 创建一个模拟的 cfg 对象
        mock_cfg = MagicMock()
        mock_get_cfg.return_value = mock_cfg

        with patch('logo_recog.DefaultPredictor') as mock_default_predictor:
            # 模拟 torch.cuda.is_available() 返回 True
            with patch('logo_recog.torch.cuda.is_available', return_value=True):
                # 调用 config_rcnn 函数
                predictor = config_rcnn('path/to/cfg.yaml', 'path/to/weights.pth', 0.7)

                # 断言 get_cfg 被调用一次
                mock_get_cfg.assert_called_once()

                # 断言 merge_from_file 被调用并传入正确的路径
                mock_cfg.merge_from_file.assert_called_once_with('path/to/cfg.yaml')

                # 断言模型权重和置信度阈值被正确设置
                assert mock_cfg.MODEL.WEIGHTS == 'path/to/weights.pth'
                assert mock_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST == 0.7

                # 断言 DefaultPredictor 被正确初始化
                mock_default_predictor.assert_called_once_with(mock_cfg)

                # 断言返回的 predictor 是 DefaultPredictor 的返回值
                assert predictor == mock_default_predictor.return_value


@pytest.fixture
def dummy_image():
    """
    创建一个虚拟图像(100x100 像素, 3 通道)，填充为白色。
    """
    return np.ones((100, 100, 3), dtype=np.uint8) * 255


def test_vis_pred_boxes_none(capsys, dummy_image):
    img_path = "dummy_path.jpg"
    with patch('cv2.imread', return_value=dummy_image.copy()) as mock_imread:
        result = vis(img_path, None)
        mock_imread.assert_called_once_with(img_path)
        # 捕获打印输出
        captured = capsys.readouterr()
        assert "Pred_boxes is None or the length of pred_boxes is 0" in captured.out
        # 检查返回的图像与原始图像相同
        np.testing.assert_array_equal(result, dummy_image)


def test_vis_pred_boxes_empty(capsys, dummy_image):
    img_path = "dummy_path.jpg"
    empty_boxes = np.empty((0, 4))
    with patch('cv2.imread', return_value=dummy_image.copy()) as mock_imread:
        result = vis(img_path, empty_boxes)
        mock_imread.assert_called_once_with(img_path)
        # 捕获打印输出
        captured = capsys.readouterr()
        assert "Pred_boxes is None or the length of pred_boxes is 0" in captured.out
        # 检查返回的图像与原始图像相同
        np.testing.assert_array_equal(result, dummy_image)


@pytest.mark.parametrize("pred_boxes,expected_colors", [
    # 单个框：第一个框应为青色 (255, 255, 0)
    (torch.tensor([[10, 10, 50, 50]]), [(255, 255, 0)]),
    # 多个框：第一个青色，其他为浅绿色
    (torch.tensor([[10, 10, 50, 50], [60, 60, 90, 90]]), [(255, 255, 0), (36, 255, 12)]),
    # 使用 numpy 数组的多个框
    (np.array([[5, 5, 20, 20], [30, 30, 40, 40], [50, 50, 70, 70]]), [(255, 255, 0), (36, 255, 12), (36, 255, 12)]),
])
def test_vis_draw_rectangles(dummy_image, pred_boxes, expected_colors):
    img_path = "dummy_path.jpg"
    with patch('cv2.imread', return_value=dummy_image.copy()) as mock_imread:
        result = vis(img_path, pred_boxes)
        mock_imread.assert_called_once_with(img_path)
        # 通过检查特定位置的像素颜色来验证是否绘制了矩形
        for idx, box in enumerate(pred_boxes):
            color = expected_colors[idx]
            # 检查矩形的左上角像素
            x1, y1 = int(box[0]), int(box[1])
            # 确保坐标在图像范围内
            x1 = min(max(x1, 0), dummy_image.shape[1] - 1)
            y1 = min(max(y1, 0), dummy_image.shape[0] - 1)
            assert tuple(result[y1, x1]) == color, f"矩形 {idx} 未使用预期颜色 {color} 绘制"
            # 同样，检查矩形的右下角像素
            x2, y2 = int(box[2]), int(box[3])
            x2 = min(max(x2, 0), dummy_image.shape[1] - 1)
            y2 = min(max(y2, 0), dummy_image.shape[0] - 1)
            assert tuple(result[y2, x2]) == color, f"矩形 {idx} 未使用预期颜色 {color} 绘制"


def test_vis_invalid_image_path(capsys):
    img_path = "invalid_path.jpg"
    with patch('cv2.imread', return_value=None) as mock_imread:
        result = vis(img_path, None)
        mock_imread.assert_called_once_with(img_path)
        # 由于 pred_boxes 为 None，应打印消息
        captured = capsys.readouterr()
        assert "Pred_boxes is None or the length of pred_boxes is 0" in captured.out
        # 函数返回 None，因为 cv2.imread 返回了 None
        assert result is None


def test_vis_pred_boxes_not_tensor_or_numpy(dummy_image):
    img_path = "dummy_path.jpg"
    invalid_pred_boxes = [[10, 10, 50, 50]]  # 使用列表而非 torch.Tensor 或 numpy.ndarray
    with pytest.raises(AttributeError):
        # 因为函数尝试对列表调用 .numpy()，应引发 AttributeError
        vis(img_path, invalid_pred_boxes)
