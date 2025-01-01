import pytest
from unittest.mock import mock_open, patch, MagicMock
from phishpedia import PhishpediaWrapper, result_file_write


def test_result_file_write():
    mock = mock_open()
    with patch('builtins.open', mock):
        with open('fake_file.txt', 'w') as f:
            result_file_write(
                f,
                folder="test_folder",
                url="http://example.com",
                phish_category=1,
                pred_target="example",
                matched_domain="example.com",
                siamese_conf=0.95,
                logo_recog_time=0.1234,
                logo_match_time=0.5678
            )
    mock.assert_called_once_with('fake_file.txt', 'w')
    handle = mock()
    handle.write.assert_any_call("test_folder\t")
    handle.write.assert_any_call("http://example.com\t")
    handle.write.assert_any_call("1\t")
    handle.write.assert_any_call("example\t")
    handle.write.assert_any_call("example.com\t")
    handle.write.assert_any_call("0.95\t")
    handle.write.assert_any_call("0.1234\t")
    handle.write.assert_any_call("0.5678\n")


@pytest.fixture
def phishpedia_wrapper():
    with patch('phishpedia.load_config') as mock_load_config:
        mock_load_config.return_value = (
            MagicMock(),  # ELE_MODEL
            0.8,           # SIAMESE_THRE
            MagicMock(),  # SIAMESE_MODEL
            [],            # LOGO_FEATS
            [],            # LOGO_FILES
            'path/to/domain_map'  # DOMAIN_MAP_PATH
        )
        wrapper = PhishpediaWrapper()
    return wrapper


@patch('phishpedia.pred_rcnn')
@patch('phishpedia.vis')
@patch('phishpedia.check_domain_brand_inconsistency')
def test_test_orig_phishpedia_no_logo(mock_check_inconsistency, mock_vis, mock_pred_rcnn, phishpedia_wrapper):
    # 设置 mock 返回值
    mock_pred_rcnn.return_value = None
    mock_vis.return_value = "visualization_image"

    # 调用方法
    result = phishpedia_wrapper.test_orig_phishpedia(
        url="http://example.com",
        screenshot_path="path/to/shot.png",
        html_path="path/to/html.txt"
    )

    # 断言返回值
    assert result[0] == 0  # phish_category
    assert result[1] is None  # pred_target
    assert result[2] is None  # matched_domain
    assert result[3] == "visualization_image"  # plotvis
    assert result[4] is None  # siamese_conf
    assert result[5] is None  # pred_boxes
    assert result[6] >= 0  # logo_recog_time
    assert result[7] == 0  # logo_match_time

    # 断言被调用
    mock_pred_rcnn.assert_called_once_with(im="path/to/shot.png", predictor=phishpedia_wrapper.ELE_MODEL)
    mock_vis.assert_called_once_with("path/to/shot.png", None)
    mock_check_inconsistency.assert_not_called()
