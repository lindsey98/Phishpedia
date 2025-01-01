from unittest.mock import mock_open, patch, MagicMock
import pickle
# 从 logo_matching 模块导入被测试的函数和需要模拟的函数
from logo_matching import check_domain_brand_inconsistency

# 示例输入数据
logo_boxes_sample = [
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 100, 110, 120],
    [130, 140, 150, 160]
]

domain_map_sample = {"example.com": "brandA"}


# 测试用例 1：无 logo 框
def test_no_logo_boxes():
    with patch("builtins.open", mock_open(read_data=pickle.dumps(domain_map_sample))) as mock_file, \
         patch("logo_matching.tldextract.extract") as mock_extract, \
         patch("logo_matching.pred_brand") as mock_pred_brand, \
         patch("logo_matching.brand_converter") as mock_brand_converter:
        
        # 设置 tldextract.extract 返回值
        mock_extract.return_value = MagicMock(domain="example", suffix="com")
        
        # 设置 brand_converter 返回值
        mock_brand_converter.return_value = None
        
        result = check_domain_brand_inconsistency(
            logo_boxes=[],
            domain_map_path="dummy_path",
            model=None,
            logo_feat_list=None,
            file_name_list=None,
            shot_path="",
            url="http://www.example.com",
            similarity_threshold=0.8
        )
        
        # 断言返回值
        assert result == (None, None, None, None)
        
        # 断言文件打开
        mock_file.assert_called_once_with("dummy_path", 'rb')
        
        # 断言 pred_brand 未被调用
        mock_pred_brand.assert_not_called()


# 测试用例 2：有 logo 框但 pred_brand 不返回匹配
def test_logo_boxes_no_match():
    with patch("builtins.open", mock_open(read_data=pickle.dumps(domain_map_sample))) as mock_file, \
         patch("logo_matching.tldextract.extract") as mock_extract, \
         patch("logo_matching.pred_brand") as mock_pred_brand, \
         patch("logo_matching.brand_converter") as mock_brand_converter:
        
        mock_file = mock_file
        # 设置 tldextract.extract 返回值
        mock_extract.return_value = MagicMock(domain="example", suffix="com")
        
        # 设置 pred_brand 返回值为空
        mock_pred_brand.return_value = (None, None, None)
        
        # 设置 brand_converter 返回值
        mock_brand_converter.return_value = None
        
        result = check_domain_brand_inconsistency(
            logo_boxes=logo_boxes_sample,
            domain_map_path="dummy_path",
            model=None,
            logo_feat_list=None,
            file_name_list=None,
            shot_path="",
            url="http://www.example.com",
            similarity_threshold=0.8
        )
        
        # 断言返回值
        assert result == (None, None, None, None)
        
        # 断言 pred_brand 被调用且次数等于 logo_boxes 数量或 topk
        assert mock_pred_brand.call_count == min(len(logo_boxes_sample), 3)


# 测试用例 3：有 logo 框且域名一致
def test_logo_boxes_domain_consistent():
    with patch("builtins.open", mock_open(read_data=pickle.dumps(domain_map_sample))) as mock_file, \
         patch("logo_matching.tldextract.extract") as mock_extract, \
         patch("logo_matching.pred_brand") as mock_pred_brand, \
         patch("logo_matching.brand_converter") as mock_brand_converter:
        
        mock_file = mock_file
        # 设置 tldextract.extract 返回值
        mock_extract.return_value = MagicMock(domain="example", suffix="com")
        
        # 设置 pred_brand 的 side_effect
        # 第一次调用返回匹配，之后返回无匹配
        mock_pred_brand.side_effect = [
            ("brandA", "example.com", 0.9),
            (None, None, None),
            (None, None, None)
        ]
        
        # 设置 brand_converter 根据输入返回不同的值
        def brand_converter_side_effect(arg):
            if arg is None:
                return None
            return "brandA_converted"
        
        mock_brand_converter.side_effect = brand_converter_side_effect
        
        result = check_domain_brand_inconsistency(
            logo_boxes=logo_boxes_sample,
            domain_map_path="dummy_path",
            model=None,
            logo_feat_list=None,
            file_name_list=None,
            shot_path="",
            url="http://www.example.com",
            similarity_threshold=0.8
        )
        print(result)
        
        # 断言返回值被清除
        assert result == (None, None, [10, 20, 30, 40], None)
        
        # 断言 pred_brand 被调用三次（一次匹配，之后两次无匹配）
        assert mock_pred_brand.call_count == 3


# 测试用例 4：有 logo 框且域名不一致
def test_logo_boxes_domain_inconsistent():
    with patch("builtins.open", mock_open(read_data=pickle.dumps(domain_map_sample))) as mock_file, \
         patch("logo_matching.tldextract.extract") as mock_extract, \
         patch("logo_matching.pred_brand") as mock_pred_brand, \
         patch("logo_matching.brand_converter") as mock_brand_converter:
        
        mock_file = mock_file
        # 设置 tldextract.extract 返回值
        mock_extract.return_value = MagicMock(domain="example", suffix="com")
        
        # 设置 pred_brand 返回不一致域名
        mock_pred_brand.return_value = ("brandB", "other.com", 0.85)
        
        # 设置 brand_converter 返回转换后的品牌
        mock_brand_converter.return_value = "brandB_converted"
        
        result = check_domain_brand_inconsistency(
            logo_boxes=logo_boxes_sample,
            domain_map_path="dummy_path",
            model=None,
            logo_feat_list=None,
            file_name_list=None,
            shot_path="",
            url="http://www.example.com",
            similarity_threshold=0.8
        )
        
        # 断言返回不一致的匹配结果
        assert result == ("brandB_converted", "other.com", [10, 20, 30, 40], 0.85)
        
        # 断言 pred_brand 被调用一次（因为域名不一致，提前退出）
        assert mock_pred_brand.call_count == 1


# 测试用例 5：超过 topk 的 logo 框
def test_logo_boxes_exceed_topk():
    with patch("builtins.open", mock_open(read_data=pickle.dumps(domain_map_sample))) as mock_file, \
         patch("logo_matching.tldextract.extract") as mock_extract, \
         patch("logo_matching.pred_brand") as mock_pred_brand, \
         patch("logo_matching.brand_converter") as mock_brand_converter:
        
        mock_file = mock_file
        # 设置 tldextract.extract 返回值
        mock_extract.return_value = MagicMock(domain="example", suffix="com")
        
        # 设置 pred_brand 返回无匹配
        mock_pred_brand.return_value = (None, None, None)
        
        # 设置 brand_converter 返回值
        mock_brand_converter.return_value = None
        
        # 创建超过 topk 的 logo 框
        large_logo_boxes = logo_boxes_sample + [[170, 180, 190, 200]]
        
        result = check_domain_brand_inconsistency(
            logo_boxes=large_logo_boxes,
            domain_map_path="dummy_path",
            model=None,
            logo_feat_list=None,
            file_name_list=None,
            shot_path="",
            url="http://www.example.com",
            similarity_threshold=0.8,
            topk=3
        )
        
        # 断言返回值
        assert result == (None, None, None, None)
        
        # 断言 pred_brand 只被调用 topk 次
        assert mock_pred_brand.call_count == 3
