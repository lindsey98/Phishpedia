import re


def pred_text_in_image(ocr_model, shot_path):
    result = ocr_model.ocr(shot_path, cls=True)
    if result is None or result[0] is None:
        return ''

    most_fit_results = result[0]
    ocr_text = [line[1][0] for line in most_fit_results]
    detected_text = ' '.join(ocr_text)

    return detected_text


def check_email_credential_taking(ocr_model, shot_path):
    detected_text = pred_text_in_image(ocr_model, shot_path)
    if len(detected_text) > 0:
        return rule_matching(detected_text)
    return False, None


def rule_matching(detected_text):
    email_login_pattern = r'邮箱.*登录|邮箱.*登陆|邮件.*登录|邮件.*登陆'
    specified_email_pattern = r'@[\w.-]+\.\w+'

    if re.findall(email_login_pattern, detected_text):
        find_email = re.findall(specified_email_pattern, detected_text)
        if find_email:
            return True, find_email[0]

    return False, None
