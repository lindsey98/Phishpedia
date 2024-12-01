import torch.nn.functional as F
import math


def resolution_alignment(img1, img2):
    '''
    Resize two images according to the minimum resolution between the two
    :param img1: first image in PIL.Image
    :param img2: second image in PIL.Image
    :return: resized img1 in PIL.Image, resized img2 in PIL.Image
    '''
    w1, h1 = img1.size
    w2, h2 = img2.size
    w_min, h_min = min(w1, w2), min(h1, h2)
    if w_min == 0 or h_min == 0:  # something wrong, stop resizing
        return img1, img2
    if w_min < h_min:
        img1_resize = img1.resize((int(w_min), math.ceil(h1 * (w_min / w1))))  # ceiling to prevent rounding to 0
        img2_resize = img2.resize((int(w_min), math.ceil(h2 * (w_min / w2))))
    else:
        img1_resize = img1.resize((math.ceil(w1 * (h_min / h1)), int(h_min)))
        img2_resize = img2.resize((math.ceil(w2 * (h_min / h2)), int(h_min)))
    return img1_resize, img2_resize


def brand_converter(brand_name):
    '''
    Helper function to deal with inconsistency in brand naming
    '''
    brand_tran_dict = {'Adobe Inc.': 'Adobe', 'Adobe Inc': 'Adobe',
                       'ADP, LLC': 'ADP', 'ADP, LLC.': 'ADP',
                       'Amazon.com Inc.': 'Amazon', 'Amazon.com Inc': 'Amazon',
                       'Americanas.com S,A Comercio Electrnico': 'Americanas.com S',
                       'AOL Inc.': 'AOL', 'AOL Inc': 'AOL',
                       'Apple Inc.': 'Apple', 'Apple Inc': 'Apple',
                       'AT&T Inc.': 'AT&T', 'AT&T Inc': 'AT&T',
                       'Banco do Brasil S.A.': 'Banco do Brasil S.A',
                       'Credit Agricole S.A.': 'Credit Agricole S.A',
                       'DGI (French Tax Authority)': 'DGI French Tax Authority',
                       'DHL Airways, Inc.': 'DHL Airways', 'DHL Airways, Inc': 'DHL Airways', 'DHL': 'DHL Airways',
                       'Dropbox, Inc.': 'Dropbox', 'Dropbox, Inc': 'Dropbox',
                       'eBay Inc.': 'eBay', 'eBay Inc': 'eBay',
                       'Facebook, Inc.': 'Facebook', 'Facebook, Inc': 'Facebook',
                       'Free (ISP)': 'Free ISP',
                       'Google Inc.': 'Google', 'Google Inc': 'Google',
                       'Mastercard International Incorporated': 'Mastercard International',
                       'Netflix Inc.': 'Netflix', 'Netflix Inc': 'Netflix',
                       'PayPal Inc.': 'PayPal', 'PayPal Inc': 'PayPal',
                       'Royal KPN N.V.': 'Royal KPN N.V',
                       'SF Express Co.': 'SF Express Co',
                       'SNS Bank N.V.': 'SNS Bank N.V',
                       'Square, Inc.': 'Square', 'Square, Inc': 'Square',
                       'Webmail Providers': 'Webmail Provider',
                       'Yahoo! Inc': 'Yahoo!', 'Yahoo! Inc.': 'Yahoo!',
                       'Microsoft OneDrive': 'Microsoft', 'Office365': 'Microsoft', 'Outlook': 'Microsoft',
                       'Global Sources (HK)': 'Global Sources HK',
                       'T-Online': 'Deutsche Telekom',
                       'Airbnb, Inc': 'Airbnb, Inc.',
                       'azul': 'Azul',
                       'Raiffeisen Bank S.A': 'Raiffeisen Bank S.A.',
                       'Twitter, Inc': 'Twitter, Inc.', 'Twitter': 'Twitter, Inc.',
                       'capital_one': 'Capital One Financial Corporation',
                       'la_banque_postale': 'La Banque postale',
                       'db': 'Deutsche Bank AG',
                       'Swiss Post': 'PostFinance', 'PostFinance': 'PostFinance',
                       'grupo_bancolombia': 'Bancolombia',
                       'barclays': 'Barclays Bank Plc',
                       'gov_uk': 'Government of the United Kingdom',
                       'Aruba S.p.A': 'Aruba S.p.A.',
                       'TSB Bank Plc': 'TSB Bank Limited',
                       'strato': 'Strato AG',
                       'cogeco': 'Cogeco',
                       'Canada Revenue Agency': 'Government of Canada',
                       'UniCredit Bulbank': 'UniCredit Bank Aktiengesellschaft',
                       'ameli_fr': 'French Health Insurance',
                       'Banco de Credito del Peru': 'bcp'
                       }
    # find the value in the dict else return the origin brand name
    tran_brand_name = brand_tran_dict.get(brand_name, None)
    if tran_brand_name:
        return tran_brand_name
    else:
        return brand_name


def l2_norm(x):
    """
    l2 normalization
    :param x:
    :return:
    """
    if len(x.shape):
        x = x.reshape((x.shape[0], -1))
    return F.normalize(x, p=2, dim=1)
