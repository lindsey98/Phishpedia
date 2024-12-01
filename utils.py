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
    if w_min == 0 or h_min == 0:  ## something wrong, stop resizing
        return img1, img2
    if w_min < h_min:
        img1_resize = img1.resize((int(w_min), math.ceil(h1 * (w_min/w1)))) # ceiling to prevent rounding to 0
        img2_resize = img2.resize((int(w_min), math.ceil(h2 * (w_min/w2))))
    else:
        img1_resize = img1.resize((math.ceil(w1 * (h_min/h1)), int(h_min)))
        img2_resize = img2.resize((math.ceil(w2 * (h_min/h2)), int(h_min)))
    return img1_resize, img2_resize

def brand_converter(brand_name):
    '''
    Helper function to deal with inconsistency in brand naming
    '''
    if brand_name == 'Adobe Inc.' or brand_name == 'Adobe Inc':
        return 'Adobe'
    elif brand_name == 'ADP, LLC' or brand_name == 'ADP, LLC.':
        return 'ADP'
    elif brand_name == 'Amazon.com Inc.' or brand_name == 'Amazon.com Inc':
        return 'Amazon'
    elif brand_name == 'Americanas.com S,A Comercio Electrnico':
        return 'Americanas.com S'
    elif brand_name == 'AOL Inc.' or brand_name == 'AOL Inc':
        return 'AOL'
    elif brand_name == 'Apple Inc.' or brand_name == 'Apple Inc':
        return 'Apple'
    elif brand_name == 'AT&T Inc.' or brand_name == 'AT&T Inc':
        return 'AT&T'
    elif brand_name == 'Banco do Brasil S.A.':
        return 'Banco do Brasil S.A'
    elif brand_name == 'Credit Agricole S.A.':
        return 'Credit Agricole S.A'
    elif brand_name == 'DGI (French Tax Authority)':
        return 'DGI French Tax Authority'
    elif brand_name == 'DHL Airways, Inc.' or brand_name == 'DHL Airways, Inc' or brand_name == 'DHL':
        return 'DHL Airways'
    elif brand_name == 'Dropbox, Inc.' or brand_name == 'Dropbox, Inc':
        return 'Dropbox'
    elif brand_name == 'eBay Inc.' or brand_name == 'eBay Inc':
        return 'eBay'
    elif brand_name == 'Facebook, Inc.' or brand_name == 'Facebook, Inc':
        return 'Facebook'
    elif brand_name == 'Free (ISP)':
        return 'Free ISP'
    elif brand_name == 'Google Inc.' or brand_name == 'Google Inc':
        return 'Google'
    elif brand_name == 'Mastercard International Incorporated':
        return 'Mastercard International'
    elif brand_name == 'Netflix Inc.' or brand_name == 'Netflix Inc':
        return 'Netflix'
    elif brand_name == 'PayPal Inc.' or brand_name == 'PayPal Inc':
        return 'PayPal'
    elif brand_name == 'Royal KPN N.V.':
        return 'Royal KPN N.V'
    elif brand_name == 'SF Express Co.':
        return 'SF Express Co'
    elif brand_name == 'SNS Bank N.V.':
        return 'SNS Bank N.V'
    elif brand_name == 'Square, Inc.' or brand_name == 'Square, Inc':
        return 'Square'
    elif brand_name == 'Webmail Providers':
        return 'Webmail Provider'
    elif brand_name == 'Yahoo! Inc' or brand_name == 'Yahoo! Inc.':
        return 'Yahoo!'
    elif brand_name == 'Microsoft OneDrive' or brand_name == 'Office365' or brand_name == 'Outlook':
        return 'Microsoft'
    elif brand_name == 'Global Sources (HK)':
        return 'Global Sources HK'
    elif brand_name == 'T-Online':
        return 'Deutsche Telekom'
    elif brand_name == 'Airbnb, Inc':
        return 'Airbnb, Inc.'
    elif brand_name == 'azul':
        return 'Azul'
    elif brand_name == 'Raiffeisen Bank S.A':
        return 'Raiffeisen Bank S.A.'
    elif brand_name == 'Twitter, Inc' or brand_name == 'Twitter':
        return 'Twitter, Inc.'
    elif brand_name == 'capital_one':
        return 'Capital One Financial Corporation'
    elif brand_name == 'la_banque_postale':
        return 'La Banque postale'
    elif brand_name == 'db':
        return 'Deutsche Bank AG'
    elif brand_name == 'Swiss Post' or brand_name == 'PostFinance':
        return 'PostFinance'
    elif brand_name == 'grupo_bancolombia':
        return 'Bancolombia'
    elif brand_name == 'barclays':
        return 'Barclays Bank Plc'
    elif brand_name == 'gov_uk':
        return 'Government of the United Kingdom'
    elif brand_name == 'Aruba S.p.A':
        return 'Aruba S.p.A.'
    elif brand_name == 'TSB Bank Plc':
        return 'TSB Bank Limited'
    elif brand_name == 'strato':
        return 'Strato AG'
    elif brand_name == 'cogeco':
        return 'Cogeco'
    elif brand_name == 'Canada Revenue Agency':
        return 'Government of Canada'
    elif brand_name == 'UniCredit Bulbank':
        return 'UniCredit Bank Aktiengesellschaft'
    elif brand_name == 'ameli_fr':
        return 'French Health Insurance'
    elif brand_name == 'Banco de Credito del Peru':
        return 'bcp'
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