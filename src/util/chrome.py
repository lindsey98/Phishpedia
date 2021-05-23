from selenium.common.exceptions import NoSuchElementException, TimeoutException, MoveTargetOutOfBoundsException, StaleElementReferenceException
import helium
import time
import requests
from seleniumwire import webdriver


def initialize_chrome_settings(lang_txt:str):
    '''
    initialize chrome settings
    :return: chrome options
    '''
    # enable translation
    white_lists = {}

    with open(lang_txt) as langf:
        for i in langf.readlines():
            i = i.strip()
            text = i.split(' ')
            print(text)
            white_lists[text[1]] = 'en'
    prefs = {
        "translate": {"enabled": "true"},
        "translate_whitelists": white_lists
    }
    options = webdriver.ChromeOptions()

    options.add_experimental_option("prefs", prefs)
    options.add_argument('--ignore-certificate-errors') # ignore errors
    options.add_argument('--ignore-ssl-errors')
    # options.add_argument("--headless") # FIXME: do not disable browser (have some issues: https://github.com/mherrmann/selenium-python-helium/issues/47)
    options.add_argument('--no-proxy-server')
    options.add_argument("--proxy-server='direct://'")
    options.add_argument("--proxy-bypass-list=*")

    options.add_argument("--start-maximized")
    options.add_argument('--window-size=1920,1080') # fix screenshot size
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36')
    options.set_capability('unhandledPromptBehavior', 'dismiss') # dismiss

    return  options

def vt_scan(url_test):
    '''
    Call VTScan
    :param url_test: url to scan
    :return: # positive engines, # total engines
    '''
    retry = 0
    api_key = "76d947105d47a73cc9fa20f1a1a5905f75ca4bb83fc0ff2f41b6cd14daa798e5"
    url = 'https://www.virustotal.com/vtapi/v2/url/report'

    params = {'apikey': api_key, 'resource': url_test, 'scan':1}
    response = requests.get(url, params=params)
    print("Response: ", response)
    response = response.json()

    # This means the url wasnt in VT's database, preparing a new scan
    while("total" not in response and "positives" not in response and retry < 3):
        print("[*] " + str(retry) + " try. Maximum of 3 tries with 30 seconds interval...")
        # Intentionally sleeping for 30 seconds before coming back to retrieve results
        time.sleep(30)
        response = requests.get(url, params=params).json()
        retry +=1

    # Getting out of the loop means either tried >= 3 times, or successfully gotten result
    try:
        positive = response['positives']
        total = response['total']
    except KeyError:
        positive = None
        total = None

    return positive, total


def get_page_text(driver):
    '''
    get body text from html
    :param driver: chromdriver
    :return: text
    '''
    try:
        body = driver.find_element_by_tag_name('body').text
    except NoSuchElementException as e: # if no body tag, just get all text
        print(e)
        try:
            body = driver.page_source
        except Exception as e:
            print(e)
            body = ''
    return body

def visit_url(driver, orig_url, popup=False, sleep=False):
    '''
    Visit a URL
    :param driver: chromedriver
    :param orig_url: URL to visit
    :param popup: click popup window or not
    :param sleep: need sleep time or not
    :return: load url successful or not
    '''
    try:
        driver.get(orig_url)
        if sleep:
            time.sleep(2)
        if popup:
            click_popup()
        alert_msg = driver.switch_to.alert.text
        driver.switch_to.alert.dismiss()
        return True, driver
    except TimeoutException as e:
        print(str(e))
        clean_up_window(driver)  # clean up the windows
        return False, driver
    except Exception as e:
        print(str(e))
        print("no alert")
        return True, driver


def click_button(button_text):
    helium.Config.implicit_wait_secs = 5 # this is the implicit timeout for helium
    helium.get_driver().implicitly_wait(5)
    try:
        helium.click(helium.Button(button_text))
        return True
    except:
        return False

def click_popup():
    '''
    Click unexpected popup (accpet terms condditions, close alerts etc.)
    :return:
    '''
    helium.Config.implicit_wait_secs = 5 # this is the implicit timeout for helium
    helium.get_driver().implicitly_wait(5)
    keyword_list = ["Accept", "Accept all cookies", "Accept and continue", "Continue", "I accept",
                    "OK", "AGREE", "close", "Close", "accept", "Accept all",
                    "I agree", "I AGREE", "Allow everyone", "Enter Now", "Confirm selection"]

    for button_text in keyword_list:
        success = click_button(button_text=button_text)
        if success:
            return


def click_text(text):
    '''
    click the text's region
    :param text:
    :return:
    '''
    helium.Config.implicit_wait_secs = 5 # this is the implicit timeout for helium
    helium.get_driver().implicitly_wait(5) # this is the implicit timeout for selenium
    try:
        # helium.highlight(text) # highlight text for debugging
        # time.sleep(1)
        helium.click(text)
        time.sleep(5) # wait until website is completely loaded
        # click_popup()
    except TimeoutException as e:
        print(e)
    except LookupError as e:
        print(e)
    except Exception as e:
        print(e)

def click_point(x, y):
    '''
    click on coordinate (x,y)
    :param x:
    :param y:
    :return:
    '''
    helium.Config.implicit_wait_secs = 5 # this is the implicit timeout for helium
    helium.get_driver().implicitly_wait(5) # this the implicit timeout for selenium
    try:
        helium.click(helium.Point(x, y))
        time.sleep(5) # wait until website is completely loaded
        # click_popup()
    except TimeoutException as e:
        print(e)
    except MoveTargetOutOfBoundsException as e:
        print(e)
    except LookupError as e:
        print(e)
    except AttributeError as e:
        print(e)
    except Exception as e:
        print(e)


def clean_up_window(driver):
    '''
    Close chrome tabs except the current tab properly
    :param driver: chromedriver
    :return:
    '''
    current_window = driver.current_window_handle
    for i in driver.window_handles: # loop over all chrome windows
        try:
            if i != current_window: # close other windows
                driver.switch_to_window(i)
                driver.close()
        except Exception as e: # unknown exception occurs
            print(e)
            pass
    # switch back to the current window
    try:
        driver.switch_to_window(current_window)
    except Exception as e:
        print(e)
        print('Cannot switch back to the current window')


def writetxt(txtpath, contents):
    '''
    write into txt file with encoding utf-8
    :param txtpath:
    :param contents: text to write
    :return:
    '''
    with open(txtpath, 'w', encoding='utf-8') as fw:
        fw.write(contents)

