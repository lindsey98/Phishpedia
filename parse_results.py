import os.path
import shutil
from datetime import datetime

def get_pos_site(result_txt):

    df = [x.strip().split('\t') for x in open(result_txt, encoding='ISO-8859-1').readlines()]
    df_pos = [x for x in df if (x[2] == '1')]
    df_pos = [x for x in df_pos if x[3] not in ['Google', 'Webmail Provider',
                                                'WhatsApp', 'Luno']]
    return df_pos



if __name__ == '__main__':
    # today = datetime.now().strftime('%Y%m%d')
    today = '20231223'
    result_txt = f'/home/ruofan/git_space/PhishEmail/datasets/{today}_results.txt'
    df_pos = get_pos_site(result_txt)
    print(len(df_pos))

    pos_result_txt = f'/home/ruofan/git_space/PhishEmail/datasets/{today}_pos.txt'
    pos_result_dir = f'/home/ruofan/git_space/PhishEmail/datasets/sjtu_phish_pos/{today}'
    os.makedirs(pos_result_dir, exist_ok=True)

    for x in df_pos:
        url = x[1]
        if os.path.exists(pos_result_txt) and url in open(pos_result_txt).read():
            pass
        else:
            with open(pos_result_txt, 'a+') as f:
                f.write(url+'\n')

        # try:
        shutil.copytree(os.path.join(f'/home/ruofan/git_space/PhishEmail/datasets/sjtu_phish/{today}', x[0]),
                            os.path.join(pos_result_dir, x[0]))
        # except FileExistsError:
        #     pass

    print(df_pos)