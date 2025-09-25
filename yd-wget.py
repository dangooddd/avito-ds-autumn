#!/usr/bin/env python
import os, sys, json
import urllib.parse as ul
from pathlib import Path

"""
Скачивает файл из Yandex.Disk с помощью wget
Благодарность за оригинальный скрипт: https://gist.github.com/Yegorov/dc61c42aa4e89e139cd8248f59af6b3e#file-ya-py.
Эта измененная версия оригинального скрипта.

Использование: ./yd-wget.py url path
"""

sys.argv.append(".") if len(sys.argv) == 2 else None
base_url = "https://cloud-api.yandex.net:443/v1/disk/public/resources/download?public_key="
url = ul.quote_plus(sys.argv[1])
folder = Path(sys.argv[2])
res = os.popen(f"wget -qO - {base_url}{url}").read()
json_res = json.loads(res)
filename = ul.parse_qs(ul.urlparse(json_res["href"]).query)["filename"][0]
os.system(f"wget '{json_res["href"]}' -O {folder / filename}")
