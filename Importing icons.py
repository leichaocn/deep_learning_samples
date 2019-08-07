"""
这是使用GAN的第一步。

在 https://icons8.com/app 下载mac版的文件，解压后，找Icons8App_for_Mac_OS\Icons8 v5.6.9\Icons8.app\Contents\Resources\
把icons.tar解压出icons文件夹。

"""
import plyvel
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import re
from io import BytesIO
import shutil
from PIL import Image
import os
import json
import unicodedata


"""
注意配置path
"""
# Adjust to your local path:
# path = '/Users/douwe/Downloads/Icons8 v5.6.3/icons'
import os
path = os.getcwd()+'/data/icons'
db = plyvel.DB(path)

splitter = re.compile(b'[\x00-\x08]')

def parse_value(value):
    res = {}
    prev = ''
    for elem in splitter.split(value):
        if not elem:
            continue
        try:
            elem = elem.decode('utf8')
        except UnicodeDecodeError:
            continue
        if elem in ('category', 'name', 'platform', 'canonical_name', 'svg'):
            if elem == 'name' and len(prev) == 1:
                prev = 'u_' + unicodedata.name(prev).lower().replace(' ', '_')
            res[elem] = prev
        prev = elem
    return res

for _, value in db:
    res = parse_value(value)
    break
print(res)

icons = {}

for key, value in db:
    try:
        res = parse_value(value)
    except ValueError:
        continue
    if res.get('platform') == 'ios':
        name = res.get('name')
        if not name:
            name = res.get('canonical_name')
            if not name:
                continue
            name = name.lower().replace(' ', '_')
        icons[name] = res
print(len(icons))

SIZES = (16, 28, 32, 50)

if os.path.isdir('icons'):
    shutil.rmtree('icons')
os.mkdir('icons')
for size in SIZES:
    os.mkdir(path+'/png%s' % size)
os.mkdir(path+'/svg')


saved = []
for icon in icons.values():
    icon = dict(icon)
    if not 'svg' in icon:
        continue
    svg = icon.pop('svg')
    try:
        drawing = svg2rlg(BytesIO(svg.encode('utf8')))
    except ValueError:
        continue
    except AttributeError:
        continue
    open(path+'/svg/%s.svg' % icon['name'], 'w').write(svg)
    p = renderPM.drawToPIL(drawing)
    for size in SIZES:
        resized = p.resize((size, size), Image.ANTIALIAS)
        resized.save(path+'/png%s/%s.png' % (size, icon['name']))
    saved.append(icon)
json.dump(saved, open(path+'/index.json', 'w'), indent=2)
print(len(saved))
print(icon['name'])
