# Copyright 2021 Tianmian Tech. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



from __future__ import print_function

import requests
import shutil
import sys
import tarfile
import zipfile
from fedv import get_data_dir
import os
import logging
import six
import uuid

def download(url, dirname, save_name=None):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    filename = os.path.join(dirname,
                            url.split('/')[-1]
                            if save_name is None else save_name)

    if os.path.exists(filename):
        filename = filename + f'.{uuid.uuid4()}'

    retry = 0
    retry_limit = 3
    while not os.path.exists(filename):
        if retry < retry_limit:
            retry += 1
        else:
            raise RuntimeError("Cannot download {0} within retry limit {1}".
                               format(url, retry_limit))
        sys.stderr.write("Cache file %s not found, downloading %s \n" %
                         (filename, url))
        sys.stderr.write("Begin to download\n")
        r = requests.get(url, stream=True)
        total_length = r.headers.get('content-length')

        if total_length is None:
            with open(filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        else:
            with open(filename, 'wb') as f:
                chunk_size = 4096
                total_length = int(total_length)
                total_iter = total_length / chunk_size + 1
                log_interval = total_iter / 20 if total_iter > 20 else 1
                log_index = 0
                for data in r.iter_content(chunk_size=chunk_size):
                    if six.PY2:
                        data = six.b(data)
                    f.write(data)
                    log_index += 1
                    if log_index % log_interval == 0:
                        sys.stderr.write("downloading chunck...\n")
                    sys.stdout.flush()
    sys.stderr.write("\nDownload finished\n")
    sys.stdout.flush()

    return filename

def extract(tar_file, target_path):
    try:
        tar = tarfile.open(tar_file, "r:gz")
        file_names = tar.getnames()
        for file_name in file_names:
            tar.extract(file_name, target_path)
        tar.close()
    except Exception as e:
        print(e)


def un_zip(file_name,target_path):
    """unzip zip file"""
    try:
        zip_file = zipfile.ZipFile(file_name)
        if os.path.isdir(target_path):
            pass
        else:
            os.mkdir(target_path)
        names = zip_file.namelist()
        for name in names:
            zip_file.extract(name,target_path)
        zip_file.close()
        root_dir = names[0].split('/')[0]
        up_1_dir = os.path.dirname(names[0])
        up_2_dir = os.path.dirname(up_1_dir)
        return up_1_dir if up_1_dir == root_dir else up_2_dir
    except Exception as e:
        print(e)
        raise Exception(f"unzip file {file_name} error as {e}")

def make_zip(source_dir, zip_file): # zip dir
    zipf = zipfile.ZipFile(zip_file, 'w')
    pre_len = len(os.path.dirname(source_dir))
    for parent, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[pre_len:].strip(os.path.sep)
            zipf.write(pathfile, arcname)
    zipf.close()

def make_zip_file(source, zip_file): # zip file
    with zipfile.ZipFile(zip_file, 'w') as zipf:
        if isinstance(source, list):
            for _source in source:
                zipf.write(_source, os.path.basename(_source))
        else:
            zipf.write(source, os.path.basename(source))

def make_zip_dir(source, zip_file):
    with zipfile.ZipFile(zip_file, 'w') as zipf:
        if isinstance(source, list):
            for _source in source:
                arch = os.path.join(os.path.basename(os.path.dirname(_source)), os.path.basename(_source))
                zipf.write(_source, arch)
        else:
            zipf.write(source, os.path.basename(source))       
            
def job_download(url, job_id,base_dir):
    try:
        data_file = download(url, base_dir, f"{job_id}.zip")
        dir_name = un_zip(data_file, base_dir)
        target_dir = os.path.join(base_dir,dir_name)
    except Exception as e:
        logging.error(f"job download with {job_id} error as {e} ")

    return target_dir


def getImageList(dir, filelist):
    newDir = dir
    if os.path.isfile(dir):
        if dir.endswith(".jpg") or dir.endswith(".JPG") or dir.endswith(".png") or dir.endswith(".PNG")\
            or dir.endswith(".jpeg") or dir.endswith(".webp") or dir.endswith(".bmp") or dir.endswith(".tif")\
            or dir.endswith(".gif"):
            filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getImageList(newDir, filelist)
    return filelist

def extractImages(src_dir):
    imageList = getImageList(src_dir,[])
    target_path = f"{os.path.dirname(src_dir)}_tmp"
    if os.path.isdir(target_path):
        pass
    else:
        os.mkdir(target_path)
    for item in imageList:
        tmp = os.path.basename(item)
        shutil.copy(item, target_path + '/' + tmp)
    shutil.rmtree(src_dir)
    os.rename(target_path,src_dir)