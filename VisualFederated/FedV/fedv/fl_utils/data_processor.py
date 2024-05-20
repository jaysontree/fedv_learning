
import os
import xml.etree.ElementTree as ET
import yaml

def _to_yolo(src_dir):
    label_list_file = os.path.join(src_dir, 'label_list.txt')
    classes = {}
    for _idx,_label in enumerate(open(label_list_file).readlines()):
        classes[_label.strip()] = _idx
    out_yaml = {}
    out_yaml['path'] = src_dir
    anno_dir = os.path.join(src_dir, 'annotations')
    label_dir = os.path.join(src_dir, 'labels')
    if not os.path.exists(label_dir):
        os.mkdir(label_dir)
    img_dir = os.path.join(src_dir, 'images')
    if os.path.exists(os.path.join(src_dir, 'train.txt')):
        train_list = open(os.path.join(src_dir, 'train.txt'),'r').readlines()
        if not os.path.exists(os.path.join(label_dir, 'train')):
            os.mkdir(os.path.join(label_dir, 'train'))
        if not os.path.exists(os.path.join(img_dir, 'train')):
            os.mkdir(os.path.join(img_dir, 'train'))
        for line in train_list:
            img_file, anno_file = line.strip().split()
            filename = os.path.basename(anno_file)
            imgname = os.path.basename(img_file)
            anno_path = os.path.join(anno_dir, filename)
            in_file = open(anno_path, 'r')
            out_file = open(os.path.join(os.path.join(label_dir,'train'), filename.split('.')[0] + '.txt'), 'w')
            tree = ET.parse(in_file)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            for obj in root.iter('object'):
                diffi = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes or int(diffi) == 1:
                    continue 
                cls_id = classes[cls]
                box = obj.find('bndbox')
                x0, x1, y0, y1 = float(box.find('xmin').text), float(box.find('xmax').text), float(box.find('ymin').text), float(box.find('ymax').text)
                x0 = max(x0, 0)
                y0 = max(y0, 0)
                x1 = min(x1, w)
                y1 = min(y1, h)
                dx, dy = (x1 - x0)/w, (y1 - y0)/h
                cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
                out_file.write(f'{str(cls_id)} {cx} {cy} {dx} {dy}\n')
            out_file.close()
            _symlink(os.path.join(img_dir, imgname), os.path.join(os.path.join(img_dir, 'train'), imgname))
        out_yaml['train'] = 'images/train'
    else:
        out_yaml['train'] = None

    if os.path.exists(os.path.join(src_dir, 'val.txt')):
        val_list = open(os.path.join(src_dir, 'val.txt'),'r').readlines()
        if not os.path.exists(os.path.join(label_dir, 'val')):
            os.mkdir(os.path.join(label_dir, 'val'))
        if not os.path.exists(os.path.join(img_dir, 'val')):
            os.mkdir(os.path.join(img_dir, 'val'))    
        for line in val_list:
            img_file, anno_file = line.strip().split()
            filename = os.path.basename(anno_file)
            imgname = os.path.basename(img_file)
            anno_path = os.path.join(anno_dir, filename)
            in_file = open(anno_path, 'r')
            out_file = open(os.path.join(os.path.join(label_dir,'val'), filename.split('.')[0] + '.txt'), 'w')
            tree = ET.parse(in_file)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            for obj in root.iter('object'):
                diffi = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes or int(diffi) == 1:
                    continue 
                cls_id = classes[cls]
                box = obj.find('bndbox')
                x0, x1, y0, y1 = float(box.find('xmin').text), float(box.find('xmax').text), float(box.find('ymin').text), float(box.find('ymax').text)
                x0 = max(x0, 0)
                y0 = max(y0, 0)
                x1 = min(x1, w)
                y1 = min(y1, h)
                dx, dy = (x1 - x0)/w, (y1 - y0)/h
                cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
                out_file.write(f'{cls_id} {cx} {cy} {dx} {dy}\n')
            out_file.close()
            _symlink(os.path.join(img_dir, imgname), os.path.join(os.path.join(img_dir, 'val'), imgname))
        out_yaml['val'] = 'images/val'
    else:
        out_yaml['val'] = None
    
    out_yaml['test'] = None
    out_yaml['names'] = {}
    for k,v in classes.items():
        out_yaml['names'][v] = k 
    
    with open(os.path.join(src_dir, 'data.yaml'), 'w') as ouf:
        yaml.dump(out_yaml, ouf)

    return os.path.join(src_dir, 'data.yaml')

def _to_image_folder(src_dir):
    label_list_file = os.path.join(src_dir, 'label_list.txt')
    classes = {}
    for line in open(label_list_file).readlines():
        _label, _idx = line.strip().split()
        classes[_label.strip()] = _idx
    # check old WEFE classification dataset tarfile, support old paddle_fl format.
    if os.path.exists(os.path.join(src_dir, 'image.tgz')):
        import tarfile
        tar = tarfile.open(os.path.join(src_dir, 'image.tgz'))
        tar.extractall(path=src_dir)
    # end
    if os.path.exists(os.path.join(src_dir, 'train_list.txt')):
        train_list = open(os.path.join(src_dir, 'train_list.txt'),'r').readlines()
        train_sub_dir = os.path.join(src_dir, 'train')
        if not os.path.exists(train_sub_dir):
            os.mkdir(train_sub_dir)
        for label in classes.values():
            train_sub_class_dir = os.path.join(train_sub_dir, label)
            if not os.path.exists(train_sub_class_dir):
                os.mkdir(train_sub_class_dir)
        for item in train_list:
            img_file, cls_idx = item.strip().split()
            label = classes.get(cls_idx)
            img_name = os.path.basename(img_file)
            tgt_file = os.path.join(os.path.join(train_sub_dir, label), img_name)
            _symlink(os.path.join(src_dir, img_file), tgt_file)
    
    if os.path.exists(os.path.join(src_dir, 'val_list.txt')):
        val_list = open(os.path.join(src_dir, 'val_list.txt'),'r').readlines()
        val_sub_dir = os.path.join(src_dir, 'val')
        if not os.path.exists(val_sub_dir):
            os.mkdir(val_sub_dir)
        for label in classes.values():
            val_sub_class_dir = os.path.join(val_sub_dir, label)
            if not os.path.exists(val_sub_class_dir):
                os.mkdir(val_sub_class_dir)
        for item in val_list:
            img_file, cls_idx = item.strip().split()
            label = classes.get(cls_idx)
            img_name = os.path.basename(img_file)
            tgt_file = os.path.join(os.path.join(val_sub_dir, label), img_name)
            _symlink(os.path.join(src_dir, img_file), tgt_file)
    return src_dir

def _symlink(src_dir, dst_dir):
    try:
        if not os.path.exists(dst_dir):
            os.symlink(os.path.abspath(src_dir), dst_dir)
    except Exception as e:
        print(e)
    # 用force的话，单节点多实例同时软连接会有问题、
