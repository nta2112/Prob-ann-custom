import os
import json
import random
import xml.etree.ElementTree as ET
import sys
import re

def write_voc_xml(annotation_dir, filename, width, height, annotations, class_names):
    root = ET.Element("annotation")
    
    folder = ET.SubElement(root, "folder")
    folder.text = "JPEGImages"
    
    filename_elem = ET.SubElement(root, "filename")
    filename_elem.text = filename
    
    size = ET.SubElement(root, "size")
    w_elem = ET.SubElement(size, "width")
    w_elem.text = str(width)
    h_elem = ET.SubElement(size, "height")
    h_elem.text = str(height)
    d_elem = ET.SubElement(size, "depth")
    d_elem.text = "3"
    
    for ann in annotations:
        obj = ET.SubElement(root, "object")
        name = ET.SubElement(obj, "name")
        name.text = class_names[ann['category_id']]
        
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"
        
        bndbox = ET.SubElement(obj, "bndbox")
        x, y, w, h = ann['bbox']
        
        xmin = int(round(x))
        ymin = int(round(y))
        xmax = int(round(x + w))
        ymax = int(round(y + h))
        
        # Clip to image boundaries
        xmin = max(1, xmin)
        ymin = max(1, ymin)
        xmax = min(width, xmax)
        ymax = min(height, ymax)
        
        if xmax <= xmin:
            xmax = xmin + 1
        if ymax <= ymin:
            ymax = ymin + 1
            
        xmin_elem = ET.SubElement(bndbox, "xmin")
        xmin_elem.text = str(xmin)
        ymin_elem = ET.SubElement(bndbox, "ymin")
        ymin_elem.text = str(ymin)
        xmax_elem = ET.SubElement(bndbox, "xmax")
        xmax_elem.text = str(xmax)
        ymax_elem = ET.SubElement(bndbox, "ymax")
        ymax_elem.text = str(ymax)
        
    image_id = os.path.splitext(filename)[0]
    xml_path = os.path.join(annotation_dir, f"{image_id}.xml")
    
    tree = ET.ElementTree(root)
    tree.write(xml_path, encoding='utf-8', xml_declaration=True)

def patch_open_world_file(class_names):
    file_path = os.path.join('datasets', 'torchvision_datasets', 'open_world.py')
    if not os.path.exists(file_path):
        # Fallback if run from a subdirectory
        file_path = os.path.join('..', 'datasets', 'torchvision_datasets', 'open_world.py')
        
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Cannot patch open_world.py.")
        return
        
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    if 'VOC_COCO_CLASS_NAMES["IP102"]' in content:
        print("open_world.py has already been patched for IP102.")
        return

    # Split class_names: task 1 has 27 classes, task 2,3,4 have 25 classes each.
    t1_classes = class_names[:27]
    t2_classes = class_names[27:52]
    t3_classes = class_names[52:77]
    t4_classes = class_names[77:]
    
    patch_code = f"""
# IP102 Class splits
IP102_T1_CLASS_NAMES = {repr(t1_classes)}
IP102_T2_CLASS_NAMES = {repr(t2_classes)}
IP102_T3_CLASS_NAMES = {repr(t3_classes)}
IP102_T4_CLASS_NAMES = {repr(t4_classes)}
VOC_COCO_CLASS_NAMES["IP102"] = tuple(itertools.chain(IP102_T1_CLASS_NAMES, IP102_T2_CLASS_NAMES, IP102_T3_CLASS_NAMES, IP102_T4_CLASS_NAMES, UNK_CLASS))
"""
    # Find print(VOC_COCO_CLASS_NAMES) and insert before it
    insertion_marker = "print(VOC_COCO_CLASS_NAMES)"
    if insertion_marker in content:
        content = content.replace(insertion_marker, patch_code + "\n" + insertion_marker)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Patched open_world.py successfully with IP102 class lists!")
    else:
        print("Warning: print(VOC_COCO_CLASS_NAMES) not found in open_world.py. Appending at the end.")
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(patch_code)

def convert_to_voc(json_path, split_name, output_dir, image_path_map, class_names, category_id_to_idx):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    anno_dir = os.path.join(output_dir, 'Annotations')
    jpeg_dir = os.path.join(output_dir, 'JPEGImages')
    
    os.makedirs(anno_dir, exist_ok=True)
    os.makedirs(jpeg_dir, exist_ok=True)
    
    img_id_to_anns = {}
    for ann in data['annotations']:
        img_id_to_anns.setdefault(ann['image_id'], []).append(ann)
        
    image_ids = []
    
    for img in data['images']:
        img_id = img['id']
        filename = img['file_name']
        width = img['width']
        height = img['height']
        
        base_name = os.path.basename(filename)
        src_path = image_path_map.get(base_name)
        if not src_path:
            src_path = image_path_map.get(f"{img_id}.jpg") or image_path_map.get(f"{img_id}.png")
            
        if not src_path:
            continue
            
        dest_path = os.path.join(jpeg_dir, base_name)
        if not os.path.exists(dest_path):
            try:
                os.symlink(src_path, dest_path)
            except OSError:
                import shutil
                shutil.copy(src_path, dest_path)
            
        img_anns = img_id_to_anns.get(img_id, [])
        mapped_anns = []
        for ann in img_anns:
            mapped_ann = ann.copy()
            cat_id = ann['category_id']
            if cat_id in category_id_to_idx:
                mapped_ann['category_id'] = category_id_to_idx[cat_id]
            else:
                if 0 <= cat_id < len(class_names):
                    mapped_ann['category_id'] = cat_id
                else:
                    mapped_ann['category_id'] = 0
            mapped_anns.append(mapped_ann)
            
        write_voc_xml(anno_dir, base_name, width, height, mapped_anns, class_names)
        
        img_id_str = os.path.splitext(base_name)[0]
        image_ids.append(img_id_str)
        
    return image_ids

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare IP102 dataset in VOC format for PROB OWOD")
    parser.add_argument('--json_dir', default='/kaggle/input/datasets/eljazouly/ip102-coco-annotations/coco_annotations', type=str)
    parser.add_argument('--image_dir', default='/kaggle/input/datasets/rtlmhjbn/ip02-dataset', type=str)
    parser.add_argument('--output_dir', default='/kaggle/working/datasets/IP102', type=str)
    args = parser.parse_args()

    train_json_path = os.path.join(args.json_dir, 'train.json')
    val_json_path = os.path.join(args.json_dir, 'val.json')
    test_json_path = os.path.join(args.json_dir, 'test.json')
    
    if not os.path.exists(train_json_path):
        # Alternative locations
        alt_json_dir = '/kaggle/input/ip102-coco-annotations/coco_annotations'
        train_json_path = os.path.join(alt_json_dir, 'train.json')
        val_json_path = os.path.join(alt_json_dir, 'val.json')
        test_json_path = os.path.join(alt_json_dir, 'test.json')
        
    if not os.path.exists(train_json_path):
        print(f"Error: Train JSON path {train_json_path} does not exist.")
        sys.exit(1)
        
    # Read categories mapping
    with open(train_json_path, 'r') as f:
        coco_data = json.load(f)
    categories = sorted(coco_data['categories'], key=lambda x: x['id'])
    category_id_to_idx = {cat['id']: idx for idx, cat in enumerate(categories)}
    class_names = [cat['name'] for cat in categories]
    
    print(f"Total IP102 Classes: {len(class_names)}")
    
    # Scan for images
    print(f"Scanning for images in {args.image_dir}...")
    image_path_map = {}
    for root, dirs, files in os.walk(args.image_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path_map[f] = os.path.join(root, f)
    print(f"Found {len(image_path_map)} image files in the dataset path.")
    
    # Patch the open_world.py file in the repository
    patch_open_world_file(class_names)
    
    # Convert splits
    print("Converting Train JSON...")
    train_ids = convert_to_voc(train_json_path, 'train', args.output_dir, image_path_map, class_names, category_id_to_idx)
    print(f"Converted {len(train_ids)} train images.")
    
    val_ids = []
    if os.path.exists(val_json_path):
        print("Converting Val JSON...")
        val_ids = convert_to_voc(val_json_path, 'val', args.output_dir, image_path_map, class_names, category_id_to_idx)
        print(f"Converted {len(val_ids)} val images.")
        
    test_ids = []
    if os.path.exists(test_json_path):
        print("Converting Test JSON...")
        test_ids = convert_to_voc(test_json_path, 'test', args.output_dir, image_path_map, class_names, category_id_to_idx)
        print(f"Converted {len(test_ids)} test images.")
        
    # If val/test ids are empty, split train_ids to create mock validation and test sets (similar to prep_ip102.py)
    if len(val_ids) == 0 or len(test_ids) == 0:
        print("Val or Test set annotations missing. Splitting train set...")
        random.seed(42)
        random.shuffle(train_ids)
        n = len(train_ids)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)
        val_ids = train_ids[train_end:val_end]
        test_ids = train_ids[val_end:]
        train_ids = train_ids[:train_end]
        print(f"Split completed: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    # Save ImageSets / IP102 split text files
    imagesets_dir = os.path.join(args.output_dir, 'ImageSets', 'IP102')
    os.makedirs(imagesets_dir, exist_ok=True)
    
    # We create the task train splits
    # Task 1 classes: index 0 to 26 (27 classes)
    # Task 2 classes: index 27 to 51 (25 classes)
    # Task 3 classes: index 52 to 76 (25 classes)
    # Task 4 classes: index 77 to 101 (25 classes)
    
    # Filter image IDs per task based on class annotations
    anno_dir = os.path.join(args.output_dir, 'Annotations')
    
    def get_img_classes(img_id):
        xml_path = os.path.join(anno_dir, f"{img_id}.xml")
        if not os.path.exists(xml_path):
            return set()
        tree = ET.parse(xml_path)
        root = tree.getroot()
        cls_indices = set()
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name in class_names:
                cls_indices.add(class_names.index(name))
        return cls_indices

    print("Splitting train images into tasks...")
    t1_train_ids = []
    t2_train_ids = []
    t3_train_ids = []
    t4_train_ids = []
    
    for img_id in train_ids:
        clses = get_img_classes(img_id)
        # Check if the image contains objects from the corresponding task ranges
        # Task 1: 0-26
        if any(c in range(0, 27) for c in clses):
            t1_train_ids.append(img_id)
        # Task 2: 27-51
        if any(c in range(27, 52) for c in clses):
            t2_train_ids.append(img_id)
        # Task 3: 52-76
        if any(c in range(52, 77) for c in clses):
            t3_train_ids.append(img_id)
        # Task 4: 77-102
        if any(c in range(77, 102) for c in clses):
            t4_train_ids.append(img_id)

    # Save to file
    with open(os.path.join(imagesets_dir, 'owod_t1_train.txt'), 'w') as f:
        f.write('\n'.join(t1_train_ids))
    with open(os.path.join(imagesets_dir, 'owod_t2_train.txt'), 'w') as f:
        f.write('\n'.join(t2_train_ids))
    with open(os.path.join(imagesets_dir, 'owod_t3_train.txt'), 'w') as f:
        f.write('\n'.join(t3_train_ids))
    with open(os.path.join(imagesets_dir, 'owod_t4_train.txt'), 'w') as f:
        f.write('\n'.join(t4_train_ids))
        
    with open(os.path.join(imagesets_dir, 'owod_all_task_test.txt'), 'w') as f:
        f.write('\n'.join(test_ids))
        
    print(f"Generated ImageSets splits under {imagesets_dir}:")
    print(f"  owod_t1_train.txt: {len(t1_train_ids)} images")
    print(f"  owod_t2_train.txt: {len(t2_train_ids)} images")
    print(f"  owod_t3_train.txt: {len(t3_train_ids)} images")
    print(f"  owod_t4_train.txt: {len(t4_train_ids)} images")
    print(f"  owod_all_task_test.txt: {len(test_ids)} images")
    
    print("Dataset setup for IP102 completed successfully!")

if __name__ == '__main__':
    main()
