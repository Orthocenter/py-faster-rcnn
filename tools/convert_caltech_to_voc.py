import xml.etree.cElementTree as ET
import cv2

caltech_dir = "/home/yxchen/RPN_BF/datasets/caltech_5k/train"
output_dir = "/home/yxchen/RPN_BF/datasets/caltech_5k/voc_format"

import glob
for imgfile in glob.glob(caltech_dir + "/images/*.jpg"):
	img = cv2.imread(imgfile)
	width = img.shape[1]
	height = img.shape[0]

	filename = imgfile[-21:-4]
	annotfile = caltech_dir + '/annotations/' + filename + '.txt'

	root = ET.Element("annotation")
	ET.SubElement(root, "folder").text = "VOC2007"
	ET.SubElement(root, "filename").text = filename + '.jpg'

	size = ET.SubElement(root, "size")
	ET.SubElement(size, "width").text = str(width)
	ET.SubElement(size, "height").text = str(height)
	ET.SubElement(size, "depth").text = "3"

	ET.SubElement(root, "segmented").text = "0"

	with open(annotfile, 'r') as f:
		f.readline()
		for line in f.readlines():
			line_split = line.split(' ')

			x1 = float(line_split[1])
			y1 = float(line_split[2])
			x2 = x1 + float(line_split[3])
			y2 = y1 + float(line_split[4])
			x1 = int(x1)
			y1 = int(y1)
			x2 = int(x2)
			y2 = int(y2)

			obj = ET.SubElement(root, "object")
			ET.SubElement(obj, "name").text = "person"
			ET.SubElement(obj, "difficult").text = "0"
			bbox = ET.SubElement(obj, "bndbox")
			ET.SubElement(bbox, "xmin").text = str(x1)
			ET.SubElement(bbox, "ymin").text = str(y1)
			ET.SubElement(bbox, "xmax").text = str(x2)
			ET.SubElement(bbox, "ymax").text = str(y2)

	tree = ET.ElementTree(root)
	tree.write(output_dir + '/' + filename + '.xml')
