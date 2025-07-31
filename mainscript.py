#!/user/bin/python3
import jetson_inference
import jetson_utils
import onnx
import sys

import argparse

print("running)")
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="/home/gabe/jetson-inference/python/training/classification/models/XrayModel2/efficientnet_b4.onnx", help="model to use, can be:  googlenet, resnet-18, ect. (see --help for others)")
parser.add_argument("--input-blob", type=str, default="input_0", help="name of the input blob")
parser.add_argument("--output-blob", type=str, default="output_0", help="name of the output blob")
parser.add_argument("--labels", type=str, default="/home/gabe/jetson-inference/python/training/classification/models/XrayModel2/labels.txt", help="filename of the labels file")
parser.add_argument("filename", type=str, help="filename of the image to process") 
parser.add_argument("output_filename", type=str, help="Directory to save the output image to")
opt = parser.parse_args()

img = jetson_utils.loadImage(opt.filename)
net = jetson_inference.imageNet(model=opt.model, input_blob=opt.input_blob, output_blob=opt.output_blob, labels=opt.labels)
class_idx, confidence = net.Classify(img)

class_desc = net.GetClassDesc(class_idx)
print("image is recognized as "+ str(class_desc) +" (class #"+ str(class_idx) +") with " + str(confidence*100)+"% confidence")

font = jetson_utils.cudaFont()
font.OverlayText(img, text=f"{str(confidence*100)}% {class_desc}", x=5, y=5 + (font.GetSize() + 5), color=font.Red, background=font.Gray40)
jetson_utils.saveImage(opt.output_filename, img)
print("output saved to output.jpg")
