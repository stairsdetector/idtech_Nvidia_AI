#### X-ray-project
         the purpose of this project was to diagnose illnesses using ai, or at least provide a proof of concept
         diagnosis ai, like this one reduce the amount of doctors needed to run hospitals, and other clinics, lowering healthcare costs, and providing a decent diagnosis for hospitals in rural or poor areas who can't afford a large                 diagnosis team
#### datasets:
         use the curl command given by kaggle
         https://www.kaggle.com/datasets/mohamedasak/chest-x-ray-6-classes-dataset
#### repositories used:
         Onnx
         jetson-inference
         pytorch
         make sure you have python installed on your computer
#### Model used:Efficientnet-b4

#### replication instructions
         step 1:make my-recognition and data files
         step 2:install https://www.kaggle.com/datasets/mohamedasak/chest-x-ray-6-classes-dataset using curl command curl -L -o ~/Downloads/chest-x-ray-6-classes-dataset.zip\
         step 3: extract dataset using unzip ~/Downloads/chest-x-ray-6-classes-dataset.zip\
         step 3: split dataset 2 using splitdataset.py
         step 4: move files over into datset 1, adding fibrosis as a new class
         step 5: switch jetson-inference/python/training/classification/train.py to run on efficient-net-b4
         step 6:enter docker at jetson-inference using ./docker/run.sh, cd to jetson-inference/python/training/classification
         step 7:re-train model using python3 train.py --model-dir=models/XrayModel data/LungXRaysG-Finalproject -a=efficientnet_b4 --epochs=10
         (you can train it for more, but it's progress stops extremely quickly)
         step 8:build script to use the resnet50.onnx file, my version is called mainscript.py, it is in the my-recognition folder
         (to find the names of the output and input of the model you need onnx)
         step 9: run code on images

#### files:
         /home/gabe/jetson-inference/my-recognition/mainscript.py
         this code loads the model and labels, using the argparser, it also gets the input file, and creates a second file, the output, both are placed in 
         /home/gabe/jetson-inference/python/training/classification/models/XrayModel/resnet50.onnx
         if you just want the model

#### running the network:
1.to run the network first move to the my-recognition directory
2.type python3 mainscript.py (path of the image you want scanned)
3.at the last line of code it should print out the diagnosis and confidence rating
