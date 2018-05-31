# Google-inception-v3-tensorflow
这个介绍有两个：
一个是Google-inception-v3-tensorflow

另一个是基于Tensorflow-GPU的cnn-text-classification-tf自建立模型
Google-inception-v3-tensorflow:
![Alt text](https://github.com/ttanzhiqiang/Google-inception-v3-tensorflow/blob/master/Google-inception-v3-tensorflow/car.png)

Images/car.jpg
![Alt text](https://github.com/ttanzhiqiang/Google-inception-v3-tensorflow/blob/master/Google-inception-v3-tensorflow/car.png)
Sprots car, Sprots car(score = 0.93507)
Convertible(score = 0.01113)
Racer,race car,racing car (score = 0.00868)
Chain saw,chainsaw(score = 0.00298)
Car wheel(score = 0.00276)


Images/cat.jpg
![Alt text](https://github.com/ttanzhiqiang/Google-inception-v3-tensorflow/blob/master/Google-inception-v3-tensorflow/ cat.png)
Egyptian cat (sorce = 0.55336)
Tabby,tabby cat(score = 0.25701)
Tiger cat(score = 0.08283)
Lynx,catamount(score = 0.05683)
Hyena,hyaena(score = 0.00275)

Images/dog.jpg
![Alt text](https://github.com/ttanzhiqiang/Google-inception-v3-tensorflow/blob/master/Google-inception-v3-tensorflow/ dog.png)
Labrador retriever (score = 0.95145)
Golden retriever(score = 0.02065)
Tennis ball(score = 0.00399)
Beagle(score = 0.00093)
Saluki,gazelle hound(score = 0.00070)


Images/plane.jpg
![Alt text](https://github.com/ttanzhiqiang/Google-inception-v3-tensorflow/blob/master/Google-inception-v3-tensorflow/ plane.png)
Warplane,military plane(score = 0.81114)
Wing(score = 0.06260)
Aircraft carrier,carrier,flattop,attack aircraft carrier(score = 0.01210)
Projectile,missile(score = 0.01136)
Missile(score = 0.00972)

Images/seaside.jpg
![Alt text](https://github.com/ttanzhiqiang/Google-inception-v3-tensorflow/blob/master/Google-inception-v3-tensorflow/ seaside.png)
Seashore,coast,seacoast,sea(score = 0.85502)
Sandbar,sand bar(score= 0.08791)
Lakeside,lakeshore(score = 0.02546)
Sarong(score = 0.00745)
Coral reef(score = 0.00354)



Tensorflow-GPU-cnn-text-classification-tf
安装CUDA
	1.准备好NVIDIA的显卡，下载安装CUDA
https://developer.nvidia.com/cuda-downloads
2.安装好之后把CUDA安装目录下的bin和lib\x64添加Path环境变量中

安装cuDNN
1.cuDNN下载
https://developer.nvidia.com/rdp/cudnn-download
2.解压压缩包，把压缩包中bin,include,lib中的文件
分别拷贝到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0目录下对应目录中
3.把C:\Program Files\NVIDIA GPU ComputingToolkit\CUDA\v8.0\extras\CUPTI\libx64\
cupti64_80.dll
拷贝到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin

安装tensorflow-gpu
1.pip uninstall tensorflow
2.pip install tensorflow-gpu

数据集下载
http://www.robots.ox.ac.uk/~vgg/data/


训练模型
python D:/Tensorflow/tensorflow-master/tensorflow/examples/image_retraining/retrain.py ^
--bottleneck_dir bottleneck ^
--how_many_training_steps 200 ^
--model_dir D:/Tensorflow/inception_model/ ^
--output_graph output_graph.pb ^
--output_labels output_labels.txt ^
--image_dir data/train/
Pause


python D:/Tensorflow/slim/train_image_classifier.py ^
--train_dir=D:/Tensorflow/slim/model ^
--dataset_name=myimages ^
--dataset_split_name=train ^
--dataset_dir=D:/Tensorflow/slim/images ^
--batch_size=10 ^
--max_number_of_steps=10000 ^
--model_name=inception_v3 ^
Pause

模型中需要imagenet-vgg-verydeep-19，放在运行目录即可
https://pan.baidu.com/s/1hP1n5cfBYBbtDBVLFS2d_A

向一个训练过的系统输入图像，我们会得到一组概率值：每个训练过的类别都有一个，然后，系统会将图像归到概率最高的类。
概括地说，神经网络是计算单元的连接，能够从提供给它的一组数据中进行学习。
把多层神经网络堆叠在一起，我们就得到了深度神经网络。建立、训练和运行深度神经网络的过程，称为深度学习。
    第1步：下载预训练模型、计算图和脚本
    第2步：运行脚本找到最佳预测

对模型进行重新训练
第1步：设置图像文件夹
    这一步设计设计文件夹结构，好让TensorFlow能轻松获取这些类别。比如说识别房子、吉他、飞机、花以及动物。将各自的图像添加到其各自的文件夹中
第2步：运行重新训练脚本
python D:/Tensorflow/tensorflow-master/tensorflow/examples/image_retraining/retrain.py ^
--bottleneck_dir bottleneck ^
--how_many_training_steps 200 ^
--model_dir D:/Tensorflow/inception_model/ ^
--output_graph output_graph.pb ^
--output_labels output_labels.txt ^
--image_dir data/train/
Pause

![Alt text](https://github.com/ttanzhiqiang/Google-inception-v3-tensorflow/blob/master/bird.jpg)
[4 2 3 0 1]
animal (score = 0.43915)
plane(score = 0.20265)
flower(score = 0.13970)
house(score = 0.11118)
guitar(score = 0.10732)



![Alt text](https://github.com/ttanzhiqiang/Google-inception-v3-tensorflow/blob/master/flower.jpg)
[3 4 2 0 1]
flower(score = 0.98955)
plane(score = 0.00351)
animal(score = 0.00294)
house(score = 0.00246)
guitar(score = 0.00154)


![Alt text](https://github.com/ttanzhiqiang/Google-inception-v3-tensorflow/blob/master/guitar.jpg)
[1 4 3 2 0]
guitar(score = 0.97126)
plane(score = 0.00871)
flower(score = 0.00737)
house(score = 0.00650)
animal(score = 0.00616)


![Alt text](https://github.com/ttanzhiqiang/Google-inception-v3-tensorflow/blob/master/house.jpg)
[0 4 2 3 1]
house(score = 0.98758)
animal (score = 0.00486)
plane(score = 0.00311)
flower(score = 0.00293)
guitar(score = 0.00152)


![Alt text](https://github.com/ttanzhiqiang/Google-inception-v3-tensorflow/blob/master/plane.jpg)

[2 4 1 3 0]
plane(score = 0.96152)
animal (score = 0.01571)
guitar(score = 0.00852)
flower(score = 0.00715)
house(score = 0.00710)

