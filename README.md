# focus_winterholiday_project
项目简介：在寒假期间了解了一些机器学习的算法，并完成了CNN神经网络的搭建以及训练，成功在本地部署了YOLO_v11，创建了自己的数据集，并且在官方模型的基础上完成了训练。
注意点：运行不同项目时，要注意python，pytorch，cuda，labelimg等等版本的适配（使用anaconda搭建虚拟环境很方便）

任务点1：cnn.py和dnn.py是手写数字识别的训练代码,index.html是自我介绍页面。
任务点2：CNN手写数字识别的函数下降曲线<img width="2559" height="1599" alt="屏幕截图 2026-01-27 125647" src="https://github.com/user-attachments/assets/f73991e5-4da1-4879-8d85-bb998087998f" />
任务点3：best.pt是依据官方的coco128数据集，在yolo11n.pt预训练模型上训练出来的，训练过程如图<img width="1918" height="1050" alt="屏幕截图 2026-01-28 153234" src="https://github.com/user-attachments/assets/7828c33c-7774-40f9-9dd2-56e79f411ba9" />
![test](https://github.com/user-attachments/assets/14b22037-1277-4c27-bcef-bbc9ca90646c)
![test](https://github.com/user-attachments/assets/efe9c6d1-5dda-45ad-aaa7-1f844da4fccd)这是训练的best.pt模型的识别效果
我之后使用LabelImg标注了3张图片作为自己的数据集，在best.pt的基础上训练出了新一版best.pt

这是训练用的三张图片以及训练过程
<img width="2057" height="1277" alt="屏幕截图 2026-01-28 185622" src="https://github.com/user-attachments/assets/41bd0486-54f8-44b5-81f6-a3f13a15e5cb" />
训练好的结果保存在这个文件夹中
<img width="1662" height="1036" alt="image" src="https://github.com/user-attachments/assets/8a1a49c1-9415-4d83-b8a9-e6830efb6b18" />
用自己训练的模型实验了一下效果,效果很烂，数据量太小了。
