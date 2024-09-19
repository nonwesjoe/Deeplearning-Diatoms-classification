# Diatom classification ResNet34
- dataset: download from https://www.kaggle.com/datasets/siyuepu/diatom-datasets/data
- detail: 1042 species of diatoms including 30,000 + image in Electron microscopy.
- each specie in there folder named after them (1042 folders)
- tensorflow 2---train on kaggle GPU P100
# model architecture
- totally ResNet34 model with input shape (224,224,3)
# the accuary
- in test dataset 88%.
-  test in 5 groups of random 100 images,the accuracy 88%,88&,87%,91%,89%.
# sample picture
[sample1](sample.jpg)
[sample2](sample2.jpg)
