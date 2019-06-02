# ResidualDenseNetWork-CycleGAN-SuperResolution
Using Residual Dense Network as one of Generator and define another RDN within a downsampler

Pytorch implement: 
[Residual Dense Network for Image Super-Resolution](https://arxiv.org/pdf/1802.08797.pdf)  
[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf)  

Different with the paper 1, I use there RDBs(Residual dense block), every RDB has three dense layers. So ,this is a sample implement the RDN(Residual Dense Network) proposed by the author.

Different with the paper 2, I use paired training data to constrain the convert performance.


# Requirements
- python3.6
- pytorch >= 1.0
- torchvision >= 2.1
- opencv 


# Dateset
you need prepare [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) 
Download the High Resolution and (2x/3x/4x) Low Resolution
put them in ./DIV2K


# References
[ResidualDenseNetwork-Pytorch](https://github.com/lizhengwei1992/ResidualDenseNetwork-Pytorch) 

[pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
