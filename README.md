PhotoGAN
---
[![Build Status](https://travis-ci.org/AdarshRevankar/PhotoGAN.png?branch=master)](https://travis-ci.org/AdarshRevankar/PhotoGAN)
[![HitCount](http://hits.dwyl.com/AdarshRevankar/PhotoGAN.svg)](http://hits.dwyl.com/AdarshRevankar/PhotoGAN)

Generation of Photo Realistic Image using `GAN` and `SPADE`. This project demonstrates the process of `Image-to-Image Translation` in which image is transformed to another form of image.

Here our Goal is to Generate Photo Realistic Image, using the `SPADE` Architecture. This Architecture helps to Generate more Robust Images using `GAN`.

---
##### Setup
1. In-order to setup we require `python 3.x` version (Better with Anaconda Environment) with requirements specified in [requirements.txt](https://github.com/AdarshRevankar/PhotoGAN/blob/master/requirements.txt).
1. Download the `checkpoints` from [this](https://drive.google.com/open?id=1hHyGiQhM5pOIOCcpor7LzyUT5hiAZnGM) link.
1. Extract the `checkpoints` in the current directory ie. `PhotoGAN > Checkpoints`.
1. To test the output from the model some data are already provided, to test those write the following command in the conda environment
   
   `python test.py --name coco_pretrained --dataset_mode coco --dataroot datasets/coco_stuff/`

---
##### Visuals
Image to Image Synthesis is performed over the `Drawing (Left)` and the Realistic Image is `Generated (Right)`.
<table style="animation: ease-in-out">
    <tr>
        <td>
            <img src="https://user-images.githubusercontent.com/48080453/80318196-ef689f00-8825-11ea-9819-7a9ff20c9ed8.png" width="200xp" height="200xp" alt="Input Image"/>
        </td>
        <td>
            <img src="https://user-images.githubusercontent.com/48080453/80318201-fa233400-8825-11ea-8232-da120fa6d66a.png" width="200xp" height="200xp" alt="Output Image"/>
        </td>
    </tr>
</table>

---
##### Contributors
Special Thanks to project partners

1. `Akshaya M`
1. `Shubham Dogra`
1. `Adarsh Revankar`