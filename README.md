# Automated Cardiac Coverage Assessment in Cardiovascular MRI

This is a PyTorch implementation of our paper accepted in Medical Physics:

[Automated Cardiac Coverage Assessment in Cardiovascular Magnetic Resonance Imaging using an Explainable Recurrent 3D Dual-Domain Convolutional Network](http://doi.org/10.1002/mp.17411)

## Proposed Framework

You can see the overall framework for cardiac coverage assessment which has been used in this study.

<p align="center">
  <img src="https://github.com/mohammadhashemii/Saliency-Detection/blob/main/images/framework.png">	
</p>

## Dual-domain Convolutional-based model for apical/basal detection

Figure below is an extension of the dual-domain convolutional baseline model based on a recurrent structure that identifies the presence/absence of the basal/apical slices.

<p align="center">
  <img src="https://github.com/mohammadhashemii/Saliency-Detection/blob/main/images/dual-domain.png">	
</p>

## Salient Region Detection Model

After training the 3D dual-domain convolutional baseline model and performing the steps related to examining the interpretability of this model, and training the two U-Net models based on the most effective super-pixel obtained to identify the basal/apical slice, the proposed model can be used to extract the salient region of new stacks. The codes for this section can be found in [`segmentation/`](https://github.com/mohammadhashemii/Saliency-Detection/tree/main/segmentation)


If you find this repo helpful, we would appreciate it if you could cite our paper:

```
Nabavi S, Hashemi M, Ebrahimi Moghaddam M, Abin AA, Frangi AF.
Automated cardiac coverage assessment in cardiovascular magnetic resonance imaging using an explainable recurrent 3D dual-domain convolutional network. Med Phys. 2024;1-15.
https://doi.org/10.1002/mp.17411
```


