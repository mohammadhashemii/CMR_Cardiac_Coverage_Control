# Automated Cardiac Coverage Assessment in Cardiovascular Magnetic Resonance Imaging using an Explainable Salient Region Detection Model

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
