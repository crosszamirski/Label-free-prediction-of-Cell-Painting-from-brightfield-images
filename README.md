# Label-Free Prediction of Cell Painting from Brightfield Images

Pytorch implementation of [Label-Free Prediction of Cell Painting from Brightfield Images](https://www.nature.com/articles/s41598-022-12914-x)  

We predict the Cell Painting image channels from a brightfield input using two models (U-Net and cWGAN-GP), and then extract the morphological features of interest. The model predictions are tested with a traditional segmentation-based feature-extraction approach, which allows us to explore evaluation methods of targeted biological relevance.



<img src="https://user-images.githubusercontent.com/88771963/161634139-15589be9-d13a-452c-8275-d5ad1278823d.jpg" width = "600">

<img src="https://user-images.githubusercontent.com/88771963/161633753-2778ea77-4e9e-4489-8576-bc28a34febd6.jpg" width="400" >

Note: we have not provided the image dataset due to AstraZeneca licenses but this may be available on reasonable request.

## Citation
If you find this work useful, please consider citing our paper:
```
@article {10.1038/s41598-022-12914-x,
	author = {Cross-Zamirski, Jan and Mouchet, Elizabeth and Williams, Guy and Sch{\"o}nlieb, Carola-Bibiane and Turkki, Riku and Wang, Yinhai},
	title = {Label-Free Prediction of Cell Painting from Brightfield Images},
	year = {2022},
	doi = {https://doi.org/10.1038/s41598-022-12914-x},
  URL = {https://www.nature.com/articles/s41598-022-12914-x#citeas},
	journal = {Sci Rep 12, 10001}
}
