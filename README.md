# ROCT-Net: A new ensemble deep convolutional model with improved spatial resolution learning for detecting common diseases from retinal OCT images

Optical coherence tomography (OCT) imaging is a well-known technology for visualizing retinal layers and helps ophthalmologists to detect possible diseases. Accurate and early diagnosis of common retinal diseases can prevent the patients from suffering critical damages to their vision. Computer-aided diagnosis (CAD) systems can significantly assist ophthalmologists in improving their examinations. This paper presents a new enhanced deep ensemble convolutional neural network for detecting retinal diseases from OCT images. Our model generates rich and multi-resolution features by employing the learning architectures of two robust convolutional models. Spatial resolution is a critical factor in medical images, especially the OCT images that contain tiny essential points. To empower our model, we apply a new post-architecture model to our ensemble model for enhancing spatial resolution learning without increasing computational costs. The introduced post-architecture model can be deployed to any feature extraction model to improve the utilization of the feature map’s spatial values. We have collected two open-source datasets for our experiments to make our models capable of detecting six crucial retinal diseases: Age-related Macular Degeneration (AMD), Central Serous Retinopathy (CSR), Diabetic Retinopathy (DR), Choroidal Neovascularization (CNV), Diabetic Macular Edema (DME), and Drusen alongside the normal cases. Our experiments on two datasets and comparing our model with some other well-known deep convolutional neural networks have proven that our architecture can increase the classification accuracy up to 5%. We hope that our proposed methods create the next step of CAD systems development and help future researches.

# Dataset

Most of the OCT datasets include few images of 2 or 3 types of disease. In order to expand our investigations and validate our results, we ran our experiments on two datasets.
The first is the [Kermany dataset](https://data.mendeley.com/datasets/rscbjbr9sj/3) which is a large dataset containing 108,312 images (37,206 with CNV, 11,349 with DME, 8,617 with DRUSEN, and 51,140 Normal) from 4,686 patients for training and 1000 images from 633 patients for evaluating the models.</br>
The second dataset is named [OCTID](https://www.sciencedirect.com/science/article/abs/pii/S0045790618330842) and belongs to the waterloo university. It includes 572 images of 5 classes: Normal, AMD, CSR, DR, and MH. In the Kermany dataset, each patient has several OCT images. We selected one image of each patient in the training set to reduce the data size, but we kept the same test set for evaluation. In the second dataset, we allocated 20% of images of each class for testing, and the rest were used for training the models. The details of the data distribution of our work are
presented in the table.1.

<p>
  <caption>A Caption</caption>
  Dataset | Number of Classes | Training Images | Testing Images | AMD | CSR | DR | MH | CNV | DME | DRUSEN | NORMAL |
------------ | ------------- | ------------- | -------------  | -------------  | -------------  | -------------  | -------------  | -------------  | -------------  | -------------  | -------------  
 [Kermany](https://data.mendeley.com/datasets/rscbjbr9sj/3)| 4 | 3213 | 1000 | N/A | N/A | N/A | N/A | 791/250 | 709/250 | 713/250 | 1000/250 |
[OCTID](https://www.sciencedirect.com/science/article/abs/pii/S0045790618330842) | 5 | 459 | 113 | 44/11 | 82/20 | 86/21 | 85/20 | N/A | N/A | N/A | 165/41 
  <caption>
<p>
