# GeoVision-AI

## Multispectral Satellite Image Semantic Segmentation using U-Net

GeoVision-AI is a deep-learning based semantic segmentation project for generating pixel-level land-cover maps from multispectral Sentinel-2 satellite imagery.

The model classifies each pixel into five land-cover categories:

- Vegetation
- Built-up
- Water
- Agriculture
- Bare land

The project uses six Sentinel-2 spectral bands instead of conventional RGB imagery:

- B2 — Blue
- B3 — Green
- B4 — Red
- B8 — Near Infrared (NIR)
- B11 — Short-Wave Infrared (SWIR)
- B12 — Short-Wave Infrared (SWIR)

A U-Net architecture implemented in PyTorch is trained to perform pixel-wise classification on 256 × 256 multispectral image tiles.

---

## Project Motivation

Land-cover mapping is an important task in remote sensing and geospatial analysis.

Traditional image classification methods often assign a single label to an entire image or region. Semantic segmentation provides a more detailed representation by assigning a class to every individual pixel.

This project focuses on automated land-cover segmentation from multispectral satellite imagery, with particular attention to the challenge of identifying heterogeneous bare-land regions.

---

## Problem Statement

Develop a semantic segmentation model capable of classifying every pixel in Sentinel-2 satellite imagery into five land-cover classes:

| Class ID | Land-cover class |
|----------|------------------|
| 0 | Vegetation |
| 1 | Built-up |
| 2 | Water |
| 3 | Agriculture |
| 4 | Bare land |

The system should use multispectral information to distinguish visually and spectrally similar land-cover categories.

---

## Dataset

The dataset consists of 256 × 256 pixel Sentinel-2 image tiles collected from multiple Indian cities.

The project uses eight cities:

- Ahmedabad
- Bengaluru
- Chennai
- Delhi
- Hyderabad
- Jaipur
- Kolkata
- Mumbai

Each TIFF tile contains seven channels:

```text
B2
B3
B4
B8
B11
B12
Label
The first six channels are used as model inputs, while the seventh channel contains the ground-truth segmentation labels.

Input shape
6 × 256 × 256
Output shape
5 × 256 × 256
Why Multispectral Data?
Instead of using only RGB channels, this project uses six spectral bands.

The additional NIR and SWIR bands provide information that can help distinguish land-cover types with similar visual appearance.

The selected bands are:

B2  → Blue
B3  → Green
B4  → Red
B8  → Near Infrared
B11 → SWIR
B12 → SWIR
This is particularly useful for separating vegetation, built-up surfaces, water, agriculture, and different types of bare land.

Model Architecture
The segmentation model is based on the U-Net architecture.

Input
  │
  ▼
6-band Sentinel-2 image
256 × 256
  │
  ▼
Encoder
  │
  ├── Feature extraction
  ├── Downsampling
  └── Context learning
  │
  ▼
Bottleneck
  │
  ▼
Decoder
  │
  ├── Upsampling
  ├── Skip connections
  └── Spatial reconstruction
  │
  ▼
5-channel output
  │
  ▼
Pixel-wise land-cover prediction
The implementation is written using PyTorch.

The model contains approximately:

31 million parameters
Data Preprocessing
Each tile is loaded from a TIFF file.

The preprocessing pipeline performs:

Load the seven-band TIFF.

Separate the six image bands from the label band.

Replace invalid numerical values.

Convert image data to float32.

Scale Sentinel-2 reflectance values by 10000.

Convert the image from:

H × W × C
to:

C × H × W
Convert labels to integer class IDs.

The final model input is:

6 × 256 × 256
Training
The dataset is divided into:

Training set
Validation set
Test set
The model was trained for:

30 epochs
Two important checkpoints were evaluated:

models/unet_next_miou_best.pth
models/unet_next_bareland_best.pth
The first checkpoint was selected based on validation mean IoU.

The second checkpoint was selected with greater emphasis on bare-land performance.

Results
The best overall test performance was obtained using:

unet_next_miou_best.pth
The model was evaluated on:

219 test tiles
Overall performance
Metric	Score
Pixel Accuracy	84.51%
Mean IoU	68.55%
Macro Precision	77.24%
Macro Recall	84.57%
Macro F1	79.45%
Per-Class Performance
Class	Precision	Recall	IoU	F1
Vegetation	80.57%	78.80%	66.21%	79.67%
Built-up	91.56%	86.86%	80.42%	89.15%
Water	95.82%	99.06%	94.96%	97.41%
Agriculture	80.88%	80.07%	67.33%	80.47%
Bare land	37.35%	78.08%	33.81%	50.53%
Confusion Matrix
The test-set confusion matrix was:

                    Predicted
                Veg    Built   Water   Agric   Bare

Actual Veg    3567187  232758   60241  574007   92768
Actual Built   252384 3234746    1856   68450  166486
Actual Water     7863     845 2301585    6518    6650
Actual Agric   593498   32947   30231 2758169   29834
Actual Bare      6748   31636    7976    3129  176313
The major challenge is bare-land classification.

Bare-Land Analysis
Bare land was the most difficult class.

The best overall model achieved:

Bare-land IoU: 33.81%
Bare-land Recall: 78.08%
Bare-land Precision: 37.35%
This indicates that the model detects many bare-land pixels but also produces a relatively large number of false positives.

Because overall accuracy alone did not explain this behavior, additional failure-case analysis was performed.

Failure Analysis
The analysis investigated:

Per-tile bare-land IoU

Bare-land area

Spectral characteristics

Confusion with other classes

Successful vs failed tiles

Spatial coherence of bare-land regions

Important finding
Bare-land performance is strongly related to the amount of bare land present in a tile.

For tiles containing bare land:

Pearson correlation between bare-land area and IoU:
0.4819

Spearman correlation:
0.5875
This suggests a moderate positive relationship between the amount of bare land available in a tile and segmentation performance.

Bare-Land Performance by Area
Ground-truth bare pixels	Mean IoU
1–50	13.56%
51–100	14.42%
101–250	17.83%
251–500	23.86%
501–1,000	27.65%
1,001–2,500	34.94%
2,501–5,000	43.08%
5,001+	51.65%
The results indicate that small and fragmented bare-land regions are substantially harder to segment than large spatially coherent regions.

Spectral Analysis
Successful and failed bare-land regions were compared using the six input spectral bands.

A major observation was that successful bare-land regions generally had higher SWIR reflectance.

For ground-truth bare-land pixels:

Band	Failed	Successful
B2	0.094	0.159
B3	0.128	0.213
B4	0.126	0.259
B8	0.161	0.290
B11	0.166	0.370
B12	0.141	0.375
The strongest differences occurred in:

B11 — SWIR
B12 — SWIR
This suggests that the model learns a particular spectral representation of bare land very effectively, while struggling with other spectral forms.

Example Failure Cases
Several difficult test tiles were analyzed in detail.

Chennai tile 19
Ground-truth bare pixels: 102
Predicted bare pixels: 41
Bare-land IoU: 0.00
Most ground-truth bare pixels were classified as built-up.

Ahmedabad tile 178
Ground-truth bare pixels: 301
Predicted bare pixels: 4
Bare-land IoU: 0.0033
Most bare pixels were classified as water.

Kolkata tile 39
Ground-truth bare pixels: 184
Predicted bare pixels: 41
Bare-land IoU: 0.0045
The dominant confusion was with built-up and water.

Successful Bare-Land Cases
Chennai tile 157
Ground-truth bare pixels: 1,528
Predicted bare pixels: 1,576
IoU: 0.9522
Chennai tile 124
Ground-truth bare pixels: 2,073
Predicted bare pixels: 2,451
IoU: 0.7981
These examples demonstrate that the model can segment large, spectrally distinctive bare-land regions very effectively.

Key Findings
The analysis produced several important observations:

1. Bare land is the primary segmentation challenge
Water and built-up areas are classified much more reliably than bare land.

2. Bare-land detection is not simply a threshold problem
Changing the probability threshold did not meaningfully change the predictions because pixels above the threshold were already generally the argmax class.

3. Spectral diversity matters
The model performs very well on some spectral forms of bare land but poorly on others.

4. SWIR information is important
B11 and B12 showed particularly strong differences between successful and failed bare-land regions.

5. Spatial area matters
Large, spatially coherent bare-land regions are significantly easier for the U-Net to segment.

6. Failure cases are geographically and spectrally heterogeneous
Different cities exhibited different failure modes, including confusion with:

Built-up
Water
Vegetation
Dashboard
The project includes an interactive Streamlit analytics dashboard.

The dashboard provides:

Overall model performance

Class-wise metrics

Confusion matrix

Bare-land analysis

Bare-land area vs IoU analysis

Failure-case analysis

Successful examples

Visual diagnostic results

Run the dashboard with:

python -m streamlit run dashboard/app.py
Then open:

http://localhost:8501
Project Structure
GeoVision-AI/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   ├── raw1/
│   └── splits/
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
│
├── models/
│   ├── unet_next_miou_best.pth
│   ├── unet_next_bareland_best.pth
│   └── ...
│
├── outputs/
│   ├── *.csv
│   ├── *.png
│   └── final_visualizations/
│
├── src/
│   ├── dataset.py
│   ├── unet.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── losses.py
│   ├── split_dataset.py
│   ├── analyze_bareland_area_vs_iou.py
│   ├── analyze_selected_tiles.py
│   ├── analyze_training_bareland.py
│   ├── create_final_visualizations.py
│   └── ...
│
├── requirements.txt
├── .gitignore
└── README.md
Installation
Create a Python environment and install the required packages.

pip install -r requirements.txt
PyTorch should be installed with the appropriate CUDA configuration for the target machine when GPU acceleration is required.

Running Evaluation
After preparing the dataset and model checkpoint:

python src/evaluate.py
Running Inference
Inference can be performed using:

python src/inference.py
The system generates a predicted segmentation mask and visualization for the input tile.

Visual Analysis
The repository contains visualization outputs for:

Successful bare-land segmentation

Bare-land failure cases

Area-vs-IoU relationship

Test-tile diagnostics

Prediction visualizations

These outputs are useful for understanding model behavior beyond aggregate metrics.

Future Improvements
Potential next steps include:

Increasing the diversity of bare-land training examples

Oversampling rare bare-land regions

Using targeted hard-example mining

Adding spatial/contextual features

Experimenting with attention-based U-Net variants

Testing stronger data augmentation

Exploring focal/Tversky-style losses

Improving class balancing

Evaluating larger contextual windows

Testing alternative segmentation architectures

The main objective is to improve generalization across different spectral and spatial forms of bare land.

Technical Challenge
How can semantic segmentation models reliably distinguish spectrally and spatially heterogeneous bare-land regions from visually similar land-cover classes such as built-up areas, agriculture, and water?

This became the central analytical challenge after evaluating the initial model.

Conclusion
GeoVision-AI demonstrates an end-to-end workflow for multispectral satellite image semantic segmentation:

Sentinel-2 imagery
       ↓
Data preprocessing
       ↓
Multispectral U-Net
       ↓
Pixel-wise prediction
       ↓
Quantitative evaluation
       ↓
Failure-case analysis
       ↓
Spectral + spatial investigation
       ↓
Future model improvements
The project achieved approximately:

84.5% Pixel Accuracy
68.6% Mean IoU
while providing a detailed analysis of the primary weakness of the model: bare-land segmentation.

The project goes beyond simply training a model by investigating why the model fails, using quantitative metrics, spectral analysis, spatial analysis, and visual diagnostics.


### Then upload the README

Since your GitHub repository is already connected, run:

```powershell
cd D:\GeoVision-AI
notepad README.md
Paste the README above, save, then:

git add README.md
git commit -m "Add project documentation"
git push
