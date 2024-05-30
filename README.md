# Spatial Data Machine Learning Model

This project demonstrates how to download, process, and analyze geospatial data using machine learning models. The project includes downloading data from Google Drive, processing raster and vector data, and applying Random Forest classifiers to make predictions.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Overview](#project-overview)
- [Data Preparation](#data-preparation)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Visualization](#visualization)

## Installation

To run this project, you'll need to install the required dependencies. The dependencies include:

- `py7zr`
- `geopandas`
- `rasterio`
- `scikit-learn`
- `imbalanced-learn`

You can install them using `pip`:

```bash
pip install py7zr geopandas rasterio scikit-learn imbalanced-learn
```
## Usage

- Clone the repository:
```bash
git clone https://github.com/shivam-nath-1003/Spatial-ML-Pipeline.git
```
- Ensure you have Jupyter Notebook installed and open the notebook:
```bash
jupyter notebook SpatialData_ML_model.ipynb
```
- Run the notebook cells sequentially to download data, process it, and train models.

## Project Overview
The project consists of several steps:

- Downloading and extracting geospatial data from Google Drive.
- Processing raster and vector data using libraries such as rasterio and geopandas.
- Training and evaluating machine learning models using Random Forest classifiers.
- Visualizing the results with matplotlib.

## Data Preparation
The data is downloaded from a Google Drive link and extracted using the py7zr library. The data includes raster files (.tif) and a vector file (.gpkg).

```python
dl_from_gdrive('1ai6QR_YQTPDRsDQ0s8AZg_c4dq4_E64s', 'tas_sn_w_datasets.7z')
with py7zr.SevenZipFile('/content/tas_sn_w_datasets.7z', mode='r') as z:
    z.extractall(r'/content/')
```

## Model Training and Evaluation
The project uses Random Forest classifiers to predict spatial data. The data is prepared by stacking raster data into a 3D numpy array and converting it into a 2D tabular format for training.

```python
model1 = RandomForestClassifier(n_jobs=-1)
model1.fit(X_train, y_train)
```
### Undersampling and Checkerboard Pattern
To handle imbalanced classes, the RandomUnderSampler from imbalanced-learn is used. A checkerboard pattern is created to split the data into different regions for cross-validation.

```python
rus = RandomUnderSampler(random_state=42)
checker = make_checkerboard(data[0].shape, (400,400))
```
## Visualization
The results are visualized using matplotlib, showing ROC curves, probability maps, and clustered points.

```python
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(fpr, tpr, label='AUC={}'.format(round(roc_auc,2)))
plt.show()
```
## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.
