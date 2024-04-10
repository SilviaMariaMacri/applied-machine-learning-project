# Applied Machine Learning project

This repository has been created for the final project of the Applied Machine Learning course at University of Bologna, academic year 2022/2023.
It contains some data exploration and clustering analysis of genomic data of patients affected by breast cancer disease.

It consists of three jupiter notebook files, in which the data are explored and analysed, one .py file, whose module is imported and used during the analysis, and two folders containing some intermediate results of the data processing/analysis:

* **data.ipynb**

  describes the data chosen for the analysis; they have been downloaded from the public databases https://www.cbioportal.org/study/summary?id=brca_metabric, https://cancer.sanger.ac.uk/census and https://panelapp.genomicsengland.co.uk/panels/158/
* **data_processing.ipynb**

  processes the input raw data described in the previous notebook in order to extract the most interesting characteristics and convert them in an analyzable format
* **analysis.ipynb**

  performs cluster analysis on preprocessed data       
* **autoencoder.py**

  defines the autoencoder class, imported in the previously described notebook for the analysis
* **data**           

  folder containing two files obtained after the processing performed in data.ipynb and data_processing.ipynb; the raw data are not present due to memory limit
* **ae_results**

  folder containing the files obtained during the autoencoder training in analysis.ipynb
