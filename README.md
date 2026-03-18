# Cardiac Fibrosis Detection 

## Overview

This project focuses on the automated segmentation of myocardial fibrosis in **DICOM CT scans** using image processing and deep learning techniques. The application provides an interactive interface for researchers to identify fibrotic tissue, analyze pixel intensity histograms, and generate precise masks.

A graphical user interface (GUI) built with **Tkinter** and **OpenCV** allows users to interactively select a Region of Interest (ROI) and visualize clinical data in real-time.

---

## Features

* **Interactive ROI Selection:** Draw and focus on specific heart regions for localized analysis.
* **Histogram Intensity Analysis:** Visualize pixel distributions to optimize thresholding.
* **Morphological Filtering:** Advanced noise reduction using opening operations and area-based filtering.
* **DICOM Integration:** Full support for Hounsfield Unit (HU) normalization and metadata processing.
* **Automated Result Saving:** Exporting original slices and generated binary masks for further clinical review.

---

## Technologies Used

### Programming Language
* **Python 3.x**

### Key Libraries & Tools
* **PyTorch / Deep Learning Architecture:** U-Net (Experimental)
* **OpenCV:** Image processing and ROI selection
* **PyDICOM:** Medical imaging data handling
* **Matplotlib:** Data visualization and histogram plotting
* **NumPy:** Mathematical matrix operations
