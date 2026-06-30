# py_tm2tnt: A Python GUI for Traditional Morphometric Data Conversion for Cladistics analysis

<div align="left">
  <img width="120" height="120" src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhbdwXB_EFg_UQ_wi24dN3EJ1MgsTapyelahD4VojYxY1EM9oOUa3Ryhh52_oK4gzG-koGDw75kIcgjuI8F5Y-fRC8auuLpTrTtg_6zImfoTZk_ZShDlOilkH8nLutZoF-1cqsP3A3G7dTlnCROGFA1Ds07fLYDnLjvjkAIRldPRE7IiI7rmbOr3v3dNaL6/w113-h113/Icon%20py_tm2tnt.png" alt="py_tm2tnt Logo">

  **Developed by: Jonathan Liria & Ana Soto-Vivas**

  **Neotropical Cladistic Biogeography Computing Lab (NCBC-Lab)**  
An academic initiative focused on developing high-performance computational tools for complex spatial analysis, biogeography, and systematic biology in the Neotropics.

  [![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Journal](https://img.shields.io/badge/Journal-RPB%20(2025)-green)](https://doi.org/10.15381/rpb.v32i2.30018)
</div>

---

## Overview

`py_tm2tnt` is a Python-based graphical user interface (GUI) application designed for evolutionary biologists and systematists who need to convert traditional morphometric data matrices into native format for **TNT (Tree Analysis using New Technologies)** to perform parsimony-based cladistic analyses. 

The software streamlines dataset preparation by including automated tools for calculating quantitative intervals, performing critical baseline statistical summaries, and formatting/exporting complex continuous matrices. This eliminates the need for manual text manipulation and ensures formatting compliance with TNT's data block expectations.

<div align="center">
  <img width="500" src="https://github.com/user-attachments/assets/7906b592-a69f-474c-bbbd-11e7f676b5b0" alt="py_tm2tnt Main Interface" style="border-radius: 6px; border: 1px solid #ddd; margin-top: 15px;">
  <p><em>Figure 1. Main graphical interface of py_tm2tnt for data uploading and statistical parameter configurations.</em></p>
</div>

---

## Video Tutorial

For a comprehensive walkthrough on data loading, configuration parameters, and practical export workflows, a video tutorial is available (optimized for Spanish-speaking users):

📺 **[Watch the py_tm2tnt Video Tutorial on YouTube](https://youtu.be/8DSTxQf49VE)**

---

## System Requirements & Prerequisites

Before launching `py_tm2tnt`, please ensure your local machine environment has the following components installed:

### System Environment
* **Python 3.x**

### Required Python Libraries
The application relies on standard and scientific computation frameworks:
```text
pandas         # Data matrix manipulation and alignment
tkinter        # Graphical user interface rendering window
numpy          # Multi-dimensional numerical array calculation
csv            # Standard spreadsheet table processing
scipy          # Advanced statistical calculations and functions
itertools      # High-performance iterator operations
collections    # Specialized container datatypes
```
---

### Repository structure

```text
py_tm2tnt/
│
├── CITATION.cff          # Machine-readable citation metadata file
├── LICENSE               # Full distribution text (MIT Open-Source terms)
├── README.md             # Repository documentation and landing guide
├── py_tm2tnt Manual.pdf  # Comprehensive user operations manual
└── py_tm2tnt_v4.0.py     # Main Python graphical application source code
```
---

### How to Cite

If this application saves you manual editing hours and assists you in compiling dataset blocks for your research projects, please cite our peer-reviewed work:   

Liria, J., & Soto-Vivas, A. 2025. «py_tps2tnt y py_tm2tnt: Dos programas en Python para procesamiento de datos morfométricos en análisis cladísticos con TNT». Revista Peruana de Biología, 32 (2): e30018. https://doi.org/10.15381/rpb.v32i2.30018   

---

### License

This project is licensed under the open-source MIT License - see the local repository LICENSE file for details.
