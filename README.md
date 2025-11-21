# Atlas – Retail Patterns Dashboard

This package contains a Streamlit-based retail analytics dashboard combining UK basket-level association rules (UCI Online Retail II) and Australian macro-level retail turnover (ABS 8501.0). The repository is structured to allow immediate local execution.

---

## 1. Repository Structure

### **app.py**

Entry point. Defines global layout, sidebar navigation, CSS overrides, and routes to all tabs (Overview, Basket Patterns, Time Dynamics, Insights).

### **tabs/**

Folder containing all functional modules used by the dashboard.

* **overview.py** – Loads and visualises ABS retail turnover; KPIs, seasonality profile, YoY growth, industry breakdown, and data notes.
* **patterns.py** – Loads association rules; filtering, scatterplots, network graph, segment summaries, glossary, and rule preview.
* **time_tab.py** – Time-series analysis of ABS turnover; year sliders, moving averages, STL decomposition, and ABS–UK category comparison.
* **insights.py** – Converts rules into store-ready and analytics insights; integrates ABS context; generates evidence-based recommendation cards.

### **data/**

Folder containing all required datasets for computation.

* **ABS.csv** – ABS Retail Trade, Australia (8501.0). Parsed dynamically using header-agnostic detection.
* **all_segment_rules_with_industry.csv** – Association rules with segment and industry tags. Used by patterns.py and insights.py.
* **network.json** – Optional network structure file (not required for core functionality).

### **assets/**

Static files for branding and UI.

* **logo.png** – Sidebar and header branding.

### **requirements.txt**

Minimal dependency list for running the Streamlit app.

### **.gitignore**

Excludes virtual environments, cache directories, and temporary files.

---

## 2. How to Use the Dashboard Locally

### **Step 1. Download the Repository**

Download all files exactly as structured.

### **Step 2. Create a Virtual Environment**

```
python -m venv ass2
```

### **Step 3. Activate the Environment**

Windows:

```
ass2\Scripts\Activate.ps1
```

Mac/Linux:

```
source ass2/bin/activate
```

### **Step 4. Install Dependencies**

```
pip install -r requirements.txt
```

### **Step 5. Run the Application**

```
streamlit run app.py
```

### **Step 6. Access the Dashboard**

A local URL will be shown, e.g.

```
http://localhost:8501
```

---

## 3. Manual Post-Processing

No manual post-processing (Photoshop, Gimp, Illustrator, etc.) was required. All results are generated entirely by Streamlit, Plotly, and Python.

---

## 4. Notes on Data

* ABS data is parsed dynamically (auto-detected column names).
* Association rules are cleaned (ID stripping, case normalisation, removing markdown artefacts).
* No raw modification of source CSVs is performed inside the project folder.

---

End of README.
