# Case Studies in Data Science — Individual Task 1

Code for Part 1.3: at-risk student classification using Random Forest
and MLP neural network models.

## Data (not included in this repo — download from UCI)

- OULAD: https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset
- Student Performance: https://archive.ics.uci.edu/dataset/320/student+performance

## How to run

1. Download both datasets from the UCI links above and extract the CSVs
   (studentInfo.csv, studentVle.csv, studentAssessment.csv,
   student-mat.csv, student-por.csv) into one folder.
2. Update the `UP` path at the top of `analysis.py` to that folder.
3. Run: `python3 analysis.py`

Requires: pandas, scikit-learn, matplotlib.
