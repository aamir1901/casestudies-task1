# Case Studies in Data Science — Individual Task 1, Part 1.3

Predicting at-risk students across two public education datasets, using a random
forest and an MLP neural network. Supports the analysis reported in Section 3 of
the submitted report.

## Datasets

Not included in this repository — download from UCI and place the CSVs in `data/`:

- **OULAD** — https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset
  (needs `studentInfo.csv`, `studentVle.csv`)
- **Student Performance** — https://archive.ics.uci.edu/dataset/320/student+performance
  (needs `student-mat.csv`, `student-por.csv`)

```
.
├── Task1_Part1_3_Wangde.ipynb
└── data/
    ├── studentInfo.csv
    ├── studentVle.csv
    ├── student-mat.csv
    └── student-por.csv
```

## Running

```bash
pip install pandas scikit-learn matplotlib jupyter
jupyter notebook Task1_Part1_3_Wangde.ipynb
```

Run the cells in order. Runtime is two to four minutes, almost all of it spent
aggregating the 10.65 million-row VLE interaction log. A fixed random seed (42)
makes every reported figure reproducible.

## Results

| Dataset | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| OULAD | Random forest | 0.880 | 0.928 | 0.837 | 0.880 | 0.947 |
| OULAD | Neural network (MLP) | 0.881 | 0.906 | 0.864 | 0.884 | 0.947 |
| Student Performance | Random forest | 0.785 | 0.511 | 0.522 | 0.516 | 0.826 |
| Student Performance | Neural network (MLP) | 0.799 | 0.600 | 0.261 | 0.364 | 0.787 |

Recall is reported for the at-risk class, which is the metric the comparison
turns on: a false negative is a struggling student who is never offered support.

## References

Cortez, P. and Silva, A. (2008) *Using data mining to predict secondary school student performance*. Proceedings of the 5th Annual Future Business Technology Conference, Porto, 5–12.

Kuzilek, J., Hlosta, M. and Zdrahal, Z. (2017) 'Open University Learning Analytics dataset', *Scientific Data*, 4, 170171. doi:10.1038/sdata.2017.171

Page, D. (2007) *Evaluating machine learning methods*. Lecture slides, CS 760, University of Wisconsin–Madison.

Pedregosa, F. et al. (2011) 'Scikit-learn: machine learning in Python', *Journal of Machine Learning Research*, 12, 2825–2830.
