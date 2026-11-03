# Pancreatic Head Cancer Survival Prediction Using Machine Learning

This repository provides the machine learning implementation associated with the research article:

**Prognosticating Outcome in Pancreatic Head Cancer With the Use of a Machine Learning Algorithm**

**Published in:** *Technology in Cancer Research & Treatment*, 2021, Volume 20.

**Paper:** https://doi.org/10.1177/15330338211050767

## Overview

This study investigates the application of supervised machine learning algorithms to predict survival outcomes in patients with pancreatic head adenocarcinoma undergoing pancreaticoduodenectomy (Whipple procedure).

The objective was to identify significant prognostic factors and develop a machine learning model capable of classifying patients based on their survival outcomes following surgery.

The study involved:

- Retrospective analysis of 93 patients.
- Evaluation of 50 potential prognostic variables.
- Selection of 11 prognostic features using statistical analysis, recursive feature elimination, and stability selection.
- Development of a nonlinear Support Vector Machine (SVM) classifier.
- Comparison with Random Forest and Naïve Bayes classifiers.

## Methodology

The proposed machine learning framework consists of the following stages:

1. **Data Preparation:** Collection and preprocessing of clinical, pathological, imaging, and treatment-related characteristics.
2. **Feature Selection:** Identification of important prognostic variables through statistical significance testing, recursive feature elimination, and stability selection.
3. **Model Development:** Training of supervised machine learning classifiers to distinguish between patients surviving less than two years and those surviving more than two years after surgery.
4. **Model Evaluation:** Assessment of predictive performance using leave-one-out cross-validation (LOOCV), testing accuracy, sensitivity, and specificity.

## Results

The nonlinear SVM classifier demonstrated the highest testing accuracy among the evaluated models.

| Machine Learning Model | Testing Accuracy |
|---|---:|
| Support Vector Machine (SVM) | 75.26% |
| Random Forest | 67.33% |
| Naïve Bayes | 74.69% |

The selected SVM model achieved:

- **Accuracy:** 75.26%
- **Sensitivity:** 41.9%
- **Specificity:** 97.5%

Sensitivity and specificity refer to the prediction of survival shorter than two years.

## Research Significance

This research demonstrates the potential of machine learning for integrating multiple clinical and pathological factors to support survival prognostication in pancreatic head cancer.

The study provides a foundation for developing more comprehensive prognostic prediction models. However, the limited sample size and inclusion of postoperative variables mean that further validation and development are required before clinical application.

## Citation

If you use or adapt the code or methodology associated with this research, cite the following publication:

Baig, Z., Abu-Omar, N., Khan, R., Verdiales, C., Frehlick, R., Shaw, J., Wu, F.-X., & Luo, Y. (2021). Prognosticating outcome in pancreatic head cancer with the use of a machine learning algorithm. *Technology in Cancer Research & Treatment, 20*, 15330338211050767.

https://doi.org/10.1177/15330338211050767

### BibTeX

```bibtex
@article{baig2021pancreatic,
  title={Prognosticating Outcome in Pancreatic Head Cancer
         With the Use of a Machine Learning Algorithm},
  author={Baig, Zarrukh and Abu-Omar, Nawaf
          and Khan, Rayyan and Verdiales, Carlos
          and Frehlick, Ryan and Shaw, John
          and Wu, Fang-Xiang and Luo, Yigang},
  journal={Technology in Cancer Research & Treatment},
  volume={20},
  pages={15330338211050767},
  year={2021},
  doi={10.1177/15330338211050767}
}
```

## Usage and Permissions

This repository is intended for academic, educational, and non-commercial research purposes.

- Users are required to cite the associated publication when using or adapting this work.
- Commercial use or redistribution for commercial purposes requires prior written permission from the code copyright holder.
- The implementation is provided for research purposes and is not intended for clinical decision-making.

## Disclaimer

This research implementation is provided for scientific and educational purposes only. It has not been established as a clinically validated tool and must not be used independently for patient prognosis, treatment selection, or clinical decision-making.
