# CMPE 255 — Assignment 3

K-means clustering and the AutoGluon, NVIDIA RAPIDS, and PyCaret libraries.

## Notebooks

Each notebook includes its code, explanations, and saved execution outputs. The Drive links open the corresponding editable Colab copies.

The public repository notebooks can also be opened through Colab's GitHub tab using this repository URL.

| Part | Topic | Notebook | Colab | Video |
| --- | --- | --- | --- | --- |
| 1 | K-means and its variations | [Notebook](notebooks/01-kmeans/notebook.ipynb) | [Open](https://colab.research.google.com/drive/1VvqHxe4neAbb4UomQWfox5NJtkZeuxah) | [final_kmeans_zero_to_hero.mp4](https://youtu.be/9VRA44HFHqs) |
| 2 | AutoGluon capabilities | [Notebook](notebooks/02-autogluon-capabilities/notebook.ipynb) | [Open](https://colab.research.google.com/drive/18lcnIsmQaXRISkdp0atZfWFGRQZIunJ3) | [final_autogluon_capabilities_tour.mp4](https://youtu.be/esaNDn5biVo) |
| 3 | AutoGluon end-to-end ML and metrics | [Notebook](notebooks/03-autogluon-end-to-end/notebook.ipynb) | [Open](https://colab.research.google.com/drive/1qFp3tuhu-C0dVvQ4V6-XdeLT_1GNowXM) | [final_autogluon_zero_to_hero.mp4](https://youtu.be/FkQFIbohmxY) |
| 4 | NVIDIA RAPIDS and CPU/GPU comparisons | [Notebook](notebooks/04-rapids-cpu-comparison/notebook.ipynb) | [Open](https://colab.research.google.com/drive/15_4S9xnA2K5QM7XQT6WPt2KkloZWhG1Z) | [final_nvidia_rapids_zero_to_hero.mp4](https://youtu.be/xQ76GCqRXzc) |
| 5 | PyCaret capabilities | [Notebook](notebooks/05-pycaret-capabilities/notebook.ipynb) | [Open](https://colab.research.google.com/drive/1iiMgSPbzJ8hP5ZAG32Ibyx5Hobn8m2GD) | [final_pycaret_capbilities_tour.mp4](https://youtu.be/hzou_5Qcbg4) |
| 6 | PyCaret end-to-end ML and MLOps | [Notebook](notebooks/06-pycaret-mlops/notebook.ipynb) | [Open](https://colab.research.google.com/drive/1WUn6qLGa1UdfAtYqWHFctxiXufbcaDtA) | [final_pycaret_zero_to_hero.mp4](https://youtu.be/BDEAo6j8YtM) |

## Saved execution checks

| Part | Executed nonempty code cells | Saved PNG outputs |
| --- | ---: | ---: |
| 1 | 78 | 46 |
| 2 | 84 | 25 |
| 3 | 105 | 39 |
| 4 | 118 | 40 |
| 5 | 94 | 37 |
| 6 | 136 | 47 |
| Total | 615 | 234 |

All nonempty code cells have execution counts, and the saved notebooks contain no unhandled cell errors. Tables and printed results are saved alongside the plots. Blank code cells are excluded from the execution count.

## Reproducing the runs

1. Open the desired Colab copy, or upload the repository's `.ipynb` file to Colab.
2. Select the runtime described below and run its dependency setup cell. Restart the session once if packages were installed or replaced.
3. Run the remaining cells in order. Allow model fitting, benchmarks, and optional model downloads to finish.
4. Save the notebook with outputs enabled. Files written only to a temporary runtime directory must be downloaded separately if needed later.

Use a separate environment for each library family; PyCaret's pinned dependencies differ from those used by AutoGluon and RAPIDS.

| Notebook family | Runtime |
| --- | --- |
| K-means | Standard Colab CPU. Optional embedding/GPU branches report the method available in the runtime. |
| AutoGluon | Python 3.13 and AutoGluon 1.6.3. The saved CPU runs use macOS; the setup output records the platform. Text, image, and foundation-model sections may download checkpoints or use the explicitly labeled alternatives provided in the notebook. |
| RAPIDS | NVIDIA GPU runtime. The saved benchmark output identifies the GPU and library versions. Use one consistent CUDA package family; do not mix CUDA 12 and CUDA 13 RAPIDS distributions. |
| PyCaret | Python 3.11 and PyCaret 3.3.2. Use the reference's Colab runtime version `2025.07` if offered, or a compatible local Python 3.11 kernel. The saved runs use a separate Python 3.11 environment on macOS; the first setup output records the platform and version. |

The notebooks generate their main teaching datasets from fixed seeds. K-means also uses scikit-learn sample datasets and optional text/embedding downloads. Download-dependent branches describe their fallback behavior in the code and output. Timing, selected models, and stochastic model results can differ across hardware and library versions.

For a local environment, use the appropriate Python version and install either [AutoGluon requirements](requirements-autogluon.txt) or [PyCaret requirements](requirements-pycaret.txt). On Apple Silicon, LightGBM also needs the OpenMP runtime (`brew install libomp`).

## Notes on the reference notebooks

- Table-formatting helpers apply decimal formats to numeric columns while retaining text columns and caller-supplied format dictionaries.
- AutoGluon setup avoids repeatedly upgrading `setuptools` beyond the version allowed by AutoGluon's dependencies. Its model files use a dedicated temporary directory.
- AutoGluon per-model CPU requests stay within the four-core budget assigned to each fit.
- The end-to-end multimodal example uses the two-layer Google BERT checkpoint that works with the installed Transformers version.
- RAPIDS host conversions distinguish device arrays from pandas objects; pandas `.get(key)` is not a device-to-host transfer.
- PyCaret's plotting environment uses compatible IPython and Matplotlib versions and embeds figures in the notebook.
- The PyCaret MLflow example uses a compatible MLflow version and retrieves the named experiment explicitly.
- Deliberate failure demonstrations, optional integrations, and approximate alternatives are labeled in the corresponding cells. A fallback's output should be interpreted as the method that actually ran.

## Assignment reference

[Canvas assignment](https://sjsu.instructure.com/courses/1636688/assignments/7916965)

The notebooks are adapted from the six references linked in the assignment. Their original explanations and source links are retained inside the notebooks.
