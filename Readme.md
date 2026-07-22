
This repository contains the code for the paper "Duration Dependence and Heterogeneity: Learning from Early Notice of Layoff" by Div Bhagia. The latest version of the paper can be found on this link: [https://arxiv.org/abs/2305.17344](https://arxiv.org/abs/2305.17344).

## Software Requirements

The project was built using Python 3.12.

You can install Python 3.12 using `pyenv`.
```bash
pyenv install 3.12
```
To create a virtual environment for this project using `pyenv`, you need `pyenv-virtualenv`. To create a new virtual environment, use the following command:
```bash
pyenv virtualenv 3.12.x env-notice
```
Here, `x` is the version of python you are using and `env-notice` is the name of the virtual environment. You can give it any name you like. Note that `pyenv` manages environments in a central location but allows you to "activate" them in specific directories. 

## Running the code

If you used `pyenv` to create a virtual environment as described above, you can clone the repository, install the required packages, and run all the code using the following commands in the terminal:
```bash
git clone https://github.com/divbhagia/notice.git
cd notice
pyenv activate env-notice
pip install -r requirements.txt
make 
```
If you did not create a virtual environment, just omit the `pyenv activate env-notice` command.

## Data
The paper uses the Displaced Worker Survey (DWS) of the Current Population Survey (CPS). The data is downloaded from IPUMS-CPS and is hosted on Dropbox at this [link](https://www.dropbox.com/scl/fo/r2gg07w5qy9kygd00uhbq/AFRzAmOEtgqU7uE2kauVTPw?rlkey=7rz8wj46r8gdrnjm9gbzdx47g&dl=0). The code will automatically download the raw data and process it to create the final dataset.

## Project structure

After running the code, the project directory will have the following structure:

```
.
├── data
│   ├── raw
├── draft
├── output
│   ├── quants
├── scripts
├── tests
└── utils
```

The code for the output for the paper is in the `scripts` folder. The `utils` folder contains helper functions and the `tests` folder contains tests for the helper functions. The `draft` folder contains a tex file that compiles all the figures and tables included in the paper.

The following tables lists the name of the scripts used to generate each figure and table in the main paper and the appendix:

Main paper:

| Figure/Table | Script |
|--------------|--------|
Table 1 | `03a_summary_stats.py` |
Table 2 | `03c_reg_tables.py` |
Figure 1 | `03b_hazard_plots.py` |
Table 3, Figure 2 | `04_estimation.py` |
Figure 3 | `05a_search_model.py` |

Appendix:

| Figure/Table | Script |
|--------------|--------|
Figure B1 | `05a_search_model.py`  |
Table C1, Figure C1 | `02_sample_and_ipw.py` |
Table C2 | `03a_summary_stats.py` |
Figures C2, C3 | `03d_add_desc.py` |
Table C3 | `03c_reg_tables.py`  |
Table C4, Figure C4 | `03d_add_desc.py` |
Figure C5 | `03d_add_desc.py` |
Figure C6 | `03b_hazard_plots.py`  |
Figures D1, D3, D4 | `04_estimation.py` |
Figure D2 | `04a_robust_notcats.py` |
Figures D5, D6 | `06_binning_duration.py` |
Figures E1, E2 | `07a_bounds_value.py`, `07b_bounds_est.py` |
Figures F1, F2 | `05b_sim_search.py` |



