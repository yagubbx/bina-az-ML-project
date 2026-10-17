# Start

Use Python 3.11 or 3.12. Run commands from the repository root.

```powershell
git clone https://github.com/yagubbx/bina-az-ML-project.git ML_Final_Project
cd ML_Final_Project
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you are already in the project folder, start with the environment command.

Download `house_sale.csv` from the dataset linked in `data/README.md`.
Place it in `data/` and rename it to `bina_az_sale.csv`.

```powershell
.\.venv\Scripts\python.exe -m src.run_all --check
.\.venv\Scripts\python.exe -m src.run_all
```

For an existing dataset with a different filename:

```powershell
.\.venv\Scripts\python.exe -m src.run_all --data "data/house_sale.csv"
```

Figures are generated in `report/figures/`. Machine-readable results are
generated in `report/figures/results/`. The console shows the final comparison.
Use `--quick` for a small smoke run; its outputs go to `report/figures/quick/`
and are not final benchmark results.

The checks use the installed project dependencies; pytest is not required.


