import pandas as pd
import numpy as np
from models.var_model import fit_country_var
from models.monte_carlo import _next_state, _recalculate_debt

df = pd.read_csv("data/processed/panel_clean.csv")
br = df[df["country"] == "Brazil"]
fit = fit_country_var(br)

history = np.array([fit.history])
print("Variables:", fit.variables)
for step in range(5):
    next_values = _next_state(history, fit, np.zeros((1, 6)), exog_value=0.0)
    next_values = _recalculate_debt(next_values, history, fit)
    print(f"Year {2025+step}: {next_values[0]}")
    history = np.concatenate([history[:, 1:], np.expand_dims(next_values, 1)], axis=1)

