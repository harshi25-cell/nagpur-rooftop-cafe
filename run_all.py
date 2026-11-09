"""python run_all.py : regenerates the mock survey and prints the main results."""
from pathlib import Path
import pandas as pd
import generate_mock_survey, analytics as an
generate_mock_survey.main()
R=an.run(pd.read_csv(Path(__file__).parent/"data"/"Harshita_Chopda_Survey_Data.csv"))
print(R["log"]);c=R["clf"];print(c["best"],c["cm"].tolist(),round(c["auc"],3))
print(R["clu"]["profile"][["segment","tier","people","conversion"]]);print(R["arm"]["launch"]);print(R["reg"]["best"],round(R["reg"]["mae"]));print(R["biz"]["verdict"])
