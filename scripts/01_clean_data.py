##########################################################
# Housekeeping
##########################################################

# Import external libraries
import numpy as np
import pandas as pd

# Import custom functions
from utils.dataclean import get_occ, get_broad_occ
from utils.dataclean import get_ind, get_broad_ind
from utils.dataclean import indicator, group_dur

# Import parameters
from utils.config import IPUMS_DIR, DATA_DIR
from utils.config import RAW_DATA_DIR, SELECTED_VARS

##########################################################
# Load data, remove armed forces and keep select variables
##########################################################

# Load CPS data
df = pd.read_csv(f"{IPUMS_DIR}/cps_raw.csv")

# Filter armed forces & children under 14/15
df = df[df["popstat"] == 1]

# Indicator for dws
df["dws"] = np.where((df["dwstat"] == 1) & (df["dwresp"] == 2), 1, 0)

# Keep selected variables (optional)
df = df[SELECTED_VARS]

# Declare all variables as float
df = df.astype(float, errors="ignore")

##########################################################
# Missing values & top codes for continous variables
##########################################################

# CPS variables
df["uhrsworkt"] = np.where(df["uhrsworkt"] >= 997, np.nan, df["uhrsworkt"])
df["uhrswork1"] = np.where(df["uhrswork1"] == 0, np.nan, df["uhrswork1"])
df["uhrswork1"] = np.where(df["uhrswork1"] >= 997, np.nan, df["uhrswork1"])
df["faminc"] = np.where(df["faminc"] >= 995, 995, df["faminc"])
df["durunemp"] = np.where(df["durunemp"] == 999, np.nan, df["durunemp"])
df["durunem2"] = np.where(df["durunem2"] == 99, np.nan, df["durunem2"])
df["numjob"] = np.where(df["numjob"] == 0, np.nan, df["numjob"])

# Tenure at lost job
df["dwyears"] = np.where(df["dwyears"] > 99, np.nan, df["dwyears"])
df["dwyears"] = np.where(df["dwyears"] > 24, 24, df["dwyears"])

# Earnings at lost and current job
df["dwweekl"] = np.where(df["dwweekl"] > 9999, np.nan, df["dwweekl"])
df["dwweekc"] = np.where(df["dwweekc"] > 9999, np.nan, df["dwweekc"])
df["dwweekl"] = np.where(df["dwweekl"] == 0, np.nan, df["dwweekl"])
df["dwweekc"] = np.where(df["dwweekc"] == 0, np.nan, df["dwweekc"])

# Other DWS variables
df["dwwksun"] = np.where(df["dwwksun"] >= 996, np.nan, df["dwwksun"])
varlist = ["dwjobsince", "dwhrswkc", "dwunion", "dwlastwrk", "dwhi", "dwnotice"]
for var in varlist:
    df[var] = np.where(df[var] >= 96, np.nan, df[var])
df["dwreas"] = np.where(df["dwreas"] >= 95, np.nan, df["dwreas"])
df["dwnotice"] = np.where(df["dwnotice"] >= 5, np.nan, df["dwnotice"])

##########################################################
# CPI adjustment
##########################################################

# Load CPI data
cpi_file = RAW_DATA_DIR + "/cpi99.txt"
cpi99 = pd.read_csv(cpi_file, sep="\t", header=None, comment="#", usecols=[0, 3])
cpi99.columns = ["year", "cpi99"]

# Merge CPI data with CPS data
df = pd.merge(df, cpi99, on="year")

# Adjust dollar values for inflation
df["dwweekl"] = df["dwweekl"] * df["cpi99"]
df["dwweekc"] = df["dwweekc"] * df["cpi99"]
df["faminc"] = df["faminc"] * df["cpi99"]

##########################################################
# Generate additional variables
##########################################################

# Create new binary variables (preserves missing values)
df["female"] = indicator(df["sex"], 2)
df["black"] = indicator(df["race"], 200)
df["married"] = indicator(df["marst"], [1, 2])
df["col"] = indicator(df["educ"], 110, "greater")
df["pc"] = indicator(df["dwreas"], 1)
df["union"] = indicator(df["dwunion"], 2)
df["hi"] = indicator(df["dwhi"], 2)
df["jf"] = indicator(df["dwjobsince"], 1, "greater")
df["cens"] = 1 - df["jf"]
df["in_metro"] = indicator(df["metro"], [2, 3, 4])
df["emp"] = indicator(df["empstat"], [10, 11, 12])
df["unemp"] = indicator(df["empstat"], [20, 21, 22])
df["nilf"] = indicator(df["empstat"], [30, 36], "range")

# Log earnings
df["lnearnl"] = np.log(df["dwweekl"])
df["lnearnc"] = np.log(df["dwweekc"])

# Displacement year, j2j, and notice
df["dyear"] = df["year"] - df["dwlastwrk"]
df["dyear"] = np.where(df["dwlastwrk"].isna(), np.nan, df["dyear"])
df["j2j"] = np.where((df["dwwksun"] == 0) & (df["dwjobsince"] > 0), 1, 0)
df["j2j"] = np.where(df["dwjobsince"].isnull(), np.nan, df["j2j"])
df["notice"] = indicator(df["dwnotice"], 4)

# Convert race to 1 digit
df["race"] = df["race"].apply(lambda x: int(str(x)[0]))

# Generate education categories
df["educ_cat"] = np.where(df["educ"] <= 72, "Less than HS", np.nan)
df["educ_cat"] = np.where(df["educ"] == 73, "HS Degree", df["educ_cat"])
df["educ_cat"] = np.where(
    (80 <= df["educ"]) & (df["educ"] <= 110), "Some College", df["educ_cat"]
)
df["educ_cat"] = np.where(df["educ"] == 111, "College", df["educ_cat"])
df["educ_cat"] = np.where(df["educ"] > 111, "Graduate Degree", df["educ_cat"])
df["educ_cat"] = np.where(df["educ"] == 999, np.nan, df["educ_cat"])
df["hs"] = indicator(df["educ"], 73, "less")
df["sc"] = indicator(df["educ"], [80, 110], "range")
df["col"] = indicator(df["educ"], 111, "greater")

##########################################################
# Unemployment duration
##########################################################

# Generate observed duration variable
df["obsdur"] = np.where(df["jf"] == 1, df["dwwksun"], df["durunemp"])

# Generate duration 4, 9, and 12 week intervals
df["dur"] = group_dur(df["obsdur"], 12)
df["dur_4week"] = group_dur(df["obsdur"], 4)
df["dur_9week"] = group_dur(df["obsdur"], 9)

##########################################################
# Occupation & Industry Categories
##########################################################

df["occ"] = df["dwocc1990"].apply(get_occ)
df["occ_cat"] = df["occ"].apply(get_broad_occ)
df["ind"] = df["dwind1990"].apply(get_ind)
df["ind_cat"] = df["ind"].apply(get_broad_ind)

##########################################################
# Save data
##########################################################

dws = df[df["dws"] == 1].copy()
dws.to_csv(f"{DATA_DIR}/dws.csv", index=False)
df.to_csv(f"{DATA_DIR}/cps.csv", index=False)

##########################################################
