# Some parameters
DOWNLOAD_RAW = False
EXTRACT_IPUMS = False
SIM_SM_AGAIN = False
RUN_EXT_AGAIN = False
SIMBIN_AGAIN = False

# Download link for the raw data
DATA_LINK = "https://www.dropbox.com/scl/fi/l45oehwvjucfz9xpafdj1/raw.zip?rlkey=g4zdee32tgwcox4boc9950zxo&st=5ojpr02p&dl=1"

# Specify directories
DATA_DIR = "data"
RAW_DATA_DIR = DATA_DIR + "/raw"
IPUMS_DIR = RAW_DATA_DIR + "/ipums-extract"
OUTPUT_DIR = "output"
QUANTS_DIR = "output/quants"
TEST_QUANTS_DIR = "output/quants"
DIRS = [DATA_DIR, OUTPUT_DIR, QUANTS_DIR, TEST_QUANTS_DIR]

# Specify font for plots
FONT = {"family": "serif", "font": "PT Serif", "size": 9.5}

# Critical value for confidence intervals
CRIT = 1.645  # 90% confidence interval


# Colors
class Colors:
    RED = (0.7, 0.1, 0.2)
    BLACK = (0, 0, 0)
    GREY = (0.5, 0.5, 0.5)
    DGREY = (0.3, 0.3, 0.3)
    BLUE = (0.2, 0.2, 0.65)
    GREEN = (0.2, 0.5, 0.3)
    WHITE = (1, 1, 1)


# Variables to keep
cpsvars = [
    "serial",
    "cpsid",
    "mish",
    "pernum",
    "hwtfinl",
    "wtfinl",
    "year",
    "month",
    "statefip",
    "metro",
    "metarea",
    "county",
    "faminc",
    "sex",
    "marst",
    "empstat",
    "labforce",
    "occ1990",
    "ind1990",
    "classwkr",
    "numjob",
    "educ",
    "race",
    "durunem2",
    "age",
    "uhrsworkt",
    "uhrswork1",
    "durunemp",
]
dwsvars = [
    "dwstat",
    "dwresp",
    "dwrecall",
    "dwfulltime",
    "dwclass",
    "dwsuppwt",
    "dws",
    "dwreas",
    "dwnotice",
    "dwlastwrk",
    "dwunion",
    "dwben",
    "dwexben",
    "dwhi",
    "dwind1990",
    "dwocc1990",
    "dwmove",
    "dwhinow",
    "dwyears",
    "dwweekl",
    "dwweekc",
    "dwjobsince",
    "dwhrswkc",
    "dwwksun",
]
SELECTED_VARS = cpsvars + dwsvars


# Rando functions
def print_bold(text):
    print(f"\033[1m{text}\033[0m")
