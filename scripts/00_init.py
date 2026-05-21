# Import libraries
import os
import requests
from zipfile import ZipFile

# Import parameters
from utils.config import DATA_LINK, DATA_DIR, DIRS, IPUMS_DIR
from utils.config import DOWNLOAD_RAW, EXTRACT_IPUMS

# Create directories if they do not exist
for d in DIRS:
    if not os.path.exists(d):
        os.makedirs(d)

# Download raw data
if DOWNLOAD_RAW:
    zipfile = DATA_DIR + "/raw.zip"
    r = requests.get(DATA_LINK)
    with open(zipfile, "wb") as f:
        f.write(r.content)
    with ZipFile(zipfile, "r") as z:
        for file in z.namelist():
            if file.startswith("raw/"):
                z.extract(file, DATA_DIR)
    os.remove(zipfile)

# Extract IPUMS data (takes ~10 mins, optional as extracted data is included)
if EXTRACT_IPUMS:
    from ipumspy import readers  # type: ignore

    ddi = readers.read_ipums_ddi(f"{IPUMS_DIR}/cps_00039.xml")
    df = readers.read_microdata(ddi, f"{IPUMS_DIR}/cps_00039.dat")
    df.columns = df.columns.str.lower()
    df.to_csv(f"{IPUMS_DIR}/cps_raw.csv")
