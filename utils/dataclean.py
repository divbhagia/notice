import numpy as np

##########################################################
# Function to group duration
##########################################################


def group_dur(dur_var, interval):
    dur = 0
    for i in range(0, 53, interval):
        dur = np.where(
            (dur_var >= i) & (dur_var < i + interval), i + 0.5 * interval, dur
        )
    dur = np.where(dur_var > i, i + 0.5 * interval, dur)
    return dur


##########################################################
# Create new binary variables in pandas frames
##########################################################


# Create new binary variables in pandas frames
def indicator(var, values, opt="values", rm_na=True):
    """
    This function creates a binary indicator variable based on
    the values of a given variable.

    Parameters:
    var (pandas.Series): A pandas Series object.
    values (int or list): The values to use for the indicator.
    opt (str): The type of indicator to create.

    Options:
    - 'values': Indicator if variable is equal to specific values.
    - 'range': Indicator if variable is within a specific interval.
    - 'greater': Indicator if variable is greater than a specific value.
    - 'less': Indicator if variable is less than a specific value.
    Default is 'values'.

    Returns:
    newvar (pandas.Series): A binary indicator variable.
    """

    values = [values] if type(values) == int else values
    if opt == "values":
        newvar = var.isin(values)
    elif opt == "range":
        newvar = (var >= values[0]) & (var <= values[1])
    elif opt == "greater":
        newvar = var >= values[0]
    elif opt == "less":
        newvar = var <= values[0]
    else:
        raise ValueError(
            "Option not recognized. Please use: values, interval, greater, or less."
        )

    if rm_na:
        newvar = np.where(var.isnull(), np.nan, newvar)

    return newvar


################################################################
# Occupation Categories
################################################################

occ_map = {
    (4, 22): 1,  # Executive, Administrative, & Managerial
    (23, 37): 2,  # Management Related
    (43, 199): 3,  # Professional Specialty
    (203, 235): 4,  # Technicians and Related Support
    (243, 283): 5,  # Sales
    (303, 389): 6,  # Administrative Support
    (405, 408): 7,  # Housekeeping and Cleaning
    (415, 427): 8,  # Protective Service
    (433, 472): 9,  # Other Service
    (473, 475): 10,  # Farm Operators and Managers
    (479, 498): 11,  # Other Agricultural and Related
    (503, 549): 12,  # Mechanics and Repairers
    (558, 599): 13,  # Construction
    (614, 617): 14,  # Extractive Occupations
    (628, 699): 15,  # Precision Production
    (703, 799): 16,  # Machine Operators, Assemblers, & Inspectors
    (803, 899): 17,  # Transportation and Material Moving
}


def get_occ(var):
    for (start, end), category in occ_map.items():
        if start <= var <= end:
            return category
    return None


################################################################
# Broad Occupation Categories Labelled
################################################################


def get_broad_occ(var):
    if 1 <= var <= 3:
        return "Executive, Managerial, Professional"
    elif 4 <= var <= 6:
        return "Tech Support, Sales, Admin"
    elif 7 <= var <= 9:
        return "Services"
    elif var in [10, 11, 12, 14]:
        return "Other"
    elif var == 13:
        return "Construction"
    elif 15 <= var <= 16:
        return "Production"
    elif var == 17:
        return "Transportation"
    else:
        return "Unknown"


################################################################
# Industry Categories
################################################################

ind_map = {
    (60, 60): 1,  # Construction
    (10, 32): 2,  # Agriculture, Forestry, & Fishing
    (40, 50): 3,  # Mining
    (100, 392): 4,  # Manufacturing
    (400, 472): 5,  # Transport & Communication
    (500, 571): 6,  # Wholesale Trade
    (580, 691): 7,  # Retail Trade
    (700, 712): 8,  # FIRE
    (721, 760): 9,  # Business & Repair Services
    (761, 791): 10,  # Personal Services
    (800, 810): 11,  # Entertainment & Recreation
    (812, 893): 12,  # Professional & Related Services
    (900, 932): 13,  # Public Admin
    (940, 960): 14,  # Military
}


def get_ind(var):
    for (start, end), category in ind_map.items():
        if start <= var <= end:
            return category
    return None


################################################################
# Broad Industry Categories Labelled
################################################################


def get_broad_ind(var):
    if var == 1:
        return "Construction"
    elif var == 4:
        return "Manufacturing"
    elif var == 5:
        return "Transport & Communication"
    elif 6 <= var <= 7:
        return "Wholesale & Retail Trade"
    elif var == 8:
        return "Finance, Insurance, Real Estate"
    elif var == 9:
        return "Business & Repair Services"
    elif var == 12:
        return "Professional Services"
    elif var in [2, 3, 10, 11, 13, 14]:
        return "Other"
    else:
        return "Unknown"


################################################################
