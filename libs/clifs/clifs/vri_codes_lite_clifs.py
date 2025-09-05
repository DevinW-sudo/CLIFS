# lite re-implementation of the violence risk index. Removed categories unnecessary for
# our fusion analysis (also removed harmful categories; e.g. ones including racial slurs)
# implemented in Python instead of R
# based on the work of Julia Ebner, Christopher Kavanagh, and Harvey Whitehouse (2023)
# "Measuring socio-psychological drivers of extreme violence in online terrorist manifestos: an alternative linguistic risk assessment model"

import re

#FUSION
fusion = ["Brother","sister","family","motherland","our blood","fatherland","sons",
          "daughters","kin","my people","my race","our people","European race",
          "ancestry","ancestor","descendant","fellow","brethren","comrades"]

# IDENTIFICATION
identification1 = [r"\bwe\b", r"\bus\b", r"\bour\b", r"\bthey\b", r"\bthem\b", r"\btheir\b"]
identification2 = [r"\bI\b", r"\bme\b", r"\bmy\b", r"\byou\b", r"\byour\b"]

#CATEGORIES LIST
categorieslist = ["fusion", "identification%"]
