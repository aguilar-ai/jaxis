from typing import NewType

Quantile = NewType("Quantile", float)
MaxNormalizedViolation = NewType("MaxNormalizedViolation", float)
FractionOver1 = NewType("FractionOver1", float)

MeanAbsErr = NewType("MeanAbsErr", float)
RootMeanSquareError = NewType("RootMeanSquareError", float)
MaxAbsDelta = NewType("MaxAbsDelta", float)

RelL1 = NewType("RelL1", float)
RelL2 = NewType("RelL2", float)
RelLInf = NewType("RelLInf", float)

CosineDistance = NewType("CosineDistance", float)

KLDivergence = NewType("KLDivergence", float)
JSDivergence = NewType("JSDivergence", float)

KendallTauDistance = NewType("KendallTauDistance", float)
TopKOverlap = NewType("TopKOverlap", float)

NumNonFinite = NewType("NumNonFinite", int)
NumNaN = NewType("NumNaN", int)
NumInf = NewType("NumInf", int)

OffTargetRatio = NewType("OffTargetRatio", float)
LeakageEnergyRatio = NewType("LeakageEnergyRatio", float)
OffTargetMaxAbs = NewType("OffTargetMaxAbs", float)
OnTargetMag = NewType("OnTargetMag", float)
