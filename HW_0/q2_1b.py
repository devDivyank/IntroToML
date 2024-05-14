import numpy as np

# ============================================================================
# =========================== SUBSAMPLED DATA ================================
# ============================================================================

subsample = np.loadtxt('Frogs-subsample.csv', delimiter=',', dtype=str)
subsample = subsample[1:]
mfcc10 = subsample[:, 0].astype(float)
mfcc17 = subsample[:, 1].astype(float)
species = subsample[:, 2]


mean_mfcc10 = np.mean(mfcc10)
mean_mfcc17 = np.mean(mfcc17)
covarianceMatrix = np.cov(subsample[:, 0:2].astype(float), rowvar=False)
std_mfcc10 = np.std(mfcc10)
std_mfcc17 = np.std(mfcc17)

print("FOR SUBSAMPLED DATA:")
print("Mean ->")
print("mfcc10: " + str(mean_mfcc10))
print("mfcc17: " + str(mean_mfcc17) + '\n')
print("Standard deviation -> ")
print("mfcc10: " + str(std_mfcc10))
print("mfcc17: " + str(std_mfcc17) + '\n')
print("Covariance Matrix ->")
print(covarianceMatrix)
print()

# ============================================================================
# =============================== FULL DATA ==================================
# ============================================================================

fullData = np.loadtxt('Frogs.csv', delimiter=',', dtype=str)
fullData = fullData[1:]
mfcc10 = fullData[:, 0].astype(float)
mfcc17 = fullData[:, 1].astype(float)
species = fullData[:, 2]

mean_mfcc10 = np.mean(mfcc10)
mean_mfcc17 = np.mean(mfcc17)
covarianceMatrix = np.cov(fullData[:, 0:2].astype(float), rowvar=False)
std_mfcc10 = np.std(mfcc10)
std_mfcc17 = np.std(mfcc17)

print("FOR FULL DATA:")
print("Mean ->")
print("mfcc10: " + str(mean_mfcc10))
print("mfcc17: " + str(mean_mfcc17) +'\n')
print("Standard deviation -> ")
print("mfcc10: " + str(std_mfcc10))
print("mfcc17: " + str(std_mfcc17) + '\n')
print("Covariance Matrix ->")
print(covarianceMatrix)