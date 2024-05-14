import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# =========================== SUBSAMPLED DATA ================================
# ============================================================================

subsample = np.loadtxt('Frogs-subsample.csv', delimiter=',', dtype=str)
subsample = subsample[1:]
mfcc10 = subsample[:, 0].astype(float)
mfcc17 = subsample[:, 1].astype(float)
species = subsample[:, 2]
colors = {'HylaMinuta': 'red', 'HypsiboasCinerascens': 'blue'}


# ===== SCATTER PLOT FOR RAW FEATURES ======
for specie, color in colors.items():
    plt.scatter(mfcc10[specie == species], mfcc17[specie == species], c=color, label=specie)
plt.title('Scatter Plot - MFCCs_10 vs. MFCCs_17 (for subsampled data)')
plt.xlabel('MFCCs_10')
plt.ylabel('MFCCs_17')
plt.legend()
plt.show()


# ===== HISTOGRAM - HylaMinuta, MFCCs_10 ======
for specie, color in colors.items():
    plt.hist(mfcc10[species == 'HylaMinuta'], bins=12, alpha=0.5, color=colors['HylaMinuta'], label=specie)
plt.title('Histogram - MFCCs_10 (for HylaMinuta)')
plt.xlabel('MFCCs_10')
plt.ylabel('Frequency')
plt.show()

# ===== HISTOGRAM - HylaMinuta, MFCCs_17 ======
for specie, color in colors.items():
    plt.hist(mfcc17[species == 'HylaMinuta'], bins=12, alpha=0.5, color=colors['HylaMinuta'], label=specie)
plt.title('Histogram - MFCCs_17 (for HylaMinuta)')
plt.xlabel('MFCCs_17')
plt.ylabel('Frequency')
plt.show()

# ===== HISTOGRAM - HypsiboasCinerascens, MFCCs_10 ======
for specie, color in colors.items():
    plt.hist(mfcc10[species == 'HypsiboasCinerascens'], bins=12, alpha=0.5, color=colors['HypsiboasCinerascens'], label=specie)
plt.title('Histogram - MFCCs_10 (for HypsiboasCinerascens)')
plt.xlabel('MFCCs_10')
plt.ylabel('Frequency')
plt.show()

# ===== HISTOGRAM - HypsiboasCinerascens, MFCCs_17 ======
for specie, color in colors.items():
    plt.hist(mfcc17[species == 'HypsiboasCinerascens'], bins=12, alpha=0.5, color=colors['HypsiboasCinerascens'], label=specie)
plt.title('Histogram - MFCCs_17 (for HypsiboasCinerascens)')
plt.xlabel('MFCCs_17')
plt.ylabel('Frequency')
plt.show()


# ===== LINE GRAPH - HylaMinuta, MFCCs_10 ======
plt.plot(np.sort(mfcc10[species == 'HylaMinuta']), label='MFCCs_10', color=colors['HylaMinuta'])
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Line Graph - MFCCs_10 (for HylaMinuta)')
plt.legend()
plt.show()

# ===== LINE GRAPH - HylaMinuta, MFCCs_17 ======
plt.plot(np.sort(mfcc17[species == 'HylaMinuta']), label='MFCCs_17', color=colors['HylaMinuta'])
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Line Graph - MFCCs_17 (for HylaMinuta)')
plt.legend()
plt.show()

# ===== LINE GRAPH - HypsiboasCinerascens, MFCCs_10 ======
plt.plot(np.sort(mfcc10[species == 'HypsiboasCinerascens']), label='MFCCs_10', color=colors['HypsiboasCinerascens'])
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Line Graph - MFCCs_10 (for HypsiboasCinerascens)')
plt.legend()
plt.show()

# ===== LINE GRAPH - HypsiboasCinerascens, MFCCs_17 ======
plt.plot(np.sort(mfcc17[species == 'HypsiboasCinerascens']), label='MFCCs_17', color=colors['HypsiboasCinerascens'])
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Line Graph - MFCCs_17 (for HypsiboasCinerascens)')
plt.legend()
plt.show()


# ===== BOX PLOT - MFCCs_10s ======
data = [mfcc10[species == 'HylaMinuta'], mfcc10[species == 'HypsiboasCinerascens']]
plt.boxplot(data, labels=['HylaMinuta', 'HypsiboasCinerascens'])
plt.xlabel('Species')
plt.ylabel('MFCCs_10')
plt.title('Boxplot - MFCCs_10s (for subsampled data)')
plt.show()

# ===== BOX PLOT - MFCCs_17s ======
data = [mfcc17[species == 'HylaMinuta'], mfcc17[species == 'HypsiboasCinerascens']]
plt.boxplot(data, labels=['HylaMinuta', 'HypsiboasCinerascens'])
plt.xlabel('Species')
plt.ylabel('MFCCs_17')
plt.title('Boxplot - MFCCs_17s (for subsampled data)')
plt.show()


# ===== BAR GRAPH WITH ERRORS ======
meanHyla10 = np.mean(mfcc10[species == 'HylaMinuta'])
meanHyla17 = np.mean(mfcc17[species == 'HylaMinuta'])
stdHyla10 = np.std(mfcc10[species == 'HylaMinuta'])
stdHyla17 = np.std(mfcc17[species == 'HylaMinuta'])
meanHypsi10 = np.mean(mfcc10[species == 'HypsiboasCinerascens'])
meanHypsi17 = np.mean(mfcc17[species == 'HypsiboasCinerascens'])
stdsHypsi10 = np.std(mfcc10[species == 'HypsiboasCinerascens'])
stdsHypsi17 = np.std(mfcc17[species == 'HypsiboasCinerascens'])
x = np.arange(2)
fig, axes = plt.subplots()
bar1 = axes.bar(x - 0.2, [meanHyla10, meanHyla17], 0.4, label='HylaMinuta', yerr=[stdHyla10, stdHyla17])
bar2 = axes.bar(x + 0.2, [meanHypsi10, meanHypsi17], 0.4, label='HypsiboasCinerascens', yerr=[stdsHypsi10, stdsHypsi17])
axes.set_xlabel('Features')
axes.set_ylabel('Value')
axes.set_title('Bar Graph with Error Bars (for subsampled data)')
axes.set_xticks(x)
axes.set_xticklabels(['MFCCs_10', 'MFCCs_17'])
axes.legend()
plt.show()

# ============================================================================
# =============================== FULL DATA ==================================
# ============================================================================

fullData = np.loadtxt('Frogs.csv', delimiter=',', dtype=str)
fullData = fullData[1:]
mfcc10 = fullData[:, 0].astype(float)
mfcc17 = fullData[:, 1].astype(float)
species = fullData[:, 2]
colors = {'HylaMinuta': 'red', 'HypsiboasCinerascens': 'blue'}

# ===== SCATTER PLOT FOR RAW FEATURES ======
for specie, color in colors.items():
    plt.scatter(mfcc10[specie == species], mfcc17[specie == species], c=color, label=specie)
plt.title('Scatter Plot of MFCCs_10 vs. MFCCs_17 (for full sample data)' )
plt.xlabel('MFCCs_10')
plt.ylabel('MFCCs_17')
plt.legend()
plt.show()


# ===== HISTOGRAM - HylaMinuta, MFCCs_10 ======
for specie, color in colors.items():
    plt.hist(mfcc10[species == 'HylaMinuta'], bins=12, alpha=0.5, color=colors['HylaMinuta'], label=specie)
plt.title('Histogram - MFCCs_10 (for HylaMinuta)')
plt.xlabel('MFCCs_10')
plt.ylabel('Frequency')
plt.show()

# ===== HISTOGRAM - HylaMinuta, MFCCs_17 ======
for specie, color in colors.items():
    plt.hist(mfcc17[species == 'HylaMinuta'], bins=12, alpha=0.5, color=colors['HylaMinuta'], label=specie)
plt.title('Histogram - MFCCs_17 (for HylaMinuta)')
plt.xlabel('MFCCs_17')
plt.ylabel('Frequency')
plt.show()

# ===== HISTOGRAM - HypsiboasCinerascens, MFCCs_10 ======
for specie, color in colors.items():
    plt.hist(mfcc10[species == 'HypsiboasCinerascens'], bins=12, alpha=0.5, color=colors['HypsiboasCinerascens'], label=specie)
plt.title('Histogram - MFCCs_10 (for HypsiboasCinerascens)')
plt.xlabel('MFCCs_10')
plt.ylabel('Frequency')
plt.show()

# ===== HISTOGRAM - HypsiboasCinerascens, MFCCs_17 ======
for specie, color in colors.items():
    plt.hist(mfcc17[species == 'HypsiboasCinerascens'], bins=12, alpha=0.5, color=colors['HypsiboasCinerascens'], label=specie)
plt.title('Histogram - MFCCs_17 (for HypsiboasCinerascens)')
plt.xlabel('MFCCs_17')
plt.ylabel('Frequency')
plt.show()


# ===== LINE GRAPH - HylaMinuta, MFCCs_10 ======
plt.plot(np.sort(mfcc10[species == 'HylaMinuta']), label='MFCCs_10', color=colors['HylaMinuta'])
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Line Graph - MFCCs_10 (for HylaMinuta)')
plt.legend()
plt.show()

# ===== LINE GRAPH - HylaMinuta, MFCCs_17 ======
plt.plot(np.sort(mfcc17[species == 'HylaMinuta']), label='MFCCs_17', color=colors['HylaMinuta'])
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Line Graph - MFCCs_17 (for HylaMinuta)')
plt.legend()
plt.show()

# ===== LINE GRAPH - HypsiboasCinerascens, MFCCs_10 ======
plt.plot(np.sort(mfcc10[species == 'HypsiboasCinerascens']), label='MFCCs_10', color=colors['HypsiboasCinerascens'])
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Line Graph - MFCCs_10 (for HypsiboasCinerascens)')
plt.legend()
plt.show()

# ===== LINE GRAPH - HypsiboasCinerascens, MFCCs_17 ======
plt.plot(np.sort(mfcc17[species == 'HypsiboasCinerascens']), label='MFCCs_17', color=colors['HypsiboasCinerascens'])
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Line Graph - MFCCs_17 (for HypsiboasCinerascens)')
plt.legend()
plt.show()


# ===== BOX PLOT - MFCCs_10s ======
data = [mfcc10[species == 'HylaMinuta'], mfcc10[species == 'HypsiboasCinerascens']]
plt.boxplot(data, labels=['HylaMinuta', 'HypsiboasCinerascens'])
plt.xlabel('Species')
plt.ylabel('MFCCs_10')
plt.title('Boxplot - MFCCs_10s (for full data)')
plt.show()

# ===== BOX PLOT - MFCCs_17s ======
data = [mfcc17[species == 'HylaMinuta'], mfcc17[species == 'HypsiboasCinerascens']]
plt.boxplot(data, labels=['HylaMinuta', 'HypsiboasCinerascens'])
plt.xlabel('Species')
plt.ylabel('MFCCs_17')
plt.title('Boxplot - MFCCs_17s (for full data)')
plt.show()


# ===== BAR GRAPH WITH ERRORS ======
meanHyla10 = np.mean(mfcc10[species == 'HylaMinuta'])
meanHyla17 = np.mean(mfcc17[species == 'HylaMinuta'])
stdHyla10 = np.std(mfcc10[species == 'HylaMinuta'])
stdHyla17 = np.std(mfcc17[species == 'HylaMinuta'])
meanHypsi10 = np.mean(mfcc10[species == 'HypsiboasCinerascens'])
meanHypsi17 = np.mean(mfcc17[species == 'HypsiboasCinerascens'])
stdsHypsi10 = np.std(mfcc10[species == 'HypsiboasCinerascens'])
stdsHypsi17 = np.std(mfcc17[species == 'HypsiboasCinerascens'])
x = np.arange(2)
fig, axes = plt.subplots()
bar1 = axes.bar(x - 0.2, [meanHyla10, meanHyla17], 0.4, label='HylaMinuta', yerr=[stdHyla10, stdHyla17])
bar2 = axes.bar(x + 0.2, [meanHypsi10, meanHypsi17], 0.4, label='HypsiboasCinerascens', yerr=[stdsHypsi10, stdsHypsi17])
axes.set_xlabel('Features')
axes.set_ylabel('Value')
axes.set_title('Bar Graph with Error Bars (for full data)')
axes.set_xticks(x)
axes.set_xticklabels(['MFCCs_10', 'MFCCs_17'])
axes.legend()
plt.show()
