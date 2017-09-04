#!/bin/python

import numpy as np
import os
import inspect
from scipy.misc import imresize

# ================================================================
# Constants
# ================================================================

# ================================================================
# Functions
# ================================================================

# ------------------------------- Debug output

def printVar(name, val):
	print(name + "=")
	print(val)

# ------------------------------- Inspection

def getNumArgs(func):
	(args, varargs, varkw, defaults) = inspect.getargspec(func)
	return len(args)

# ------------------------------- Array attribute checks

def isScalar(x):
	return not hasattr(x, "__len__")


def isSingleton(x):
	return (not isScalar(x)) and (len(x) == 1)


def is1D(x):
	return len(x.shape) == 1


def is2D(x):
	return len(x.shape) == 2


def nrows(A):
	return A.shape[0]


def ncols(A):
	return A.shape[1]


# ------------------------------- Array manipulations

def asColumnVect(V):
	return V.reshape((V.size,1))


def prependOnesCol(A):
	N, P = A.shape
	return np.hstack((np.ones((N,1)), A))


def prependOnesRow(A):
	N, P = A.shape
	return np.vstack((np.ones((1,P)), A))


def nonzeroCols(A):
	"""Return the columns of A that contain nonzero elements"""
	if np.any(np.isnan(A)):
		print("WARNING: nonzeroCols: cols containing NaNs may be removed")
	if is2D(A):
		return np.where(np.sum(np.abs(A), 0))[0]  # [0] to unpack singleton
	return np.where(np.abs(A))[0]


def removeCols(A, cols):
	if cols is not None:
		return np.delete(A, cols, 1)
	return A


def removeZeroCols(A):
	return A[:,nonzeroCols(A)]


def extractCols(A, cols):
	extracted = A[:, cols]
	remnants = removeCols(A, cols)
	return extracted, remnants


def meanNormalizeRows(A):
	rowMeans = np.mean(A, 1).reshape(A.shape[0], 1)
	return A - rowMeans


def meanNormalizeCols(A):
	return A - np.mean(A, 0)


def stdNormalizeCols(A):
	A = removeZeroCols(A)
	colStds = np.std(A,0)
	return A / colStds


def zNormalizeCols(A):
	return stdNormalizeCols(meanNormalizeCols(A))


def normalizeCols(A):
	return A / np.linalg.norm(A, axis=0)


def downsampleMat(A, rowsBy=1, colsBy=1):
	newShape = A.shape / np.array([rowsBy, colsBy], dtype=np.float)
	newShape = newShape.astype(np.int)  # round to int
	return imresize(A, newShape)


def zeroOneScaleMat(A):
	minVal = np.min(A)
	maxVal = np.max(A)
	return (A - minVal) / (maxVal - minVal)


def array2tuple(V):
	return tuple(map(tuple,V))[0]


def dictsTo2DArray(dicts):
	"""returns a dense 2D array with a column for each key in any dictionary
	and a row for each dictionary; where a key is absent in a dictionary, the
	corresponding entry is populated by a 0. Note that the columns are
	ordered according to the sorting of the keys

	EDIT: also returns the column headers as a tuple
	"""
	allKeys = set()
	for d in dicts:
		allKeys.update(d.keys())
	sortedKeys = sorted(allKeys)
	numKeys = len(sortedKeys)
	numDicts = len(dicts)
	keyIdxs = np.arange(numKeys)
	key2Idx = dict(zip(sortedKeys, keyIdxs))
	ar = np.zeros((numDicts, numKeys))
	for i, d in enumerate(dicts):
		for key, val in d.iteritems():
			idx = key2Idx[key]
			ar[i, idx] = val
	return ar, tuple(sortedKeys)

# ------------------------------- Array searching

def findRow(A, q):
	"""return the row indices of all rows in A that match the vector q"""
	assert(ncols(A) == len(q))
	assert(is1D(q))
	rowEqualsQ = np.all(A == q, axis=1)
	return np.where(rowEqualsQ)


def numNonZeroElements(A):
	return len(np.where(A.flatten()))

# ------------------------------- Filesystem

def ls():
	return os.listdir('.')

def isHidden(path):
	filename = os.path.basename(path)
	return filename.startswith('.')

def isVisible(path):
	return not isHidden(path)

def joinPaths(dir, contents):
	return map(lambda f: os.path.join(dir, f), contents)

def filesInDirMatching(dir, prefix=None, suffix=None, absPaths=False,
						onlyFiles=False, onlyDirs=False):
	files = os.listdir(dir)
	if prefix:
		files = filter(lambda f: f.startswith(prefix), files)
	if suffix:
		files = filter(lambda f: f.endswith(suffix), files)
	if onlyFiles or onlyDirs:
		paths = joinPaths(dir, files)
		if onlyFiles:
			newFiles = []
			for f, path in zip(files, paths):
				if os.path.isfile(path):
					newFiles.append(f)
			files = newFiles
		if onlyDirs:
			newFiles = []
			for f, path in zip(files, paths):
				if os.path.isdir(path):
					newFiles.append(f)
			files = newFiles
	if absPaths:
		files = joinPaths(dir, files)
	return files

def listSubdirs(dir, startswith=None, endswith=None, absPaths=False):
	return filesInDirMatching(dir, startswith, endswith, absPaths,
		onlyDirs=True)

def listFilesInDir(dir, startswith=None, endswith=None, absPaths=False):
	return filesInDirMatching(dir, startswith, endswith, absPaths,
		onlyFiles=True)

def listHiddenFilesInDir(dir, startswith=None, endswith=None, absPaths=False):
	contents = filesInDirMatching(dir, startswith, endswith, absPaths,
		onlyFiles=True)
	return filter(isHidden, contents)

def listVisibleFilesInDir(dir, startswith=None, endswith=None, absPaths=False):
	contents = filesInDirMatching(dir, startswith, endswith, absPaths,
		onlyFiles=True)
	return filter(isVisible, contents)

def ensureDirExists(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)

def basename(f, noexts=False):
	name = os.path.basename(f)
	if noexts:
		name = name.split('.')[0]
	return name

# ------------------------------- Numerical Funcs

def rnd(A, dec=6):
	return np.round(A, decimals=dec)

# ------------------------------- Testing

def main():
	pass

if __name__ == '__main__':
	main()
