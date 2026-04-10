from __future__ import annotations
from typing import List, Tuple, Any, Annotated

import importlib

from pathlib import Path

import numpy as np
import xarray as xr

import re

import json

from ..core import IQ, moments, spectra
from ..utils import pathUtils as pu

AvgStrat = Annotated[Tuple[int, int] | None, "None if no averaging, or a Tuple of (K_r, K_az)"]

supportedInputs = ["rk"]

def frxxDataFromFile(filename: str | Path):
	ds = xr.open_dataset(filename)
	if "frxx_data_type" not in ds.attrs:
		raise AttributeError("Netcdf file not created by frxx.")
	
	type = ds.attrs["frxx_data_type"]
	if type == "IQ":
		return IQ(ds)
	elif type == "moments":
		return moments(ds)
	elif type == "spectra":
		return spectra(ds)
	
def constructBeamSpec(angleSpacingDeg: float, beamOverlapDeg):
	 return ("s"+str(angleSpacingDeg)+"o"+str(beamOverlapDeg))
def parseBeamSpec(arg: str):
	match = re.match(r'^s([\d.+-]+)o([\d.+-]+)$', arg.strip())
	if not match:
		return False

	angleSpacingDeg = float(match.group(1))
	beamOverlapDeg = float(match.group(2))

	return angleSpacingDeg, beamOverlapDeg

def constructFourierSpec(
		pulsesIn: int | None, nFFTOut: int | None, 
		avgStrat: AvgStrat = None, 
		bootstrap: int | None = None
	):
	def _parseNum(num: int | None, io: str) -> str:
		if num is None:
			return "0"
		if num < 2:
			raise ValueError(f"Can't take FFT with {io} array less than 2 elements long.")
		return str(num)
	return (
		(f"i{_parseNum(pulsesIn, 'input')}") +
		(f"o{_parseNum(nFFTOut, 'output')}") +
		("aR1A1" if avgStrat is None else f"aR{avgStrat[0]}A{avgStrat[1]}") +
		("" if bootstrap is None else f"b{bootstrap}")
	)
def parseFourierSpec(arg: str):
	s = arg.strip()
	idx = 0
	result: dict[str, Any] = {
		"pulsesIn": None,
		"nFFTOut": None,
		"avgStrat": None,
		"bootstrap": None,
	}
	def readInt():
		nonlocal idx
		start = idx
		while idx < len(s) and s[idx].isdigit():
			idx += 1
		if start == idx:
			return None
		return int(s[start:idx])
	while idx < len(s):
		c = s[idx]
		if c == 'i':
			idx += 1
			val = readInt()
			result["pulsesIn"] = None if val == 0 else val
		elif c == 'o':
			idx += 1
			val = readInt()
			result["nFFTOut"] = None if val == 0 else val
		elif c == 'a':
			idx += 1
			if idx >= len(s) or s[idx] != 'R':
				return False
			idx += 1
			r = readInt()
			if idx >= len(s) or s[idx] != 'A':
				return False
			idx += 1
			a = readInt()
			result["avgStrat"] = None if (r == 1 and a == 1) else (r, a)
		elif c == 'b':
			idx += 1
			result["bootstrap"] = readInt()
		else:
			return False
	return result


def _getPriorityFolder(path: Path, foldersAscendingPriority: List[str], getter):
	folderpaths = [pu.validatePath(path/folder, mustBeDir=True) for folder in foldersAscendingPriority]
	if all(x is None for x in folderpaths):
		return None
	folder = None
	files = []
	for fp in folderpaths:
		if fp is not None:
			prelimFiles = getter(fp)
			if len(prelimFiles) != 0:
				folder = fp
				files = prelimFiles
	if len(files) == 0:
		return None
	
	return folder, files

class FrxxCase:
	def __init__(self, path: str | Path):
		self.path = pu.validatePath(path, mustBeDir=True)
		if self.path is None:
			raise FileNotFoundError("Path is not a valid directory.")
		
		self.iqDir = pu.validatePath(self.path/"iq", mustBeDir=True)
		if self.iqDir is None:
			raise FileNotFoundError("Must have an iq directory!")
		
		sr = _getPriorityFolder(self.iqDir, ["raw_headers", "raw", "qc_headers", "qc"], pu.getIQ)
		if sr is None:
			raise FileNotFoundError("No IQ data folders found.")
		self.iqDir, self.iqFiles = sr
		
		self.outDir = pu.validatePath(self.path/"out", mustBeDir=True)
		if self.outDir is None:
			(self.path/"out").mkdir()
			self.outDir = (self.path/"out")

		self.outFiles: dict[str, Any] = {x.name: {} for x in self.outDir.iterdir() if x.is_dir() and parseBeamSpec(x.name)}

		#it's okay if nothing is in out, but if there is something it needs to follow constraints.
		for beamSpec, contents in self.outFiles.items():
			delayedMoment = [x.name for x in pu.globPath(self.outDir/beamSpec/"moments_delayed_*") if x.is_dir()][::-1]

			m = _getPriorityFolder(self.outDir/beamSpec, [*delayedMoment, "moments"], pu.getMoments)
			s = pu.validatePath(self.outDir/beamSpec/"spectra", mustBeDir=True)

			if s is not None:
				fSpecs = {x.name: {} for x in s.iterdir() if x.is_dir() and parseFourierSpec(x.name)}	
				if len(fSpecs) == 0:
					raise FileNotFoundError("Spectra folder is empty.")
				parsed = [parseFourierSpec(x) for x in fSpecs.keys()]
				seen = []
				for p in parsed:
					if p in seen:
						raise ValueError("Duplicate fourier spec found.")
					seen.append(p)
				for fSpec, specContents in fSpecs.items():
					delayedSpectra = [x.name for x in pu.globPath(self.outDir/beamSpec/"spectra"/fSpec/"delayed_*") if x.is_dir()][::-1]
					sFSpec = _getPriorityFolder(self.outDir/beamSpec/"spectra"/fSpec, [*delayedSpectra, "full"], pu.getSpectra)
					if sFSpec is not None:
						_, l = sFSpec
						if len(l) != len(self.iqFiles):
							raise ValueError("Number of moment and iq files in folder do not match.")
						specContents["files"] = l
				contents["spectra"] = fSpecs

			if m is not None:
				_, l = m
				if len(l) != len(self.iqFiles):
					raise ValueError("Number of moment and iq files do not match.")
				contents["moment_files"] = l

			if s is None and m is None:
				raise FileNotFoundError("Either Spectra or Moments must exist.")

	def toJson(self) -> str:
		return ""



class _Computation:
	# def parseSpec(self, arg: Any, curFilename: Path | None = None):
	# 	if not isinstance(arg, str):
	# 		return arg

	# 	match = re.match(r'^(iq|m|s)\[(.*?)\]->(.+)$', arg.strip())
	# 	if not match:
	# 		return arg

	# 	dType, indexStr, varName = match.groups()
	# 	if dType == "iq":
	# 		dType = "IQ"
	# 	elif dType == "m":
	# 		dType = "moments"
	# 	elif dType == "s":
	# 		dType = "spectra"
	# 	indexStr = indexStr.strip()
	# 	varName = varName.strip()

	# 	if indexStr == '':
	# 		path = getPathFromIndex(0, dType, curIdx=curIdx)

	# 	elif indexStr.startswith('r'):
	# 		relIdx = int(indexStr[1:])
	# 		path = getPathFromIndex(relIdx, dType, curIdx=curIdx)

	# 	elif re.match(r'^-?\d+$', indexStr):
	# 		absIdx = int(indexStr)
	# 		path = getPathFromIndex(absIdx, dType, curIdx=None)

	# 	else:
	# 		path = Path(indexStr).resolve()
	# 		if not path.exists():
	# 			raise FileNotFoundError(f"Specified file does not exist: {path}")

	# 	return getVar(path, varName)


	def __init__(self, computationJson: str, case: FrxxCase | None = None):
		computationDict = json.loads(computationJson)
		self.retVars: List[str] = computationDict["return_variables"]
		path, fn = computationDict["function"].rsplit(".", 1)
		self.function = getattr(importlib.import_module(path), fn)
		self.kwargs: dict = computationDict["kwargs"]
			
	def toJson(self) -> str:
			return ""

		

class DelayedCompute:
	def __init__(self, delayedComputeJson: str | None = None, case: FrxxCase | None = None):
		self.vars: List[str] = []
		self.computations: List[_Computation] = []
			
	def toJson(self) -> str:
			return ""