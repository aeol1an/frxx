from __future__ import annotations
from typing import List, Tuple, Callable, Any, cast

import importlib

import os
from pathlib import Path

import numpy as np
import xarray as xr

import re
import json

from ..core import IQ, moments, spectra
from ..utils import pathUtils as pu
from ..utils import stringUtils as su

supportedInputs = ["rk"]

def frxxDataFromFile(filename: str | Path) -> IQ | moments | spectra:
	ds = xr.open_dataset(filename, decode_times=False, decode_timedelta=False, chunks={})
	if "frxx_data_type" not in ds.attrs:
		raise AttributeError("Netcdf file not created by frxx.")
	
	type = ds.attrs["frxx_data_type"]
	if type == "IQ":
		return IQ(ds)
	elif type == "moments":
		return moments(ds)
	elif type == "spectra":
		return spectra(ds)
	
	raise ValueError("Unknown frxx_data_type")


def _getPriorityFolder(path: Path, foldersAscendingPriority: List[str], getter) -> Tuple[Path, List[Path]] | None:
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
	
	return folder, files #type: ignore

def _iqFolderNames():
	return ["unqc_iq_headers", "unqc_iq", "iq_headers", "iq"]
def _mFolderNames(caseRoot: Path, beamSpec: str):
	return [x.name for x in pu.globPath(caseRoot/"out"/beamSpec/"moments_delayed_*") if x.is_dir()][::-1] + ["moments"]
def _sFolderNames(caseRoot: Path, beamSpec: str, fourierSpec: str):
	return [x.name for x in pu.globPath(caseRoot/"out"/beamSpec/"spectra"/fourierSpec/"spectra_delayed_*") if x.is_dir()][::-1] + ["spectra"]

class FrxxCase:
	def __init__(self, path: str | Path):
		self.path: Path = pu.validatePath(path, mustBeDir=True) #type: ignore
		if self.path is None:
			raise FileNotFoundError("Path is not a valid directory.")
		
		self.iqDir: Path = pu.validatePath(self.path/"iq", mustBeDir=True) #type: ignore
		if self.iqDir is None:
			raise FileNotFoundError("Must have an iq directory!")
		
		sr = _getPriorityFolder(self.iqDir, _iqFolderNames(), pu.getIQ)
		if sr is None:
			raise FileNotFoundError("No IQ data folders found.")
		self.iqDir, self.iqFiles = sr
		
		self.outDir: Path = pu.validatePath(self.path/"out", mustBeDir=True) #type: ignore
		if self.outDir is None:
			(self.path/"out").mkdir()
			self.outDir = (self.path/"out")

		self.outFiles: dict[str, Any] = {x.name: {} for x in self.outDir.iterdir() if x.is_dir() and su.parseBeamSpec(x.name)}

		#it's okay if nothing is in out, but if there is something it needs to follow constraints.
		for beamSpec, contents in self.outFiles.items():
			m = _getPriorityFolder(self.outDir/beamSpec, _mFolderNames(self.path, beamSpec), pu.getMoments)
			s = pu.validatePath(self.outDir/beamSpec/"spectra", mustBeDir=True)

			if s is not None:
				fSpecs = {x.name: {} for x in s.iterdir() if x.is_dir() and su.parseFourierSpec(x.name)}	
				if len(fSpecs) == 0:
					raise FileNotFoundError("Spectra folder is empty.")
				parsed = [su.parseFourierSpec(x) for x in fSpecs.keys()]
				seen = []
				for p in parsed:
					if p in seen:
						raise ValueError("Duplicate fourier spec found.")
					seen.append(p)
				for fSpec, specContents in fSpecs.items():
					sFSpec = _getPriorityFolder(self.outDir/beamSpec/"spectra"/fSpec, _sFolderNames(self.path, beamSpec, fSpec), pu.getSpectra)
					if sFSpec is not None:
						_, l = sFSpec
						if len(l) != len(self.iqFiles) or [x.name.split(".",1)[1] for x in l] == [x.name.split(".",1)[1] for x in self.iqFiles]:
							raise ValueError("Number or names of spectra and iq files in folder do not match.")
						specContents["files"] = l
					else:
						raise FileNotFoundError(f"Could not find any spectra files in folder {fSpec}")
				contents["spectra"] = fSpecs

			if m is not None:
				_, l = m
				if len(l) != len(self.iqFiles) or [x.name.split(".",1)[1] for x in l] == [x.name.split(".",1)[1] for x in self.iqFiles]:
					raise ValueError("Number or names of moment and iq files do not match.")
				contents["moment_files"] = l

			if s is None and m is None:
				raise FileNotFoundError("Either Spectra or Moments must exist.")

	def toJson(self) -> str:
		return json.dumps(pu.pathToJson(self.path), indent='\t')
	
	def getIndex(self, filename: str, prefixIncluded: bool = True) -> int:
		if prefixIncluded:
			filename = filename.split(".",1)[1]
		return [x.name.split(".",1)[1] for x in self.iqFiles].index(filename)
	
	def getAtIndex(self, index: int, dtype: str, filename: str | None = None, beamSpec: str | None = None, fourierSpec: str | None = None) -> Path:
		if dtype not in ["moments", "spectra", "IQ"]:
			raise ValueError("dtype invalid.")
		if filename is not None:
			#relative
			index += self.getIndex(filename)

		if dtype == "IQ":
			return self.iqFiles[index]
		elif dtype == "moments":
			if beamSpec is None:
				raise ValueError("beamSpec must be set if moment requested.")
			return self.outFiles[beamSpec]["moment_files"][index]
		else:
			if (beamSpec is None or fourierSpec is None):
				raise ValueError("beamSpec and fourierSpec must be set if spectra requestied.")
			match = su.matchFourierSpec(self.outFiles[beamSpec]["spectra"].keys(), fourierSpec)
			if match is None:
				raise IndexError("No matching fourierSpec found.")
			return self.outFiles[beamSpec]["spectra"][match]["files"][index]

class _Computation:
	def _parseDependency(self, arg: Any):
		if not (isinstance(arg, dict) and (arg.keys() == {"frxx_data_type", "path", "field"})):
			return arg

		dType, pathJson, varName = (arg["frxx_data_type"], arg["path"], arg["field"])
		if varName.startswith("m_"):
			depVarName = varName[2:]
		else:
			depVarName = varName

		path = pu.validatePath(pu.jsonToPath(pathJson))
		if path is None:
			raise FileNotFoundError(f"Specified file does not exist: {path}")

		data = frxxDataFromFile(path)
		if data.type != dType:
			raise RuntimeError("Gotten file doesn't match requested type.")

		retVal = getattr(data, varName)

		if dType == "iq":
			dType = "IQ"
			self.iqDeps += depVarName
		elif dType == "m":
			dType = "moments"
			self.mDeps += depVarName
		elif dType == "s":
			dType = "spectra"
			self.sDeps += depVarName

		return retVal

	def __init__(self, 
			computationJson: dict | None = None, #option one

			returnVariables: List[str] | None = None, #option two
			fn: Callable | None = None, 
			selfFilePath: Path | None = None,
			selfDtype: str | None = None
		):
		if computationJson is None:
			if any(v is None for v in (returnVariables, fn, selfFilePath, selfDtype)):
				raise ValueError("all arguments are required")
			returnVariables, fn, selfFilePath, selfDtype = cast(
				Tuple[List[str], Callable, Path, str],
				(returnVariables, fn, selfFilePath, selfDtype)
			)

			self.retVars: List[str] = returnVariables
			self.function = fn
			self.selfFilePath = selfFilePath
			self.selfFileDtype = selfDtype
			self.kwargs = {}
			self.parsedKwargs = {}
			self.iqDeps = []
			self.mDeps = []
			self.sDeps = []
		else:
			self.retVars: List[str] = computationJson["return_variables"]
			self.function = su.strToFunc(computationJson["function"])
			self.selfFilePath = pu.jsonToPath(computationJson["self_filepath"])
			self.selfFileDtype = computationJson["self_dtype"]
			self.kwargs: dict = computationJson["kwargs"]
			self.iqDeps = []
			self.mDeps = []
			self.sDeps = []
			self.parsedKwargs = {k: self._parseDependency(self.kwargs[k]) for k in self.kwargs.keys()}
			
	def constructDep(self, depDtype: str, depVarName: str, beamSpec: str | None, fourierSpec: str | None = None):
		if depDtype not in ["moments", "spectra", "IQ"]:
			raise ValueError("dtype invalid.")
		currPath = self.selfFilePath.stem
		currName = self.selfFilePath.name

		if self.selfFileDtype == "moments":
			caseDirRel = Path('.')/".."/".."/".."
		elif self.selfFileDtype == "spectra":
			caseDirRel = Path('.')/".."/".."/".."/".."/".."
		else:
			caseDirRel = Path('.')/".."/".."

		caseDirRes = currPath/caseDirRel

		if depDtype == "IQ":
			paths = _getPriorityFolder(caseDirRes/"iq", _iqFolderNames(), pu.getIQ())
			if paths is None:
				raise FileNotFoundError("Can't find any IQ files.")
			path = caseDirRel/"iq"/paths[0].name/su.changeFilenameToType(currName, "IQ")
			path = pu.normPath(currPath, path)
		elif depDtype == "moments":
			if beamSpec is None:
				raise ValueError("beamSpec must be set if moment requested.")
			mDir = caseDirRes/"out"/beamSpec
			paths = _getPriorityFolder(mDir, _mFolderNames(caseDirRes, beamSpec), pu.getMoments())
			if paths is None:
				raise FileNotFoundError("Can't find any moment files.")
			path = caseDirRel/"out"/beamSpec/paths[0].name/su.changeFilenameToType(currName, "IQ")
			path = pu.normPath(currPath, path)
		else:
			if (beamSpec is None or fourierSpec is None):
				raise ValueError("beamSpec and fourierSpec must be set if spectra requestied.")
			specDir = (caseDirRes/"out"/beamSpec/"spectra")
			fourierSpecs = [x.name for x in specDir.iterdir() if x.is_dir() and su.parseFourierSpec(x.name)]
			match = su.matchFourierSpec(fourierSpecs, fourierSpec)
			if match is None:
				raise IndexError("No matching fourierSpec found.")
			paths = _getPriorityFolder(specDir/match, _sFolderNames(caseDirRes, beamSpec, fourierSpec), pu.getSpectra())
			if paths is None:
				raise FileNotFoundError("Can't find any spectra files.")
			path = caseDirRel/"out"/beamSpec/"spectra"/fourierSpec/paths[0].name/su.changeFilenameToType(currName, "IQ")
			path = pu.normPath(currPath, path)


		return {
			"frxx_data_type": depDtype,
			"path": path,
			"field": depVarName
		}
	
	def addKwargs(self, **kwargs):
		for k, v in kwargs.items():
			self.kwargs[k] = v
			self.parsedKwargs[k] = self._parseDependency(v)

	def compute(self):
		parsedKwargs = {k: self._parseDependency(self.kwargs[k]) for k in self.kwargs.keys()}
		self.results = self.function(**parsedKwargs)


	def toJson(self) -> dict:
			if len(self.kwargs) == 0:
				raise ValueError("Need some kwargs to define a valid computation.")
			return {}

		

class DelayedCompute:
	def __init__(self, delayedComputeJson: str | None = None, case: FrxxCase | None = None):
		self.vars: List[str] = []
		self.computations: List[_Computation] = []
			
	def toJson(self) -> str:
			return ""