from __future__ import annotations
from typing import List, Tuple, Any

import importlib

from pathlib import Path

import numpy as np
import xarray as xr

import re
import json

from ..core import IQ, moments, spectra
from ..utils import pathUtils as pu
from ..utils import stringUtils as su

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

class FrxxCase:
	def __init__(self, path: str | Path):
		self.path: Path = pu.validatePath(path, mustBeDir=True) #type: ignore
		if self.path is None:
			raise FileNotFoundError("Path is not a valid directory.")
		
		self.iqDir: Path = pu.validatePath(self.path/"iq", mustBeDir=True) #type: ignore
		if self.iqDir is None:
			raise FileNotFoundError("Must have an iq directory!")
		
		sr = _getPriorityFolder(self.iqDir, ["raw_headers", "raw", "qc_headers", "qc"], pu.getIQ)
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
			delayedMoment = [x.name for x in pu.globPath(self.outDir/beamSpec/"moments_delayed_*") if x.is_dir()][::-1]

			m = _getPriorityFolder(self.outDir/beamSpec, [*delayedMoment, "moments"], pu.getMoments)
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
					delayedSpectra = [x.name for x in pu.globPath(self.outDir/beamSpec/"spectra"/fSpec/"delayed_*") if x.is_dir()][::-1]
					sFSpec = _getPriorityFolder(self.outDir/beamSpec/"spectra"/fSpec, [*delayedSpectra, "full"], pu.getSpectra)
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
	
	def getAtIndex(self, index: int, dtype: str, filename: str | None = None, beamSpec: str | None = None, fourierSpec: str | None = None):
		if dtype not in ["moments", "spectra", "IQ"]:
			raise ValueError("dtype invalid.")
		if filename is not None:
			#relative
			index += self.getIndex(filename)

		if dtype == "IQ":
			return self.iqFiles[index]
		if dtype == "moments":
			if beamSpec is None:
				raise ValueError("beamSpec must be set if moment requested.")
			return self.outFiles[beamSpec]["moment_files"][index]
		if dtype == "spectra":
			if (beamSpec is None or fourierSpec is None):
				raise ValueError("beamSpec and fourierSpec must be set if spectra requestied.")
			match = su.matchFourierSpec(self.outFiles[beamSpec]["spectra"].keys(), fourierSpec)
			if match is None:
				raise IndexError("No matching fourierSpec found.")
			return self.outFiles[beamSpec]["spectra"][match]["files"][index]




class _Computation:
	def parseSpec(
			self, arg: Any, 
			case: FrxxCase | None = None, curFilename: str | None = None, 
			beamSpec: str | None = None, fourierSpec: str | None = None
		):
		if not isinstance(arg, str):
			return arg

		match = re.match(r'^(iq|m|s)\[(.*?)\]->(.+)$', arg.strip())
		if not match:
			return arg

		dType, indexStr, varName = match.groups()
		if dType == "iq":
			dType = "IQ"
		elif dType == "m":
			dType = "moments"
		elif dType == "s":
			dType = "spectra"
		indexStr = indexStr.strip()
		varName = varName.strip()

		if indexStr == '':
			if case is None:
				raise ValueError("A valid case is needed to index by number.")
			path = case.getAtIndex(0, dType, curFilename, beamSpec, fourierSpec)

		elif indexStr.startswith('r'):
			if case is None:
				raise ValueError("A valid case is needed to index by number.")
			relIdx = int(indexStr[1:])
			path = case.getAtIndex(relIdx, dType, curFilename, beamSpec, fourierSpec)

		elif re.match(r'^-?\d+$', indexStr):
			if case is None:
				raise ValueError("A valid case is needed to index by number.")
			absIdx = int(indexStr)
			path = case.getAtIndex(absIdx, dType, None, beamSpec, fourierSpec)

		else:
			path = Path(indexStr).resolve()
			if not path.exists():
				raise FileNotFoundError(f"Specified file does not exist: {path}")

		return getVar(path, varName) #TODO implement this, should be quite easy.


	def __init__(self, computationJson: str | None = None, case: FrxxCase | None = None):
		if computationJson is None:
			#start from scratch - TODO here
		else:
			computationDict = json.loads(computationJson)
			self.retVars: List[str] = computationDict["return_variables"]
			path, fn = computationDict["function"].rsplit(".", 1)
			self.function = getattr(importlib.import_module(path), fn)

			curFileName = computationDict["filename"]
			beamSpec = computationDict["beam_spec"]
			fourierSpec = computationDict["fourier_spec"]

			self.kwargs: dict = computationDict["kwargs"]
			self.kwargs = {k: self.parseSpec(self.kwargs[k], case, ) for k in self.kwargs.keys()}

			
	def toJson(self) -> str:
			return ""

		

class DelayedCompute:
	def __init__(self, delayedComputeJson: str | None = None, case: FrxxCase | None = None):
		self.vars: List[str] = []
		self.computations: List[_Computation] = []
			
	def toJson(self) -> str:
			return ""