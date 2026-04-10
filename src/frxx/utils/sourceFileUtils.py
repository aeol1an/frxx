from typing import List, Annotated, cast
from pathlib import Path

from . import pathUtils

Bounds = Annotated[List[List[int]], "Each sub-list must have two strictly increasing elements, "
									"and flattened list must be strictly increasing."]

class SourceFile:
	def __init__(self, pathJson: dict, indices: Bounds | None = None, count: int | None = None):
		if pathJson["file"] == "hardware":
			self.isHardware = True
			if count is None or count < 0:
				raise ValueError("Hardware SourceFile must have a non-negative count.")
			self.count = count
		else:
			self.isHardware = False
			self.count = None
		if (not self.isHardware) and indices is None:
			raise ValueError("Must pass indices if source isn't hardware.")
		self.pathJson = pathJson["file"]
		self.indices = indices
		if not self.verifyIdx():
			raise ValueError("Each sub-list must have two non-decreasing elements, "
							 "and flattened list must be strictly increasing.")

	@staticmethod
	def makeFromJson(sourceFileJson: dict) -> "SourceFile":
		if sourceFileJson["file"] == "hardware":
			return SourceFile({"file": "hardware"}, count=sourceFileJson.get("count", 0))
		return SourceFile({"file": sourceFileJson["file"]}, sourceFileJson["idx"])
	
	@staticmethod
	def makeFromPathAndLength(filename: str | Path, nIdx: int | None):
		if str(filename) == "hardware":
			if nIdx is None:
				raise ValueError("Hardware SourceFile must have a count (pass nIdx).")
			sfJson = {
				"file": "hardware",
			}
			return SourceFile(sfJson, count=nIdx)
		else:
			if nIdx is None:
				raise ValueError("Filename must be hardware if nIdx is none.")
			sfJson = {
				"file": pathUtils.pathToJson(Path(filename).resolve()),
			}
			return SourceFile(sfJson, [[0, nIdx-1]])
		
	def verifyIdx(self):
		if self.isHardware:
			self.indices = None
			return True
		else:
			self.indices = cast(Bounds, self.indices)
		prevB = None
		for pair in self.indices:
			if len(pair) != 2:
				raise ValueError("A pair doesn't have two indices.")
			a, b = tuple(pair)
			if (b - a) <= -1:
				raise ValueError("A pair was found to be decreasing")
			if prevB is not None and (a - prevB) <= 1:
				raise ValueError("Start of subsequent pair is not strictly "
								 "increasing and separated.")
			prevB = b
		return True

	def nRays(self) -> int:
		if self.isHardware:
			return cast(int, self.count)
		self.indices = cast(Bounds, self.indices)
		return sum(b - a + 1 for a, b in self.indices)

	def editPlatformPrefix(self, platform: str, prefix: str):
		if self.isHardware:
			raise ValueError("Can't change a file prefix on a hardware source.")
		
		self.pathJson = cast(dict, self.pathJson)
		self.pathJson = pathUtils.editPathJsonPrefix(self.pathJson, platform, prefix)
		return self

	def toJson(self):
		if self.isHardware:
			return {
				"file": "hardware",
				"count": self.count
			}
		return {
			"file": self.pathJson,
			"idx": self.indices
		}

	def concat(self, other: "SourceFile"):
		if self.isHardware != other.isHardware:
			raise ValueError("Both need to be hardware or SourceFile.")
		if self.isHardware:
			return SourceFile({"file": "hardware"}, count=cast(int, self.count) + cast(int, other.count))

		#else we have a real file and indices are Bounds
		self.pathJson = cast(dict, self.pathJson)
		other.pathJson = cast(dict, other.pathJson)
		if not pathUtils.pathJsonEqual(self.pathJson, other.pathJson):
			raise ValueError("Attempting to merge indices in files not equal.")
		self.indices = cast(Bounds, self.indices)
		other.indices = cast(Bounds, other.indices)
		intervals = sorted(self.indices + other.indices, key=lambda x: x[0])
		mergedIdx = []
		for start, end in intervals:
			if mergedIdx and start <= mergedIdx[-1][1] + 1:
				mergedIdx[-1][1] = max(mergedIdx[-1][1], end)
			else:
				mergedIdx.append([start, end])
		return SourceFile({"file": self.pathJson}, mergedIdx)
	def __add__(self, other):
		return self.concat(other)
	def __radd__(self, other):
		if other == 0:
			return self
		return self.__add__(other)
	def __iadd__(self, other):
		concatedSrc = self.concat(other)
		self.pathJson = concatedSrc.pathJson
		self.indices = concatedSrc.indices
		self.count = concatedSrc.count
		return self
	
	def breakAt(self, index: int) -> tuple["SourceFile", "SourceFile"]:
		if self.isHardware:
			self.count = cast(int, self.count)
			if index <= 0 or index >= self.count:
				raise ValueError(f"Index {index} is not within valid break range (1 to {self.count - 1}).")
			return (
				SourceFile({"file": "hardware"}, count=index),
				SourceFile({"file": "hardware"}, count=self.count - index)
			)
		
		self.indices = cast(Bounds, self.indices)
		
		if not self.indices:
			raise ValueError("Cannot break an empty sourceFile.")
		
		firstStart = self.indices[0][0]
		lastEnd = self.indices[-1][1]
		
		if index <= firstStart or index > lastEnd:
			raise ValueError(f"Index {index} is not within valid break range ({firstStart + 1} to {lastEnd}).")
		
		leftIndices = []
		rightIndices = []
		
		for start, end in self.indices:
			if end < index:
				leftIndices.append([start, end])
			elif start >= index:
				rightIndices.append([start, end])
			else:
				# index is within this range, split it
				leftIndices.append([start, index - 1])
				rightIndices.append([index, end])
		
		left = SourceFile({"file": self.pathJson}, leftIndices)
		right = SourceFile({"file": self.pathJson}, rightIndices)
		
		return left, right