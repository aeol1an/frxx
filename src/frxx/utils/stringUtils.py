from typing import Tuple, List, Any, Annotated

import re

AvgStrat = Annotated[Tuple[int, int] | None, "None if no averaging, or a Tuple of (K_r, K_az)"]

def constructBeamSpec(angleSpacingDeg: float, beamOverlapDeg: float):
	 return ("s"+str(angleSpacingDeg)+"o"+str(beamOverlapDeg))
def parseBeamSpec(arg: str):
	match = re.match(r'^s([\d.+-]+)o([\d.+-]+)$', arg.strip())
	if not match:
		return None

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
		("" if avgStrat is None else f"aR{avgStrat[0]}A{avgStrat[1]}") +
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
				return None
			idx += 1
			r = readInt()
			if idx >= len(s) or s[idx] != 'A':
				return None
			idx += 1
			a = readInt()
			result["avgStrat"] = None if (r == 1 and a == 1) else (r, a)
		elif c == 'b':
			idx += 1
			result["bootstrap"] = readInt()
		else:
			return None
	return result
def matchFourierSpec(candidates: List[str], toMatch: str) -> str | None:
	target = parseFourierSpec(toMatch)
	if target is None:
		raise ValueError(f"Failed to parse spec: {toMatch}")
	for c in candidates:
		parsed = parseFourierSpec(c)
		if parsed is None:
			raise ValueError(f"Failed to parse spec: {c}")
		if parsed == target:
			return c
	return None