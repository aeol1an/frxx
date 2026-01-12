import numpy as np
import xarray as xr
import json
from numpy.typing import NDArray
from datetime import datetime
import pytz
from .data import frxxData

class IQ(frxxData):
	def setTime(self, unixTimeArr: NDArray[np.float64], time_zone: str = 'zulu'):
		if (unixTimeArr.dtype != np.float64):
			raise TypeError("Expected array of np.float64")
		
		startTime = unixTimeArr[0]
		endTime = unixTimeArr[-1]
		
		timeVar = unixTimeArr - startTime
		nRays = len(timeVar)
		
		self.dimensions['time'] = nRays
		
		startTimeStr = datetime.fromtimestamp(startTime, tz=pytz.timezone(time_zone))\
			.astimezone(pytz.utc).isoformat()
		endTimeStr = datetime.fromtimestamp(endTime, tz=pytz.timezone(time_zone))\
			.astimezone(pytz.utc).isoformat()
			
		self.rootAttrs["time_coverage_start"] = startTimeStr.replace('+00:00', 'Z')
		self.rootAttrs["start_datetime"] = startTimeStr
		self.rootAttrs["time_coverage_end"] = endTimeStr.replace('+00:00', 'Z')
		self.rootAttrs["end_datetime"] = endTimeStr
		
		paddedStartTime = startTimeStr.replace('+00:00', 'Z') +\
			(self.dimensions["string_length_32"] - len(startTimeStr.replace('+00:00', 'Z')))*' '
		paddedEndTime = startTimeStr.replace('+00:00', 'Z') +\
			(self.dimensions["string_length_32"] - len(startTimeStr.replace('+00:00', 'Z')))*' '
			
		self.variables["time_coverage_start"]["data"] =\
			np.array([c for c in paddedStartTime], dtype="|S1")
		self.variables["time_coverage_end"]["data"] =\
			np.array([c for c in paddedEndTime], dtype="|S1")
			
		self.variables["time"]["units"] = "seconds since " + startTimeStr.replace('+00:00', 'Z')
		self.variables["time"]["data"] = np.ma.masked_invalid(timeVar)
		
		self.variables["sweep_start_ray_index"]["data"] = np.array([0], dtype=np.int32)
		self.variables["sweep_end_ray_index"]["data"] = np.array([nRays-1], dtype=np.int32)
		
		self.requiredBools["time"] = True
		
	def setRange(self, rangeGates: NDArray[np.float32]):
		if (rangeGates.dtype != np.float32):
			raise TypeError("Expected array of np.float32")
		
		nGates = len(rangeGates)
		firstGate = np.rint(rangeGates[0])
		dG = np.rint(rangeGates[1]-rangeGates[0])
		
		self.dimensions["range"] = nGates
		
		self.variables["range"]["meters_to_center_of_first_gate"] = str(firstGate)
		self.variables["range"]["meters_between_gates"] = str(dG)
		
		self.variables["range"]["data"] = np.ma.masked_invalid(rangeGates)
		
		self.requiredBools["range"] = True
		
	def setPosition(self, lat: float, lon: float):
		if lat < -90 or lat > 90:
			raise ValueError(f'Latitude {lat} out of -90 to 90 deg range.')
		if lon < -180 or lon > 180:
			raise ValueError(f'Longitude {lon} out of -180 to 180 deg range.')
		
		self.variables["latitude"]["data"] = np.ma.masked_invalid(lat)
		self.variables["longitude"]["data"] = np.ma.masked_invalid(lon)
		
		self.requiredBools["position"] = True
		
	def setScanningStrategy(self, strategy: str):
		if strategy == "ppi":
			self.variables["sweep_mode"]["data"] =\
				np.array([[c for c in 'azimuth_surveillance            ']], dtype='|S1')
			self.variables["fixed_angle"]["units"] = "elevation degrees"
		else:
			raise ValueError("Sorry, only ppi mode supported for now.")
		
		self.requiredBools["scanning_strategy"] = True
	
	def setTargetAngle(self, targetAngle: float):
		if not self.requiredBools["scanning_strategy"]:
			raise RuntimeError("Need to call setScanningStrategy() before this function.")
		if self.variables["fixed_angle"]["units"] == "elevation degrees":
			#ppi mode
			if targetAngle < 0 or targetAngle > 90:
				raise ValueError("Radar dish shouldn't be pointing into "
									"the floor or greater than vertical.")
			self.variables["fixed_angle"]["data"] =\
				np.ma.masked_invalid(np.array([targetAngle], dtype=np.float32))
		else:
			raise ValueError("Sorry, only ppi mode supported for now.")
		
		self.requiredBools["target_angle"] = True
	
	def setAzimuth(self, azimuths: NDArray[np.float32]):
		if (azimuths.dtype != np.float32):
			raise TypeError("Expected array of np.float32")
		if not self.requiredBools["time"]:
			raise RuntimeError("Need to call setTime() before this function.")
		if len(azimuths) != self.dimensions["time"]:
			raise RuntimeError("Number of azimuths need to measure number "
							   "of rays from setTime() function call. "
							   f'For this file, that is {self.dimensions["time"]} rays.')
		
		self.variables["azimuth"]["data"] = np.ma.masked_invalid(azimuths)
		
		self.requiredBools["azimuth"] = True
		
	def setElevation(self, elevations: NDArray[np.float32]):
		if (elevations.dtype != np.float32):
			raise TypeError("Expected array of np.float32")
		if not self.requiredBools["time"]:
			raise RuntimeError("Need to call setTime() before this function.")
		if len(elevations) != self.dimensions["time"]:
			raise RuntimeError("Number of elevations need to measure number "
							   "of rays from setTime() function call. "
							   f'For this file, that is {self.dimensions["time"]} rays.')
			
		self.variables["elevation"]["data"] = np.ma.masked_invalid(elevations)
		
		self.requiredBools["elevation"] = True
	
	def setPulseWidthSeconds(self, pulseWidths: NDArray[np.float32]):
		if (pulseWidths.dtype != np.float32):
			raise TypeError("Expected array of np.float32")
		if not self.requiredBools["time"]:
			raise RuntimeError("Need to call setTime() before this function.")
		if len(pulseWidths) != self.dimensions["time"]:
			raise RuntimeError("Number of pulse widths need to measure number "
							   "of rays from setTime() function call. "
							   f'For this file, that is {self.dimensions["time"]} rays.')
			
		self.variables["pulse_width"]["data"] = np.ma.masked_invalid(pulseWidths)
		
		self.requiredBools["pulse_width"] = True
		
	def setPrtSeconds(self, pulse_repetition_times: NDArray[np.float32]):
		if (pulse_repetition_times.dtype != np.float32):
			raise TypeError("Expected array of np.float32")
		if not self.requiredBools["time"]:
			raise RuntimeError("Need to call setTime() before this function.")
		if len(pulse_repetition_times) != self.dimensions["time"]:
			raise RuntimeError("Number of prt values need to measure number "
							   "of rays from setTime() function call. "
							   f'For this file, that is {self.dimensions["time"]} rays.')
			
		self.variables["prt"]["data"] = np.ma.masked_invalid(pulse_repetition_times)
		
		self.requiredBools["prt"] = True
		
	def setWavelengthMeters(self, wavelengths: NDArray[np.float32]):
		if (wavelengths.dtype != np.float32):
			raise TypeError("Expected array of np.float32")
		if not self.requiredBools["time"]:
			raise RuntimeError("Need to call setTime() before this function.")
		if not self.requiredBools["prt"]:
			raise RuntimeError("Need to call setPrtSeconds() before this function, "
							   "for nyquist velocity calculation.")
		if len(wavelengths) != self.dimensions["time"]:
			raise RuntimeError("Number of wavelength values need to measure number "
							   "of rays from setTime() function call. "
							   f'For this file, that is {self.dimensions["time"]} rays.')
		
		self.variables["wavelength"]["data"] = wavelengths
		self.variables["nyquist_velocity"]["data"] =\
			np.ma.masked_invalid(0.25 * wavelengths / self.variables["prt"]["data"])
			
		self.requiredBools["wavelength"] = True

	def __init__(self):
		super().__init__()
		
		self.ds.attrs["dataType"] = "iq"

		self.ds = self.ds.assign_coords(
			iq=np.arange(2)
		)

		self.optionalBools = {
			#None of these are require true, but are helpful for documentation
			"history": False,
			"addtl_comments": False,
		}