import numpy as np
from numpy.typing import NDArray
from datetime import datetime
import pytz
import netCDF4

class raxpolCf:
    #------Start Required Functions-----------------------------------------------------------------
    def setVolume(self, volNum: int = 0):
        self.variables["volume_number"]["data"] = volNum
        self.requiredBools["volume"] = True
        
    def setSweep(self, sweepNum: int = 0):
        self.variables["sweep_number"]["data"] = np.array([sweepNum], dtype=np.int32)
        self.requiredBools["sweep"] = True
        
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
    #------End Required Functions-------------------------------------------------------------------
    
    
    #------Start Data Functions---------------------------------------------------------------------
    def setDBZ(self, DBZ: NDArray[np.float64]):
        if (DBZ.dtype != np.float64):
            raise TypeError("Expected array of np.float64")
        if not self.requiredBools["time"]:
            raise RuntimeError("Need to call setTime() before this function.")
        if not self.requiredBools["range"]:
            raise RuntimeError("Need to call setRange() before this function.")
        if DBZ.shape != (self.dimensions["time"], self.dimensions["range"]):
            raise RuntimeError("Number of reflectivity values need to measure number "
                               "of rays and gates from setTime() abd setRange() function calls. "
                               f'For this file, that is {self.dimensions["time"]} rays, '
                               f'and {self.dimensions["range"]} gates.')
        
        self.variables["DBZ"]["data"] = np.ma.masked_invalid(DBZ)
        
        self.radarVarBools["DBZ"] = True
        
    def setVEL(self, VEL: NDArray[np.float64]):
        if (VEL.dtype != np.float64):
            raise TypeError("Expected array of np.float64")
        if not self.requiredBools["time"]:
            raise RuntimeError("Need to call setTime() before this function.")
        if not self.requiredBools["range"]:
            raise RuntimeError("Need to call setRange() before this function.")
        if VEL.shape != (self.dimensions["time"], self.dimensions["range"]):
            raise RuntimeError("Number of velocity values need to measure number "
                               "of rays and gates from setTime() abd setRange() function calls. "
                               'For this file, that is {self.dimensions["time"]} rays, '
                               f'and {self.dimensions["range"]} gates.')
        
        self.variables["VEL"]["data"] = np.ma.masked_invalid(VEL)
        
        self.radarVarBools["VEL"] = True
        
    def setWIDTH(self, WIDTH: NDArray[np.float64]):
        if (WIDTH.dtype != np.float64):
            raise TypeError("Expected array of np.float64")
        if not self.requiredBools["time"]:
            raise RuntimeError("Need to call setTime() before this function.")
        if not self.requiredBools["range"]:
            raise RuntimeError("Need to call setRange() before this function.")
        if WIDTH.shape != (self.dimensions["time"], self.dimensions["range"]):
            raise RuntimeError("Number of spectrum width values need to measure number "
                               "of rays and gates from setTime() abd setRange() function calls. "
                               f'For this file, that is {self.dimensions["time"]} rays, '
                               f'and {self.dimensions["range"]} gates.')
        
        self.variables["WIDTH"]["data"] = np.ma.masked_invalid(WIDTH)
        
        self.radarVarBools["WIDTH"] = True
        
    def setZDR(self, ZDR: NDArray[np.float64]):
        if (ZDR.dtype != np.float64):
            raise TypeError("Expected array of np.float64")
        if not self.requiredBools["time"]:
            raise RuntimeError("Need to call setTime() before this function.")
        if not self.requiredBools["range"]:
            raise RuntimeError("Need to call setRange() before this function.")
        if ZDR.shape != (self.dimensions["time"], self.dimensions["range"]):
            raise RuntimeError("Number of differential reflectivity values need to measure number "
                               "of rays and gates from setTime() abd setRange() function calls. "
                               f'For this file, that is {self.dimensions["time"]} rays, '
                               f'and {self.dimensions["range"]} gates.')
            
        self.variables["ZDR"]["data"] = np.ma.masked_invalid(ZDR)
        
        self.radarVarBools["ZDR"] = True
        
    def setPHIDP(self, PHIDP: NDArray[np.float64], units: str):
        if (PHIDP.dtype != np.float64):
            raise TypeError("Expected array of np.float64")
        if not self.requiredBools["time"]:
            raise RuntimeError("Need to call setTime() before this function.")
        if not self.requiredBools["range"]:
            raise RuntimeError("Need to call setRange() before this function.")
        if PHIDP.shape != (self.dimensions["time"], self.dimensions["range"]):
            raise RuntimeError("Number of differential phase values need to measure number "
                               "of rays and gates from setTime() abd setRange() function calls. "
                               f'For this file, that is {self.dimensions["time"]} rays, '
                               f'and {self.dimensions["range"]} gates.')
        if not (units == "degrees" or units == "radians"):
            raise ValueError("Units required, and can only be \"degres\" or \"radians\".")

        self.variables["PHIDP"]["units"] = units
        self.variables["PHIDP"]["data"] = np.ma.masked_invalid(PHIDP)
        
        self.radarVarBools["PHIDP"] = True
        
    def setRHOHV(self, RHOHV: NDArray[np.float64]):
        if (RHOHV.dtype != np.float64):
            raise TypeError("Expected array of np.float64")
        if not self.requiredBools["time"]:
            raise RuntimeError("Need to call setTime() before this function.")
        if not self.requiredBools["range"]:
            raise RuntimeError("Need to call setRange() before this function.")
        if RHOHV.shape != (self.dimensions["time"], self.dimensions["range"]):
            raise RuntimeError("Number of correlation coefficient values need to measure number "
                               "of rays and gates from setTime() abd setRange() function calls. "
                               f'For this file, that is {self.dimensions["time"]} rays, '
                               f'and {self.dimensions["range"]} gates.')

        self.variables["RHOHV"]["data"] = np.ma.masked_invalid(RHOHV)
        
        self.radarVarBools["RHOHV"] = True
        
    def setSNRH(self, SNRH: NDArray[np.float64]):
        if (SNRH.dtype != np.float64):
            raise TypeError("Expected array of np.float64")
        if not self.requiredBools["time"]:
            raise RuntimeError("Need to call setTime() before this function.")
        if not self.requiredBools["range"]:
            raise RuntimeError("Need to call setRange() before this function.")
        if SNRH.shape != (self.dimensions["time"], self.dimensions["range"]):
            raise RuntimeError("Number of signal to noise ratio values need to measure number "
                               "of rays and gates from setTime() abd setRange() function calls. "
                               f'For this file, that is {self.dimensions["time"]} rays, '
                               f'and {self.dimensions["range"]} gates.')

        self.variables["SNRH"]["data"] = np.ma.masked_invalid(SNRH)
        
        self.radarVarBools["SNRH"] = True

    def setSNRV(self, SNRV: NDArray[np.float64]):
        if (SNRV.dtype != np.float64):
            raise TypeError("Expected array of np.float64")
        if not self.requiredBools["time"]:
            raise RuntimeError("Need to call setTime() before this function.")
        if not self.requiredBools["range"]:
            raise RuntimeError("Need to call setRange() before this function.")
        if SNRV.shape != (self.dimensions["time"], self.dimensions["range"]):
            raise RuntimeError("Number of signal to noise ratio values need to measure number "
                               "of rays and gates from setTime() abd setRange() function calls. "
                               f'For this file, that is {self.dimensions["time"]} rays, '
                               f'and {self.dimensions["range"]} gates.')

        self.variables["SNRV"]["data"] = np.ma.masked_invalid(SNRV)
        
        self.radarVarBools["SNRV"] = True
    #------End Data Functions-----------------------------------------------------------------------
    
    
    #------Begin Optional Data Functions------------------------------------------------------------
    def setPulseBoundaries(self, boundaries: NDArray[np.int32]):
        if (boundaries.dtype != np.int32):
            raise TypeError("Expected array of np.int32")
        if not self.requiredBools["time"]:
            raise RuntimeError("Need to call setTime() before this function.")
        if boundaries.shape != (self.dimensions["time"], self.dimensions["ray_start_end"]):
            raise RuntimeError("Number of start-end pairs need to match number of rays.")
        
        self.variables["pulse_boundaries"]["data"] = boundaries
        
        self.optionalVarBools["pulse_boundaries"] = True
    #-----------------------------------------------------------------------------------------------
    
    
    #------Begin Optional Documentation Functions---------------------------------------------------
    def setTitle(self, title: str):
        self.rootAttrs["title"] = title
        self.optionalBools["title"] = True
        
    def setHistory(self, history: str):
        self.rootAttrs["history"] = history
        self.optionalBools["history"] = True
        
    def setRadarTeam(self, team: str):
        self.rootAttrs["radar_team"] = team
        self.optionalBools["radar_team"] = True
        
    def setAddtlComments(self, comments: str):
        self.rootAttrs["comment"] = "Generated by raxpolCf.py "\
                                    "(Author: Ameya Naik, https://github.com/aeol1an). "\
                                    "Adapted from convert_px_cfrad23.m. " + comments
        self.optionalBools["addtl_comments"] = True
    #------End Optional Functions-------------------------------------------------------------------
    
    def saveToFile(self, filename: str):
        if not np.all(np.array([val for val in self.requiredBools.values()])):
            missing_functions = ""
            for key, val in self.requiredBools.items():
                if not val:
                    missing_functions += "\n" + key
            raise RuntimeError("All required functions need to be "
                               "called. Missing:" + missing_functions)
        if not np.any(np.array([val for val in self.radarVarBools.values()])):
            raise RuntimeError("At least one radar variable needs to be set.")
        
        file = netCDF4.Dataset(filename, 'w', format='NETCDF4')
        
        for key, value in self.rootAttrs.items():
            setattr(file, key, value)

        for key, value in self.dimensions.items():
            file.createDimension(key, value)
        
        #Define dict fields that are not just string attributes
        coreVarFields = ["type", "fill_value", "dims", "data"]
        for var in self.requiredVars:
            varDict = self.variables[var]
            ncvar = file.createVariable(var, varDict["type"], varDict["dims"], 
                                        fill_value=varDict["fill_value"])
            for key, val in varDict.items():
                if not key in coreVarFields:
                    setattr(ncvar, key, val)
            ncvar[:] = varDict["data"]
            
        for var, exists in self.optionalVarBools.items():
            if not exists:
                continue
            varDict = self.variables[var]
            ncvar = file.createVariable(var, varDict["type"], varDict["dims"], 
                                        fill_value=varDict["fill_value"])
            for key, val in varDict.items():
                if not key in coreVarFields:
                    setattr(ncvar, key, val)
            ncvar[:] = varDict["data"]
                
        for var, exists in self.radarVarBools.items():
            if not exists:
                continue
            varDict = self.variables[var]
            ncvar = file.createVariable(var, varDict["type"], varDict["dims"], 
                                        fill_value=varDict["fill_value"])
            for key, val in varDict.items():
                if not key in coreVarFields:
                    setattr(ncvar, key, val)
            ncvar[:] = varDict["data"]
            
        file.close()

    def __init__(self):
        self.requiredBools = {
            #All of these are required true
            "volume": False,
            "sweep": False,
            "time": False,
            "range": False,
            "position": False,
            "scanning_strategy": False,
            "target_angle": False,
            "azimuth": False,
            "elevation": False,
            "pulse_width": False,
            "prt": False,
            "wavelength": False
        }

        self.optionalBools = {
            #None of these are require true, but are helpful for documentation
            "title": False,
            "history": False,
            "radar_team": False,
            "addtl_comments": False,
        }
        
        self.rootAttrs = {
            "Conventions": "CF/Radial",
            "title": "",
            "institution": "University of Oklahoma",
            "references": "https://github.com/OURadar/RadarKit",
            "source": "RadarKit raw I/Q",
            "history": "",
            "comment": "Generated by raxpolCf.py (Author: Ameya Naik, https://github.com/aeol1an). "
                       "Adapted from convert_px_cfrad23.m.",
            "instrument_name": "RaXPol",
            "radar_team": "",
            "time_coverage_start": "tbd",
            "time_coverage_end": "tbd",
            "start_datetime": "tbd",
            "end_datetime": "tbd",
            "version": "CF-Radial-1.3",
        }
        
        self.dimensions = {
            "time": "tbd",
            "range": "tbd",
            "sweep": 1,
            "ray_start_end": 2,
            "string_length_8": 8,
            "string_length_32": 32
        }
        
        self.requiredVars = [
            "volume_number",
            "time_coverage_start",
            "time_coverage_end",
            "latitude",
            "longitude",
            "altitude",
            "sweep_number",
            "sweep_mode",
            "fixed_angle",
            "sweep_start_ray_index",
            "sweep_end_ray_index",
            "time",
            "range",
            "azimuth",
            "elevation",
            "pulse_width",
            "prt",
            "wavelength",
            "nyquist_velocity"
        ]
        
        self.optionalVarBools = {
            "pulse_boundaries": False
        }
        
        self.radarVarBools = {
            #At least one is required true
            "DBZ": False,
            "VEL": False,
            "WIDTH": False,
            "ZDR": False,
            "PHIDP": False,
            "RHOHV": False,
            "SNRH": False,
            "SNRV": False,
        }
        
        self.variables = {
            #Required variables here
            "volume_number": {
                "type": "i4",
                "fill_value": -9999,
                "dims": (),
                "standard_name": "data_volume_index_number",
                "data": "tbd"
            },
            "time_coverage_start": {
                "type": "|S1",
                "fill_value": b' ',
                "dims": ("string_length_32",),
                "standard_name": "data_volume_start_time_utc",
                "comment": "ray times are relative to start time in secs",
                "data": "tbd"
            },
            "time_coverage_end": {
                "type": "|S1",
                "fill_value": b' ',
                "dims": ("string_length_32",),
                "standard_name": "data_volume_end_time_utc",
                "comment": "ray times are relative to start time in secs",
                "data": "tbd"
            },
            "latitude": {
                "type": "f8",
                "fill_value": -9999.0,
                "dims": (),
                "long_name": "latitude",
                "units": "degrees_north",
                "data": "tbd",
            },
            "longitude": {
                "type": "f8",
                "fill_value": -9999.0,
                "dims": (),
                "long_name": "longitude",
                "units": "degrees_east",
                "data": "tbd",
            },
            "altitude": {
                "type": "f8",
                "fill_value": -9999.0,
                "dims": (),
                "long_name": "altitude",
                "units": "meters",
                "data": 2.5
            },
            "sweep_number": {
                "type": "i4",
                "fill_value": -9999,
                "dims": ("sweep",),
                "long_name": "sweep_index_number_0_based",
                "data": "tbd"
            },
            "sweep_mode": {
                "type": "|S1",
                "fill_value": b' ',
                "dims": ("sweep", "string_length_32"),
                "long_name": "scan_mode_for_sweep",
                "data": "tbd"
            },
            "fixed_angle": {
                "type": "f4",
                "fill_value": -9999.0,
                "dims": ("sweep",),
                "long_name": "ray_target_fixed_angle",
                "units": "tbd",
                "data": "tbd"
            },
            "sweep_start_ray_index": {
                "type": "i4",
                "fill_value": -9999,
                "dims": ("sweep",),
                "long_name": "index_of_first_ray_in_sweep",
                "data": "tbd"
            },
            "sweep_end_ray_index": {
                "type": "i4",
                "fill_value": -9999,
                "dims": ("sweep",),
                "long_name": "index_of_last_ray_in_sweep",
                "data": "tbd"
            },
            "time": {
                "type": "f8",
                "fill_value": -9999.0,
                "dims": ("time",),
                "standard_name": "time",
                "long_name": "time in seconds since volume start",
                "units": "tbd",
                "data": "tbd",
            },
            "range": {
                "type": "f4",
                "fill_value": -9999.0,
                "dims": ("range",),
                "long_name": "Range from instrument to center of gate",
                "units": "meters",
                "spacing_is_constant": "true",
                "meters_to_center_of_first_gate": "tbd",
                "meters_between_gates": "tbd",
                "data": "tbd"
            },
            "azimuth": {
                "type": "f4",
                "fill_value": -9999.0,
                "dims": ("time",),
                "long_name": "ray_azimuth_angle",
                "units": "degrees",
                "data": "tbd"
            },
            "elevation": {
                "type": "f4",
                "fill_value": -9999.0,
                "dims": ("time",),
                "long_name": "ray_elevtion_angle",
                "units": "degrees",
                "data": "tbd",
            },
            "pulse_width": {
                "type": "f4",
                "fill_value": -9999.0,
                "dims": ("time",),
                "long_name": "transmitter_pulse_width",
                "units": "seconds",
                "data": "tbd"
            },
            "prt": {
                "type": "f4",
                "fill_value": -9999.0,
                "dims": ("time",),
                "long_name": "pulse_repetition_time",
                "units": "seconds",
                "data": "tbd"
            },
            "wavelength": {
                "type": "f4",
                "fill_value": -9999.0,
                "dims": ("time",),
                "long_name": "radar_wavelength",
                "units": "meters",
                "data": "tbd"
            },
            "nyquist_velocity": {
                "type": "f4",
                "fill_value": -9999.0,
                "dims": ("time",),
                "long_name": "unambiguous_doppler_velocity",
                "units": "meters per second",
                "data": "tbd"
            },
            
            #Add optional variables here
            "pulse_boundaries": {
                "type": "i4",
                "fill_value": -9999,
                "dims": ("time", "ray_start_end"),
                "long_name": "first_and_last_pulse_indices_in_ray",
                "comment": "First and last pulse index in a ray in corresponding rkc file. Values "
                           "valid after filtering pulses with goodData.csv and badPulseSwaths.csv.",
                "data": "tbd"
            },
            
            #Add radar variables here
            "DBZ": {
                "type": "i2",
                "fill_value": -32768,
                "dims": ("time", "range"),
                "long_name": "reflectivity",
                "standard_name": "equivalent_reflectivity_factor",
                "units": "dBZ",
                "scale_factor": 0.01,
                "add_offset": 0.0,
                "grid_mapping": "grid_mapping",
                "coordinates": "time range",
                "data": "tbd"
            },
            "VEL": {
                "type": "i2",
                "fill_value": -32768,
                "dims": ("time", "range"),
                "long_name": "doppler_velocity",
                "standard_name": "radial_velocity_of_scatterers_away_from_instrument",
                "units": "m/s",
                "scale_factor": 0.01,
                "add_offset": 0.0,
                "grid_mapping": "grid_mapping",
                "coordinates": "time range",
                "data": "tbd"
            },
            "WIDTH": {
                "type": "i2",
                "fill_value": -32768,
                "dims": ("time", "range"),
                "long_name": "spectrum_width",
                "standard_name": "doppler_spectrum_width",
                "units": "m/s",
                "scale_factor": 0.01,
                "add_offset": 0.0,
                "grid_mapping": "grid_mapping",
                "coordinates": "time range",
                "data": "tbd"
            },
            "ZDR": {
                "type": "i2",
                "fill_value": -32768,
                "dims": ("time", "range"),
                "long_name": "differential_reflectivity",
                "standard_name": "log_differential_reflectivity_hv",
                "units": "dB",
                "scale_factor": 0.01,
                "add_offset": 0.0,
                "grid_mapping": "grid_mapping",
                "coordinates": "time range",
                "data": "tbd"
            },
            "RHOHV": {
                "type": "i2",
                "fill_value": -32768,
                "dims": ("time", "range"),
                "long_name": "cross_correlation_ratio",
                "standard_name": "cross_correlation_ratio_hv",
                "units": "unitless",
                "scale_factor": 0.01,
                "add_offset": 0.0,
                "grid_mapping": "grid_mapping",
                "coordinates": "time range",
                "data": "tbd"
            },
            "PHIDP": {
                "type": "i2",
                "fill_value": -32768,
                "dims": ("time", "range"),
                "long_name": "differential_phase",
                "standard_name": "differential_phase_hv",
                "units": "tbd",
                "scale_factor": 0.01,
                "add_offset": 0.0,
                "grid_mapping": "grid_mapping",
                "coordinates": "time range",
                "data": "tbd"
            },
            "SNRH": {
                "type": "i2",
                "fill_value": -32768,
                "dims": ("time", "range"),
                "long_name": "horizontal_channel_signal_to_noise_ratio",
                "standard_name": "signal_to_noise_ratio_h",
                "units": "dB",
                "scale_factor": 0.01,
                "add_offset": 0.0,
                "grid_mapping": "grid_mapping",
                "coordinates": "time range",
                "data": "tbd"
            },
            "SNRV": {
                "type": "i2",
                "fill_value": -32768,
                "dims": ("time", "range"),
                "long_name": "vertical_channel_signal_to_noise_ratio",
                "standard_name": "signal_to_noise_ratio_v",
                "units": "dB",
                "scale_factor": 0.01,
                "add_offset": 0.0,
                "grid_mapping": "grid_mapping",
                "coordinates": "time range",
                "data": "tbd"
            },
        }