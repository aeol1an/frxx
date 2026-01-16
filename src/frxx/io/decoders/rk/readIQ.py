from pathlib import Path
import os

import numpy as np
import dask.array as da
from dask import delayed
import struct

from typing import Union
from numpy.typing import NDArray

from .... import core

class rkcfile:
    def __init__(self, filename, maxPulse = None, posFilename = None, verbose = True):
        
        #------Class Properties---------------------------------------------------------------------
        self.constants = {
            'RKName': 128,
            'RKFileHeader': 4096,
            'RKMaxMatchedFilterCount': 8,
            'RKFilterAnchorSize': 64,
            'RKMaximumStringLength': 4096,
            'RKMaximumPathLength': 1024,
            'RKMaximumPrefixLength': 8,
            'RKMaximumFolderPathLength': 768,
            'RKMaximumWaveformCount': 22,
            'RKMaximumFilterCount': 8,
            'RKRadarDescOffset': 256,
            'RKRadarDesc': 1072,
            'RKConfigV1': 1441,
            'RKConfig': 1024,
            'RKMaximumCommandLength': 512,
            'RKMaxFilterCount': 8,
            'RKPulseHeaderV1': 256,
            'RKPulseHeader': 384,
            'RKWaveFileGlobalHeader': 512
        }
        
        self.filename = ""
        
        self.header = {
            'preface': [],
            'buildNo': 6,
            'dataType': [],
            'desc': [],
            'config': [],
            'waveform': []
        }
        
        self.pulses: Union[np.memmap, NDArray]
        #-------------------------------------------------------------------------------------------
        
        
        #------Set Filename Property----------------------------------------------------------------
        if verbose:
            print(f"Filename: {filename}")
        self.filename = filename
        #-------------------------------------------------------------------------------------------
        
        
        #------Open file for partial reads----------------------------------------------------------
        file = open(self.filename, 'rb')
        #-------------------------------------------------------------------------------------------
        
        
        #------Get preface and build number---------------------------------------------------------
        self.header['preface'] = struct.unpack(f"{self.constants['RKName']}s", 
            file.read(self.constants['RKName']))[0].decode()\
            .replace('\x00', ' ').strip()
        self.header['buildNo'] = struct.unpack('I', file.read(4))[0]
        if verbose:
            print(f"preface = {self.header['preface']} "
                f"  buildNo = {self.header['buildNo']}")
        #-------------------------------------------------------------------------------------------
        
        
        #------If build 5 or higher, get dataType---------------------------------------------------
        if self.header['buildNo'] >= 5:
            self.header['dataType'] = struct.unpack('B', file.read(1))[0]
        #-------------------------------------------------------------------------------------------


        #------Radar Description--------------------------------------------------------------------
        if self.header['buildNo'] >= 2:
            if self.header['buildNo'] >= 6:
                offset = self.constants['RKRadarDescOffset']
            else:
                offset = self.constants['RKName'] + 4
            h = np.memmap(self.filename, mode='r', offset=offset, shape=(1,),
                dtype=np.dtype([
                    ('initFlags', 'uint32'),
                    ('pulseCapacity', 'uint32'),
                    ('pulseToRayRatio', 'uint16'),
                    ('doNotUse', 'uint16'),
                    ('healthNodeCount', 'uint32'),
                    ('healthBufferDepth', 'uint32'),
                    ('statusBufferDepth', 'uint32'),
                    ('configBufferDepth', 'uint32'),
                    ('positionBufferDepth', 'uint32'),
                    ('pulseBufferDepth', 'uint32'),
                    ('rayBufferDepth', 'uint32'),
                    ('productBufferDepth', 'uint32'),
                    ('controlCapacity', 'uint32'),
                    ('waveformCalibrationCapacity', 'uint32'),
                    ('healthNodeBufferSize', 'uint64'),
                    ('healthBufferSize', 'uint64'),
                    ('statusBufferSize', 'uint64'),
                    ('configBufferSize', 'uint64'),
                    ('positionBufferSize', 'uint64'),
                    ('pulseBufferSize', 'uint64'),
                    ('rayBufferSize', 'uint64'),
                    ('productBufferSize', 'uint64'),
                    ('pulseSmoothFactor', 'uint32'),
                    ('pulseTicsPerSecond', 'uint32'),
                    ('positionSmoothFactor', 'uint32'),
                    ('positionTicsPerSecond', 'uint32'),
                    ('positionLatency', 'f8'),
                    ('latitude', 'f8'),
                    ('longitude', 'f8'), 
                    ('heading', 'f4'),
                    ('radarHeight', 'f4'),
                    ('wavelength', 'f4'),
                    ('name_raw', 'uint8', (self.constants['RKName'],)),
                    ('filePrefix_raw', 'uint8', (self.constants['RKMaximumPrefixLength'],)),
                    ('dataPath_raw', 'uint8', (self.constants['RKMaximumFolderPathLength'],))
                ])
            )
        else:
            h = np.memmap(self.filename, mode='r',
                offset=self.constants['RKName'] + 4, shape=(1,),
                dtype=np.dtype([
                    ('initFlags', 'uint32'),
                    ('pulseCapacity', 'uint32'),
                    ('pulseToRayRatio', 'uint32'),
                    ('healthNodeCount', 'uint32'),
                    ('configBufferDepth', 'uint32'),
                    ('positionBufferDepth', 'uint32'),
                    ('pulseBufferDepth', 'uint32'),
                    ('rayBufferDepth', 'uint32'),
                    ('controlCount', 'uint32'),
                    ('latitude', 'f8'),
                    ('longitude', 'f8'), 
                    ('heading', 'f4'),
                    ('radarHeight', 'f4'),
                    ('wavelength', 'f4'),
                    ('name_raw', 'uint8', (self.constants['RKName'],)),
                    ('filePrefix_raw', 'uint8', (self.constants['RKMaximumPrefixLength'],)),
                    ('dataPath_raw', 'uint8', (self.constants['RKMaximumFolderPathLength'],))
                ])
            )
        self.header['desc'] = {field: h[0][field] for field in h[0].dtype.names}
        self.header['desc']['name'] = ''\
            .join(chr(num) for num in self.header['desc']['name_raw'])\
            .replace('\x00', ' ').strip()
        self.header['desc']['filePrefix'] = ''\
            .join(chr(num) for num in self.header['desc']['filePrefix_raw'])\
            .replace('\x00', ' ').strip()
        self.header['desc']['dataPath'] = ''\
            .join(chr(num) for num in self.header['desc']['dataPath_raw'])\
            .replace('\x00', ' ').strip()
        
        if not (posFilename is None):
            posOverrideFields = ["latitude", "longitude", "heading"]
            with open(posFilename, "r") as posFile:
                for posField in posOverrideFields:
                    line = posFile.readline()
                    if not line:
                        break
                    val = float(line.strip())
                    self.header['desc'][posField] = val
        #-------------------------------------------------------------------------------------------
        
        
        #------Get config and waveforms-------------------------------------------------------------
        #Build 8 config
        if self.header['buildNo'] >= 8:
            offset = self.constants['RKRadarDescOffset'] +\
                self.constants['RKRadarDesc']
            c = np.memmap(self.filename, mode='r', offset=offset, shape=(1,),
                dtype=np.dtype([
                    ('i', 'uint64'),
                    ('volumeIndex', 'uint64'),
                    ('sweepIndex', 'uint64'),
                    ('sweepElevation', 'f4'),
                    ('sweepAzimuth', 'f4'),
                    ('startMarker', 'uint32'),
                    ('prt', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('pw', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('pulseGateCount', 'uint32'),
                    ('pulseGateSize', 'f4'),
                    ('transitionGateCount', 'uint32'),
                    ('ringFilterGateCount', 'uint32'),
                    ('waveformId', 'uint32', (self.constants['RKMaxFilterCount'],)),
                    ('noise', 'f4', (2,)),
                    ('systemZCal', 'f4', (2,)),
                    ('systemDCal', 'f4'),
                    ('systemPCal', 'f4'),
                    ('ZCal', 'f4', (2,self.constants['RKMaxFilterCount'])),
                    ('DCal', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('PCal', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('SNRThreshold', 'f4'),
                    ('SQIThreshold', 'f4'),
                    ('waveformName', 'uint8', (self.constants['RKName'],)),
                    ('trash', 'uint64', (3,)),
                    ('momentMethod', 'uint8'),
                    ('userIntegerParameters', 'uint32', (8,)),
                    ('userFloatParameters', 'f4', (8,)),
                    ('vcpDefinition', 'uint8', (480,))
                ])
            )
            config = {field: c[0][field] for field in c[0].dtype.names}
            config['ZCal'] = config['ZCal'].T
            config['waveformName'] = ''\
                .join(chr(num) for num in config['waveformName'])\
                .replace('\x00', ' ').strip()
            config['vcpDefinition'] = ''\
                .join(chr(num) for num in config['vcpDefinition'])\
                .replace('\x00', ' ').strip()
            del config['trash']
            self.header['config'] = config
            
        
        #Build 6/7 config
        if self.header['buildNo'] == 6 or self.header['buildNo'] == 7:
            offset = self.constants['RKRadarDescOffset'] +\
                self.constants['RKRadarDesc']
            c = np.memmap(self.filename, mode='r', offset=offset, shape=(1,),
                dtype=np.dtype([
                    ('i', 'uint64'),
                    ('sweepElevation', 'f4'),
                    ('sweepAzimuth', 'f4'),
                    ('startMarker', 'uint32'),
                    ('prt', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('pw', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('pulseGateCount', 'uint32'),
                    ('pulseGateSize', 'f4'),
                    ('transitionGateCount', 'uint32'),
                    ('ringFilterGateCount', 'uint32'),
                    ('waveformId', 'uint32', (self.constants['RKMaxFilterCount'],)),
                    ('noise', 'f4', (2,)),
                    ('systemZCal', 'f4', (2,)),
                    ('systemDCal', 'f4'),
                    ('systemPCal', 'f4'),
                    ('ZCal', 'f4', (2,self.constants['RKMaxFilterCount'])),
                    ('DCal', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('PCal', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('SNRThreshold', 'f4'),
                    ('SQIThreshold', 'f4'),
                    ('waveformName', 'uint8', (self.constants['RKName'],)),
                    ('trash', 'uint64', (2,)),
                    ('momentMethod', 'uint8'),
                    ('vcpDefinition', 'uint8', (512,))
                ])
            )
            config = {field: c[0][field] for field in c[0].dtype.names}
            config['ZCal'] = config['ZCal'].T
            config['waveformName'] = ''\
                .join(chr(num) for num in config['waveformName'])\
                .replace('\x00', ' ').strip()
            config['vcpDefinition'] = ''\
                .join(chr(num) for num in config['vcpDefinition'])\
                .replace('\x00', ' ').strip()
            del config['trash']
            self.header['config'] = config
            
        #Build 6/7/8 Waveforms
        if self.header['buildNo'] >= 6 and self.header['buildNo'] <= 8:
            offset = self.constants['RKFileHeader']
            w = np.memmap(self.filename, mode='r', offset=offset, shape=(1,),
                dtype=np.dtype([
                    ('count', 'uint8'),
                    ('depth', 'uint32'),
                    ('type', 'uint32'),
                    ('name', 'uint8', (128,)),
                    ('fc', 'f8'),
                    ('fs', 'f8'),
                    ('filterCounts', 'uint8', (self.constants['RKMaximumWaveformCount'],))
                ])
            )
            self.header['waveform'] =\
                {field: w[0][field] for field in w[0].dtype.names}
            self.header['waveform']['filterCounts'] =\
                self.header['waveform']['filterCounts'][0:self.header['waveform']['count']]
            self.header['waveform']['name'] = ''\
                .join(chr(num) for num in self.header['waveform']['name'])\
                .replace('\x00', ' ').strip()
                
            offset += self.constants['RKWaveFileGlobalHeader']
            filters = []
            tones = []
            for i in range(self.header['waveform']['count']):
                tmp = []
                for j in range(self.header['waveform']['filterCounts'][i]):
                    w = np.memmap(self.filename, mode='r', offset=offset,
                        shape=(self.header['waveform']['filterCounts'][i],),
                        dtype=np.dtype([
                            ('name', 'uint32'),
                            ('origin', 'uint32'),
                            ('length', 'uint32'),
                            ('inputOrigin', 'uint32'),
                            ('outputOrigin', 'uint32'),
                            ('maxDataLength', 'uint32'),
                            ('subCarrierFrequency', 'f4'),
                            ('sensitivityGain', 'f4'),
                            ('filterGain', 'f4'),
                            ('fullScale', 'f4'),
                            ('lowerBoundFrequency', 'f4'),
                            ('upperBoundFrequency', 'f4'),
                            ('padding', 'uint8', (16,))
                        ])
                    )
                    for filter in w:
                        filter = {field: filter[field] for field in filter.dtype.names}
                        del filter['padding']
                        tmp.append(filter)
                    offset += self.constants['RKFilterAnchorSize']
                filters += tmp
                
                depth = self.header['waveform']['depth']
                w2 = np.memmap(self.filename, mode='r', offset=offset, shape=(1,),
                    dtype=np.dtype([
                        ('samples', 'f4', (depth,2)),
                        ('iSamples', 'int16', (depth,2))
                    ])
                )
                offset += 2 * depth * (4 + 2)
                x = w2[0]['samples']
                y = w2[0]['iSamples']
                gsamp = {
                    'samples': x[:,0] + 1j*x[:,1],
                    'iSamples': y[:,0] + 1j*y[:,1],  
                }
                tones += [gsamp]
            if len(filters) == 1:
                filters = filters[0]
            if len(tones) == 1:
                tones = tones[0]
            self.header['waveform']['filters'] = filters
            self.header['waveform']['tones'] = tones
            self.header['config']['pw'] =\
                self.header['config']['pw'][self.header['waveform']['filterCounts'][0]]
            self.header['config']['prt'] =\
                self.header['config']['prt'][self.header['config']['prt'] > 0]
            if len(self.header['config']['prt']) == 1:
                self.header['config']['prt'] = self.header['config']['prt'][0]
                
        #Build 5 Config and Waveforms
        elif self.header['buildNo'] == 5:
            offset = self.constants['RKName'] + 4 + self.constants['RKRadarDesc']
            c = np.memmap(self.filename, mode='r', offset=offset, shape=(1,),
                dtype=np.dtype([
                    ('i', 'uint64'),
                    ('sweepElevation', 'f4'),
                    ('sweepAzimuth', 'f4'),
                    ('startMarker', 'uint32'),
                    ('filterCount', 'uint8')
                ])
            )
            offset += 21
            c2 = np.memmap(self.filename, mode='r', offset=offset,
                shape=(self.constants['RKMaxFilterCount'],),
                dtype=np.dtype([
                    ('name', 'uint32'),
                    ('origin', 'uint32'),
                    ('length', 'uint32'),
                    ('inputOrigin', 'uint32'),
                    ('outputOrigin', 'uint32'),
                    ('maxDataLength', 'uint32'),
                    ('subCarrierFrequency', 'f4'),
                    ('sensitivityGain', 'f4'),
                    ('filterGain', 'f4'),
                    ('fullScale', 'f4'),
                    ('lowerBoundFrequency', 'f4'),
                    ('upperBoundFrequency', 'f4'),
                    ('padding', 'uint8', (16,))
                ])
            )
            offset += self.constants['RKMaxFilterCount'] * self.constants['RKFilterAnchorSize']
            c3 = np.memmap(self.filename, mode='r', offset=offset, shape=(1,),
                dtype = np.dtype([
                    ('prt', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('pw', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('pulseGateCount', 'uint32'),
                    ('pulseGateSize', 'f4'),
                    ('pulseRingFilterGateCount', 'uint32'),
                    ('waveformId', 'uint32', (self.constants['RKMaxFilterCount'],)),
                    ('noise', 'f4', (2,)),
                    ('systemZCal', 'f4', (2,)),
                    ('systemDCal', 'f4'),
                    ('systemPCal', 'f4'),
                    ('ZCal', 'f4', (2,self.constants['RKMaxFilterCount'])),
                    ('DCal', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('PCal', 'f4'),
                    ('SNRThreshold', 'f4'),
                    ('SQIThreshold', 'f4'),
                    ('waveform_raw', 'uint8', (self.constants['RKName'],)),
                    ('vcpDefinition_raw', 'uint8', (self.constants['RKName'],))
                ])
            )
            self.header['config'] = {field: c[0][field] for field in c[0].dtype.names}
            self.header['config']['filterAnchors'] = []
            for filterAnchor in c2:
                filterAnchor = {field: filterAnchor[field] for field in filterAnchor.dtype.names}
                self.header['config']['filterAnchors'] += [filterAnchor]
            if len(self.header['config']['filterAnchors']) == 1:
                self.header['config']['filterAnchors'] = self.header['config']['filterAnchors'][0]
            for field in c3[0].dtype.names[0:-2]:
                self.header['config'][field] = c3[0][field]
            self.header['config']['ZCal'] = self.header['config']['ZCal'].T
            self.header['config']['waveform'] = ''\
                .join(chr(num) for num in c3[0]['waveform_raw'])\
                .replace('\x00', ' ').strip()
            self.header['config']['vcpDefinition'] = ''\
                .join(chr(num) for num in c3[0]['vcpDefinition_raw'])\
                .replace('\x00', ' ').strip()
            offset = self.constants['RKName'] + 4 +\
                self.constants['RKRadarDesc'] + self.constants['RKConfigV1']
            file.seek(offset, 0)
            self.header['dataType'] = struct.unpack('B', file.read(1))[0]
            offset = self.constants['RKFileHeader']
            self.header['config']['pw'] =\
                self.header['config']['pw'](self.header['config']['filterCount'])
            self.header['config']['prt'] =\
                self.header['config']['prt'][self.header['config']['prt'] > 0]
            if len(self.header['config']['prt']) == 1:
                self.header['config']['prt'] = self.header['config']['prt'][0]
                
        #Build 2-4 Config and Waveforms
        elif self.header['buildNo'] >= 2 and self.header['buildNo'] < 5:
            offset = self.constants['RKName'] + 4 + self.constants['RKRadarDesc']
            c = np.memmap(self.filename, mode='r', offset=offset, shape=(1,),
                dtype = np.dtype([
                    ('i', 'uint64'),
                    ('sweepElevation', 'f4'),
                    ('sweepAzimuth', 'f4'),
                    ('startMarker', 'uint32'),
                    ('filterCount', 'uint8'),
                ])
            )
            offset += 21
            c2 = np.memmap(self.filename, mode='r', offset=offset,
                shape=(self.constants['RKMaxFilterCount'],),
                dtype=np.dtype([
                    ('name', 'uint32'),
                    ('origin', 'uint32'),
                    ('length', 'uint32'),
                    ('inputOrigin', 'uint32'),
                    ('outputOrigin', 'uint32'),
                    ('maxDataLength', 'uint32'),
                    ('subCarrierFrequency', 'f4'),
                    ('sensitivityGain', 'f4'),
                    ('filterGain', 'f4'),
                    ('fullScale', 'f4'),
                    ('lowerBoundFrequency', 'f4'),
                    ('upperBoundFrequency', 'f4'),
                    ('padding', 'uint8', (16,))
                ])
            )
            offset += self.constants['RKMaxFilterCount'] * self.constants['RKFilterAnchorSize']
            c3 = np.memmap(self.filename, mode='r', offset=offset, shape=(1,),
                dtype = np.dtype([
                    ('prt', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('pw', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('pulseGateCount', 'uint32'),
                    ('pulseGateSize', 'f4'),
                    ('pulseRingFilterGateCount', 'uint32'),
                    ('waveformId', 'uint32', (self.constants['RKMaxFilterCount'],)),
                    ('noise', 'f4', (2,)),
                    ('systemZCal', 'f4', (2,)),
                    ('systemDCal', 'f4'),
                    ('systemPCal', 'f4'),
                    ('ZCal', 'f4', (2,self.constants['RKMaxFilterCount'])),
                    ('DCal', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('PCal', 'f4'),
                    ('SNRThreshold', 'f4'),
                    ('waveform_raw', 'uint8', (self.constants['RKName'],)),
                    ('vcpDefinition_raw', 'uint8', (self.constants['RKName'],))
                ])
            )
            self.header['config'] = {field: c[0][field] for field in c[0].dtype.names}
            self.header['config']['filterAnchors'] = []
            for filterAnchor in c2:
                filterAnchor = {field: filterAnchor[field] for field in filterAnchor.dtype.names}
                self.header['config']['filterAnchors'] += [filterAnchor]
            if len(self.header['config']['filterAnchors']) == 1:
                self.header['config']['filterAnchors'] = self.header['config']['filterAnchors'][0]
            for field in c3[0].dtype.names[0:-2]:
                self.header['config'][field] = c3[0][field]
            self.header['config']['ZCal'] = self.header['config']['ZCal'].T
            self.header['config']['waveformName'] = ''\
                .join(chr(num) for num in c3[0]['waveform_raw'])\
                .replace('\x00', ' ').strip()
            self.header['config']['vcpDefinition'] = ''\
                .join(chr(num) for num in c3[0]['vcpDefinition_raw'])\
                .replace('\x00', ' ').strip()
            self.header["dataType"] = 1
            offset = self.constants['RKFileHeader']
            
        #Build 1 Config and Waveforms
        else:
            c = np.memmap(self.filename, mode='r',
                offset=self.constants['RKName'] + 4 + self.constants['RKRadarDesc'], shape=(1,),
                dtype = np.dtype([
                    ('i', 'uint64'),
                    ('pw', 'uint32', (self.constants['RKMaxFilterCount'],)),
                    ('prf', 'uint32', (self.constants['RKMaxFilterCount'],)),
                    ('gateCount', 'uint32', (self.constants['RKMaxFilterCount'],)),
                    ('waveformId', 'uint32', (self.constants['RKMaxFilterCount'],)),
                    ('noise', 'f4', (2,)),
                    ('ZCal', 'f4', (2,self.constants['RKMaxFilterCount'])),
                    ('DCal', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('PCal', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('SNRThreshold', 'f4'),
                    ('sweepElevation', 'f4'),
                    ('sweepAzimuth', 'f4'),
                    ('startMarker', 'uint32'),
                    ('waveform_name_raw', 'uint8', (self.constants['RKName'],)),
                    ('vcpDefinition_raw', 'uint8', (self.constants['RKName'],))
                ])
            )
            self.header['config'] = {field: c[0][field] for field in c[0].dtype.names}
            self.header['config']['waveformName'] = ''\
                .join(chr(num) for num in self.header['config']['waveform_raw'])\
                .replace('\x00', ' ').strip()
            self.header['config']['vcpDefinition'] = ''\
                .join(chr(num) for num in self.header['config']['vcpDefinition_raw'])\
                .replace('\x00', ' ').strip()
            self.header['dataType'] = 1
            offset = self.constants['RKFileHeader']
        #-------------------------------------------------------------------------------------------
        
        
        #------Read Pulses--------------------------------------------------------------------------
        #Partially read first pulse
        file.seek(offset+28, 0)
        pulseStart = struct.unpack('III', file.read(12))
        capacity = pulseStart[0]
        gateCount = pulseStart[1]
        downSampledGateCount = pulseStart[2]
        pulseDataSize = file.seek(0, 2) - offset
        if verbose:
            print(f"Pulse data size: {pulseDataSize}")
        file.close()
        if verbose:
            print(f"gateCount = {gateCount}   capacity = {capacity} "
                f"  downSampledGateCount = {downSampledGateCount}")
        
        #Dimensions
        if verbose:
            print(f"data offset = {offset}")
            if maxPulse != None:
                print(f"Reading {maxPulse} pulses ...")
            else:
                print("Reading pulses ...")
        
        #Read all pulses
        if self.header['buildNo'] == 7 or self.header['buildNo'] == 8:
            if self.header['dataType'] == 1:
                #Raw I/Q straight from the transceiver
                IQDtype = np.dtype([
                    ('i', 'uint64'),
                    ('n', 'uint64'),
                    ('t', 'uint64'),
                    ('s', 'uint32'),
                    ('capacity', 'uint32'),
                    ('gateCount', 'uint32'),
                    ('downSampledGateCount', 'uint32'),
                    ('marker', 'uint32'),
                    ('pulseWidthSampleCount', 'uint32'),
                    ('time_tv_sec', 'uint64'),
                    ('time_tv_usec', 'uint64'),
                    ('timeDouble', 'f8'),
                    ('rawAzimuth', 'uint8', (4,)),
                    ('rawElevation', 'uint8', (4,)),
                    ('configIndex', 'uint16'),
                    ('configSubIndex', 'uint16'),
                    ('positionIndex', 'uint32'),
                    ('gateSizeMeters', 'f4'),
                    ('elevationDegrees', 'f4'),
                    ('azimuthDegrees', 'f4'),
                    ('elevationVelocityDegreesPerSecond', 'f4'),
                    ('azimuthVelocityDegreesPerSecond', 'f4'),
                    ('padding', 'uint8', (84,)),
                    ('iq', 'int16', (2, gateCount, 2))
                ])
                IQHdtype = np.dtype([
                    ('i', 'uint64'),
                    ('n', 'uint64'),
                    ('t', 'uint64'),
                    ('s', 'uint32'),
                    ('capacity', 'uint32'),
                    ('gateCount', 'uint32'),
                    ('downSampledGateCount', 'uint32'),
                    ('marker', 'uint32'),
                    ('pulseWidthSampleCount', 'uint32'),
                    ('time_tv_sec', 'uint64'),
                    ('time_tv_usec', 'uint64'),
                    ('timeDouble', 'f8'),
                    ('rawAzimuth', 'uint8', (4,)),
                    ('rawElevation', 'uint8', (4,)),
                    ('configIndex', 'uint16'),
                    ('configSubIndex', 'uint16'),
                    ('positionIndex', 'uint32'),
                    ('gateSizeMeters', 'f4'),
                    ('elevationDegrees', 'f4'),
                    ('azimuthDegrees', 'f4'),
                    ('elevationVelocityDegreesPerSecond', 'f4'),
                    ('azimuthVelocityDegreesPerSecond', 'f4'),
                    ('padding', 'uint8', (84,)),
                ])
                numPulses = pulseDataSize // IQDtype.itemsize
                if verbose:
                    print(f"Number of pulses: {numPulses}")
                m = np.memmap(self.filename, mode='r', offset=offset, 
                    shape=(numPulses,) if maxPulse == None else (maxPulse,), dtype = IQDtype
                )
            else:
                #Compressed I/Q
                IQDtype = np.dtype([
                    ('i', 'uint64'),
                    ('n', 'uint64'),
                    ('t', 'uint64'),
                    ('s', 'uint32'),
                    ('capacity', 'uint32'),
                    ('gateCount', 'uint32'),
                    ('downSampledGateCount', 'uint32'),
                    ('marker', 'uint32'),
                    ('pulseWidthSampleCount', 'uint32'),
                    ('time_tv_sec', 'uint64'),
                    ('time_tv_usec', 'uint64'),
                    ('timeDouble', 'f8'),
                    ('rawAzimuth', 'uint8', (4,)),
                    ('rawElevation', 'uint8', (4,)),
                    ('configIndex', 'uint16'),
                    ('configSubIndex', 'uint16'),
                    ('positionIndex', 'uint32'),
                    ('gateSizeMeters', 'f4'),
                    ('elevationDegrees', 'f4'),
                    ('azimuthDegrees', 'f4'),
                    ('elevationVelocityDegreesPerSecond', 'f4'),
                    ('azimuthVelocityDegreesPerSecond', 'f4'),
                    ('padding', 'uint8', (84,)),
                    ('iq', 'f4', (2, downSampledGateCount, 2))
                ])
                IQHdtype = np.dtype([
                    ('i', 'uint64'),
                    ('n', 'uint64'),
                    ('t', 'uint64'),
                    ('s', 'uint32'),
                    ('capacity', 'uint32'),
                    ('gateCount', 'uint32'),
                    ('downSampledGateCount', 'uint32'),
                    ('marker', 'uint32'),
                    ('pulseWidthSampleCount', 'uint32'),
                    ('time_tv_sec', 'uint64'),
                    ('time_tv_usec', 'uint64'),
                    ('timeDouble', 'f8'),
                    ('rawAzimuth', 'uint8', (4,)),
                    ('rawElevation', 'uint8', (4,)),
                    ('configIndex', 'uint16'),
                    ('configSubIndex', 'uint16'),
                    ('positionIndex', 'uint32'),
                    ('gateSizeMeters', 'f4'),
                    ('elevationDegrees', 'f4'),
                    ('azimuthDegrees', 'f4'),
                    ('elevationVelocityDegreesPerSecond', 'f4'),
                    ('azimuthVelocityDegreesPerSecond', 'f4'),
                    ('padding', 'uint8', (84,)),
                ])
                numPulses = pulseDataSize // IQDtype.itemsize
                if verbose:
                    print(f"Number of pulses: {numPulses}")
                m = np.memmap(self.filename, mode='r', offset=offset, 
                    shape=(numPulses,) if maxPulse == None else (maxPulse,), dtype = IQDtype
                )
        else:
            if self.header['dataType'] == 1:
                #Raw I/Q straight from the transceiver
                IQDtype = np.dtype([
                    ('i', 'uint64'),
                    ('n', 'uint64'),
                    ('t', 'uint64'),
                    ('s', 'uint32'),
                    ('capacity', 'uint32'),
                    ('gateCount', 'uint32'),
                    ('downSampledGateCount', 'uint32'),
                    ('marker', 'uint32'),
                    ('pulseWidthSampleCount', 'uint32'),
                    ('time_tv_sec', 'uint64'),
                    ('time_tv_usec', 'uint64'),
                    ('timeDouble', 'f8'),
                    ('rawAzimuth', 'uint8', (4,)),
                    ('rawElevation', 'uint8', (4,)),
                    ('configIndex', 'uint16'),
                    ('configSubIndex', 'uint16'),
                    ('azimuthBinIndex', 'uint16'),
                    ('gateSizeMeters', 'f4'),
                    ('elevationDegrees', 'f4'),
                    ('azimuthDegrees', 'f4'),
                    ('elevationVelocityDegreesPerSecond', 'f4'),
                    ('azimuthVelocityDegreesPerSecond', 'f4'),
                    ('iq', 'int16', (2, gateCount, 2))
                ])
                IQHdtype = np.dtype([
                    ('i', 'uint64'),
                    ('n', 'uint64'),
                    ('t', 'uint64'),
                    ('s', 'uint32'),
                    ('capacity', 'uint32'),
                    ('gateCount', 'uint32'),
                    ('downSampledGateCount', 'uint32'),
                    ('marker', 'uint32'),
                    ('pulseWidthSampleCount', 'uint32'),
                    ('time_tv_sec', 'uint64'),
                    ('time_tv_usec', 'uint64'),
                    ('timeDouble', 'f8'),
                    ('rawAzimuth', 'uint8', (4,)),
                    ('rawElevation', 'uint8', (4,)),
                    ('configIndex', 'uint16'),
                    ('configSubIndex', 'uint16'),
                    ('azimuthBinIndex', 'uint16'),
                    ('gateSizeMeters', 'f4'),
                    ('elevationDegrees', 'f4'),
                    ('azimuthDegrees', 'f4'),
                    ('elevationVelocityDegreesPerSecond', 'f4'),
                    ('azimuthVelocityDegreesPerSecond', 'f4'),
                ])
                numPulses = pulseDataSize // IQDtype.itemsize
                if verbose:
                    print(f"Number of pulses: {numPulses}")
                m = np.memmap(self.filename, mode='r', offset=offset, 
                    shape=(numPulses,) if maxPulse == None else (maxPulse,), dtype = IQDtype
                )
            else:
                #Compressed I/Q (non build 7)
                IQDtype = np.dtype([
                    ('i', 'uint64'),
                    ('n', 'uint64'),
                    ('t', 'uint64'),
                    ('s', 'uint32'),
                    ('capacity', 'uint32'),
                    ('gateCount', 'uint32'),
                    ('downSampledGateCount', 'uint32'),
                    ('marker', 'uint32'),
                    ('pulseWidthSampleCount', 'uint32'),
                    ('time_tv_sec', 'uint64'),
                    ('time_tv_usec', 'uint64'),
                    ('timeDouble', 'f8'),
                    ('rawAzimuth', 'uint8', (4,)),
                    ('rawElevation', 'uint8', (4,)),
                    ('configIndex', 'uint16'),
                    ('configSubIndex', 'uint16'),
                    ('azimuthBinIndex', 'uint16'),
                    ('gateSizeMeters', 'f4'),
                    ('elevationDegrees', 'f4'),
                    ('azimuthDegrees', 'f4'),
                    ('elevationVelocityDegreesPerSecond', 'f4'),
                    ('azimuthVelocityDegreesPerSecond', 'f4'),
                    ('iq', 'f4', (2,downSampledGateCount,2)),
                ])
                IQHdtype = np.dtype([
                    ('i', 'uint64'),
                    ('n', 'uint64'),
                    ('t', 'uint64'),
                    ('s', 'uint32'),
                    ('capacity', 'uint32'),
                    ('gateCount', 'uint32'),
                    ('downSampledGateCount', 'uint32'),
                    ('marker', 'uint32'),
                    ('pulseWidthSampleCount', 'uint32'),
                    ('time_tv_sec', 'uint64'),
                    ('time_tv_usec', 'uint64'),
                    ('timeDouble', 'f8'),
                    ('rawAzimuth', 'uint8', (4,)),
                    ('rawElevation', 'uint8', (4,)),
                    ('configIndex', 'uint16'),
                    ('configSubIndex', 'uint16'),
                    ('azimuthBinIndex', 'uint16'),
                    ('gateSizeMeters', 'f4'),
                    ('elevationDegrees', 'f4'),
                    ('azimuthDegrees', 'f4'),
                    ('elevationVelocityDegreesPerSecond', 'f4'),
                    ('azimuthVelocityDegreesPerSecond', 'f4'),
                ])
                numPulses = pulseDataSize // IQDtype.itemsize
                if verbose:
                    print(f"Number of pulses: {numPulses}")
                m = np.memmap(self.filename, mode='r', offset=offset, 
                    shape=(numPulses,) if maxPulse == None else (maxPulse,), dtype = IQDtype
                )
        #OG is time, h/v, range, iq
        self.pulses = m
        self.pulsesDtype = IQDtype

        def makeLazyMemmapArray(path, dtype, shape, chunks, offset=0):
            # This function will be called by workers to get a chunk
            def getChunk(path, dtype, offset, shape, slices):
                mm = np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=shape)
                return np.asarray(mm[slices])
            
            # Build the dask array by creating delayed chunks
            chunk_slices = da.core.slices_from_chunks(da.core.normalize_chunks(chunks, shape))
            
            delayed_chunks = [
                delayed(getChunk)(path, dtype, offset, shape, slc)
                for slc in chunk_slices
            ]
            
            arrays = [
                da.from_delayed(
                    dc,
                    shape=tuple(s.stop - s.start for s in slc),
                    dtype=dtype,
                    meta=np.array([], dtype=dtype)
                )
                for dc, slc in zip(delayed_chunks, chunk_slices)
            ]
            
            return da.concatenate(arrays)
        
        self.daskPulses = makeLazyMemmapArray(self.filename, IQDtype, (numPulses,), 3000, offset)

        # #Get headers with direct OS read - slow so nvm
        # self.pulseHeaders = np.empty(numPulses, dtype=IQHdtype)
        # fd = os.open(self.filename, os.O_RDONLY)
        # hSize = IQHdtype.itemsize
        # skipSize = IQDtype.itemsize - hSize
        # for i in range(numPulses):
        #     buf = os.read(fd, hSize)
        #     self.pulseHeaders[i] = np.frombuffer(buf, dtype=IQHdtype)[0]
        #     os.lseek(fd, skipSize, os.SEEK_CUR)
        # os.close(fd)
        #-------------------------------------------------------------------------------------------
        
        
        #------Set dataType as string and see if waveform is recorded-------------------------------
        if self.header['dataType'] == 1:
            dt = 'raw'
        elif self.header['dataType'] == 2:
            dt = 'compressed'
        else:
            dt = 'unknown'
        self.header['dataType'] = dt
        
        if (not ('waveform' in self.header))\
            or (type(self.header['waveform']) == str and len(self.header['waveform']) == 0):
            self.header['waveform'] = 'not recorded'
        
    def pulseToDict(self, idx: int):
        return {field: self.pulses[idx][field] for field in self.pulses[idx].dtype.names}
    
    def elArray(self):
        return self.pulses['elevationDegrees']
    
    def azArray(self):
        return self.pulses['azimuthDegrees']
    
def readIQ(path: str | Path, copy: bool = True, **kwargs) -> core.IQ:
    if "tzStr" not in kwargs:
        raise TypeError("Needed 'tzStr' kwarg. It's probably 'US/Central'.")
    else:
        tzStr = kwargs["tzStr"]

    if "instrumentMame" not in kwargs:
        instrName = "RaXPol"
    else:
        instrName = kwargs["instrumentName"]

    if "institution" not in kwargs:
        institution = "The University of Oklahoma"
    else:
        institution = kwargs["institution"]

    ret = core.IQ()
    rkc = rkcfile(path, verbose=False)

    if "correctedLat" in kwargs:
        rkc.header['desc']['latitude'] = kwargs["correctedLat"]
    if "correctedLon" in kwargs:
        rkc.header['desc']['longitude'] = kwargs["correctedLon"]

    if copy:
        iq = np.ascontiguousarray(rkc.pulses["iq"].transpose((1, 2, 0, 3)))
    else:
        iq = rkc.daskPulses["iq"].transpose((1, 2, 0, 3))

    timeDoubleArr = (
        rkc.daskPulses['time_tv_sec'].compute() +
        rkc.daskPulses['time_tv_usec'].compute()/1000000.
    ).astype(np.float64)

    if rkc.header['buildNo'] >= 4:
        if rkc.header['dataType'] == 'raw':
            dr = rkc.header['config']['pulseGateSize']
        elif rkc.header['dataType'] == 'compressed':
            dr = rkc.header['desc']['pulseToRayRatio'] * rkc.header['config']['pulseGateSize']
        else:
            print("Inconsistency detected. This should not happen.")
            dr = 30.
    else:
        dr = 30.

    pulse1 = rkc.pulseToDict(0)

    #remember iq is h/v, range, iq
    ng = pulse1["iq"].shape[1]
    nRay = len(rkc.pulses)
    rr = (dr/2) + (np.arange(0,ng) * dr)

    pulseWidthArr = np.tile(rkc.header['config']['pw'], (nRay,)).astype(np.float32)
    prtArr = np.tile(rkc.header['config']['prt'], (nRay,)).astype(np.float32)
    wavelengthArr = np.tile(rkc.header['desc']['wavelength'], (nRay,)).astype(np.float32)

    ret.setInstrument(
        name = instrName,
        institution = institution,
        source = rkc.header["preface"]
    )
    ret.setVolume(0)
    ret.setTime(timeDoubleArr, tzStr)
    ret.setSweep(0)
    ret.setRange(rr.astype(np.float32), True)
    ret.setPosition(
        rkc.header['desc']['latitude'], 
        rkc.header['desc']['longitude'], 
        rkc.header["desc"]["radarHeight"]
    )
    ret.setScanningStrategy("ppi", rkc.header["config"]["sweepElevation"])
    ret.setAzimuth(rkc.azArray().astype(np.float32))
    ret.setElevation(rkc.elArray().astype(np.float32))
    ret.setPulseWidthSeconds(pulseWidthArr)
    ret.setPrtSeconds(prtArr)
    ret.setWavelengthMeters(wavelengthArr)
    ret.setSourceFile(rkc.filename)

    ret.setPol(2)
    ret.setNoisedB(rkc.header["config"]["noise"])
    ret.setCal(
        rkc.header["config"]["systemZCal"], 
        rkc.header["config"]["systemDCal"],
        rkc.header["config"]["systemPCal"]
    )

    ret.addDataField(
        'iq',
        data = iq,
        dims = ["pol", "range", "time", "iqdim"],
        attrs = {
            "units": "arbitrary_raw_transciever_units",
            "long_name": "raw_in-phase/quadrature_returns",
        },
        encoding = {
            "dtype": "float32",
            "_FillValue": core._FILL_VALUES["float32"]
        }
    )

    if not ret.validateSelf():
        raise RuntimeError("Structure wasn't able to validate. Check bools.")

    return ret