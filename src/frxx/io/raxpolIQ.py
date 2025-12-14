import numpy as np
from .radarkitIQ import rkcfile
from .raxpolCf import raxpolCf
from pathlib import Path
import pandas as pd
import multiprocessing

def _getRootRelativeFiles(root, caseName):
    caseDir = root/"cases"/caseName
    with open(caseDir/"allFiles.txt", 'r') as f:
        dir = f.readline().strip()
        files = f.read()
        files = files.split('\n')
        files = [file for file in files if file.strip()]
    return (dir, files)

def _toVecInDegreeRange(val, low, high):
    low = low if low > 0 else low + 360
    high = high if high < 360 else high - 360
    return (val > low and val < high
            if low < high
            else val < (high + 360)
                if val > low
                else (val + 360) < (high + 360))
    
def _processRayShared(sharedRh, sharedRv, Rshape, sharedC, Cshape, Xh, Xv, iRay):
    Rh = np.frombuffer(sharedRh.get_obj(), dtype=np.complex128).reshape(Rshape)
    Rv = np.frombuffer(sharedRv.get_obj(), dtype=np.complex128).reshape(Rshape)
    C = np.frombuffer(sharedC.get_obj(), dtype=np.complex128).reshape(Cshape)
    
    N = Rshape[2]-1
    
    for i in range(N+1):
        Rh[:,iRay,i] = np.mean(Xh[:,i:] * np.conjugate(Xh[:,:(None if i == 0 else -i)]),axis=1)
        Rv[:,iRay,i] = np.mean(Xv[:,i:] * np.conjugate(Xv[:,:(None if i == 0 else -i)]),axis=1)
    
    for i in range(-N, N+1):
        if i < 0:
            C[:,iRay,i+N] = np.mean(Xh[:,:i] * np.conjugate(Xv[:,-i:]),axis=1)
        else:
            C[:,iRay,i+N] = np.mean(Xh[:,i:] * np.conjugate(Xv[:,:(None if i == 0 else -i)]),axis=1)
            
def _processRaySequential(Rh, Rv, Rshape, C, Cshape, Xh, Xv, iRay):
    N = Rshape[2]-1
    
    for i in range(N+1):
        Rh[:,iRay,i] = np.mean(Xh[:,i:] * np.conjugate(Xh[:,:(None if i == 0 else -i)]),axis=1)
        Rv[:,iRay,i] = np.mean(Xv[:,i:] * np.conjugate(Xv[:,:(None if i == 0 else -i)]),axis=1)
    
    for i in range(-N, N+1):
        if i < 0:
            C[:,iRay,i+N] = np.mean(Xh[:,:i] * np.conjugate(Xv[:,-i:]),axis=1)
        else:
            C[:,iRay,i+N] = np.mean(Xh[:,i:] * np.conjugate(Xv[:,:(None if i == 0 else -i)]),axis=1)
            
def _averageByGstep(nparr, gstep: int):
    m, n, o = nparr.shape
    fullGroups = m//gstep
    if fullGroups > 0:
        mainPart = nparr[:fullGroups*gstep,:,:].reshape((fullGroups, gstep, n, o)).mean(axis=1)
    else:
        mainPart = np.array([]).reshape((0, n, o))
        
    if m % gstep != 0:
        remainder = nparr[fullGroups*gstep:,:,:].mean(axis=0)
        result = np.concatenate((mainPart, remainder), axis=0)
    else:
        result = mainPart
        
    return result

class raxpolrkc(rkcfile):
    def staticGetExistingCfrad(root, caseName: str, fileNo: int, 
                               azimuthBeamWidthDeg: float = 1, beamOverlapDeg: float = 0):
        root = Path(root)
        dir, files = _getRootRelativeFiles(root, caseName)
        if fileNo >= len(files) or fileNo < 0:
            raise IndexError(f"Only {len(files)} files in this case. "
                             f"Choose a fileNo between 0 and {len(files)-1}.")
        
        filename = root/dir/files[fileNo]
        casePath = root/"cases"/caseName
        cfFilename = Path("cfrad." + Path(filename).stem + '.nc')
        outDir = casePath/"out"/("s"+str(azimuthBeamWidthDeg)+"o"+str(beamOverlapDeg))
        if not outDir.exists():
            raise FileNotFoundError(f"Directory {outDir} does not exist.")
        if not (outDir/cfFilename).exists():
            raise FileNotFoundError(f"File {outDir/cfFilename} does not exist. Raw IQ data likely marked bad.")

        return outDir/cfFilename
    
    def staticCfradNum2RkcNum(root, caseName: str, fileNo: int, 
                              azimuthBeamWidthDeg: float = 1, beamOverlapDeg: float = 0):
        root = Path(root)
        casePath = root/"cases"/caseName
        outDir = casePath/"out"/("s"+str(azimuthBeamWidthDeg)+"o"+str(beamOverlapDeg))
        if not outDir.exists():
            raise FileNotFoundError(f"Directory {outDir} does not exist.")
        cfFilenames = sorted(Path(outDir).iterdir(), key=lambda x: x.name)
        if fileNo >= len(cfFilenames) or fileNo < 0:
            raise IndexError(f"Only {len(cfFilenames)} files in this case. "
                             f"Choose a fileNo between 0 and {len(cfFilenames)-1}.")
        cfFilename = cfFilenames[fileNo]

        _, rkcfiles = _getRootRelativeFiles(root, caseName)
        filename = str(cfFilename.stem)[6:] + '.rkc'
        
        return rkcfiles.index(filename)

    def __init__(self, root, caseName: str, fileNo: int, maxPulse: int = None):
        root = Path(root)
        dir, files = _getRootRelativeFiles(root, caseName)
        if fileNo >= len(files) or fileNo < 0:
            raise IndexError(f"Only {len(files)} files in this case. "
                             f"Choose a fileNo between 0 and {len(files)-1}.")
            
        self.casePath = root/"cases"/caseName
        self.dataFiltered = False
        
        if (self.casePath/"correctedLatLonHead.txt").exists():
            posFilename = self.casePath/"correctedLatLonHead.txt"
        else:
            posFilename = None
        
        super().__init__(root/dir/files[fileNo], maxPulse, posFilename)

    def getExistingCfrad(self, azimuthBeamWidthDeg: float = 1, beamOverlapDeg: float = 0):
        if not self.dataFiltered:
            raise RuntimeError("Must run filterBadData method first.")
        cfFilename = Path("cfrad." + Path(self.filename).stem + '.nc')
        outDir = self.casePath/"out"/("s"+str(azimuthBeamWidthDeg)+"o"+str(beamOverlapDeg))
        if not outDir.exists():
            raise FileNotFoundError(f"Directory {outDir} does not exist.")
        
        return outDir/cfFilename
        
    def filterBadData(self):
        if self.dataFiltered:
            return True
        goodData = pd.read_csv(self.casePath/"goodData.csv")
        goodData = goodData.set_index(goodData.columns[0])
        if not (goodData.loc[Path(self.filename).name, 'good']):
            print("Data flagged as not good, skipping.")
            return False

        startPulse = goodData.loc[Path(self.filename).name, 'startPulse']
        endPulse = goodData.loc[Path(self.filename).name, 'endPulse']
        goodPulses = np.tile(False, (len(self.pulses),))
        goodPulses[startPulse:(endPulse+1)] = True
        
        badPulses = pd.read_csv(self.casePath/"badPulseSwaths.csv")
        badPulses = badPulses[badPulses['filename'] == Path(self.filename).name]
        for _, row in badPulses.iterrows():
            currDeleted = [row['startPulse'], row['endPulse']]
            goodPulses[currDeleted[0]:currDeleted[1]+1] = False
        
        self.pulses = self.pulses[goodPulses]
        
        self.dataFiltered = True
        
        return True
        
    def ppiToCfrad(self, azimuthBeamWidthDeg: float = 1,
                   beamOverlapDeg: float = 0, useGoodData: bool = True):
        if (not (self.casePath/"goodData.csv").exists()) and useGoodData:
            raise FileNotFoundError("useGoodData was specified as true,"
                                    " but no goodData.csv exists in case directory")
        if (not (self.casePath/"badPulseSwaths.csv").exists()) and useGoodData:
            raise FileNotFoundError("useGoodData was specified as true,"
                                    " but no badPulseSwaths.csv exists in case directory")
        if self.header['waveform']['name'][0] == 'h':
            print('Frequency hopping detected, skipping.')
            return
        
        if useGoodData:
            if not self.filterBadData():
                return

        pulses = self.pulses
            
        az = pulses['azimuthDegrees']
        el = pulses['elevationDegrees']
        
        pulses = pulses['iq'].transpose((1,2,3,0))
        pulses = pulses[0,:,:,:] + 1.j*pulses[1,:,:,:]
        pulses = pulses.transpose((0,2,1)).astype(np.complex128)

        c = 299792458.0
        
        ng = pulses.shape[0]
        ns = pulses.shape[1]
        gstep = 1
        
        if self.header['buildNo'] >= 4:
            if self.header['dataType'] == 'raw':
                dr = self.header['config']['pulseGateSize']
                dt = dr * 2 / (c/1e6)
            elif self.header['dataType'] == 'compressed':
                dr = self.header['desc']['pulseToRayRatio'] * self.header['config']['pulseGateSize']
                dt = dr * 2 / (c/1e6)
            else:
                print("Inconsistency detected. This should not happen.")
                dr = 30.
                dt = 1./50
        else:
            dr = 30.
            dt = 1./50
        
        rr = (dr/2) + (np.arange(0,ng,gstep) * dr)
        rr_km = rr/1000
        tt = (dt/2) + (np.arange(0,ng,gstep) * dt)
        
        if (len(np.unique(np.rint(el))) > len(np.unique(np.rint(az))))\
            and (len(np.unique(np.rint(el))) > 5):
            
            print("Elevation varies by more than 5 degrees. This might be an RHI. Skipping.")
            return
   
        censordB = 4
        azSpacing = azimuthBeamWidthDeg
        azSwath = azimuthBeamWidthDeg+(2*beamOverlapDeg)
        
        azDiscrete = np.rint(az / azSpacing) * azSpacing
        azDiscrete[azDiscrete == 360] = 0
        azUnique = [azDiscrete[0]]
        currAz = azDiscrete[0]
        for azimuth in azDiscrete:
            if not azimuth == currAz:
                azUnique.append(azimuth)
                currAz = azimuth
        azUnique = np.array(azUnique, dtype=np.float32)
        
        repAz = np.tile(az, (len(azUnique),1))
        azTranspose = np.array([azUnique]).T
        
        isInDegreeRange = np.vectorize(_toVecInDegreeRange)
        
        azBool = isInDegreeRange(repAz, azTranspose-0.5*azSwath, azTranspose+0.5*azSwath)
        pulseBoundaries = []
        for swath in azBool:
            pulseBoundaries.append([np.where(swath)[0][0], np.where(swath)[0][-1]])
        pulseBoundaries = np.array(pulseBoundaries, dtype=np.int32)
        middlePulses = np.rint(np.mean(pulseBoundaries, axis=1)).astype(np.int32)
        
        nRay = len(azUnique)
        
        timeDoubleArr =\
            (self.pulses[middlePulses]['time_tv_sec'] +
             self.pulses[middlePulses]['time_tv_usec']/1000000).astype(np.float64)
            
        elevations = (self.pulses[middlePulses]['elevationDegrees']).astype(np.float32)
        
        if (self.casePath/"pulseTimeZone.txt").exists():
            with open(self.casePath/"pulseTimeZone.txt", "r") as f:
                tzstr = f.readline().strip() 
        else:
            print("pulseTimeZone.txt does not exist, using Zulu")
            tzstr = 'zulu'
        
        N=1
        
        Rshape = (ng,nRay,1+N)
        Cshape = (ng,nRay,1+2*N)
        
        #------Begin Parallel Code (runs slower than sequential due to copying memory overhead)-----
        # sharedRh = multiprocessing.Array('d', Rshape[0]*Rshape[1]*Rshape[2]*2)
        # sharedRv = multiprocessing.Array('d', Rshape[0]*Rshape[1]*Rshape[2]*2)
        # sharedC = multiprocessing.Array('d', Cshape[0]*Cshape[1]*Cshape[2]*2)

        # processes = []
        # for iRay, b in enumerate(pulseBoundaries):
        #     Xh = pulses[:,b[0]:b[1]+1,0]
        #     Xv = pulses[:,b[0]:b[1]+1,1]
            
        #     p = multiprocessing.Process(target=_processRay,
        #         args=(sharedRh, sharedRv, Rshape, sharedC, Cshape, Xh, Xv, iRay)
        #     )
        #     processes.append(p)
        #     p.start()
        
        # for p in processes:
        #     p.join()
            
        # Rh = np.frombuffer(sharedRh.get_obj(), dtype=np.complex128).reshape(Rshape)
        # Rv = np.frombuffer(sharedRv.get_obj(), dtype=np.complex128).reshape(Rshape)
        # C = np.frombuffer(sharedC.get_obj(), dtype=np.complex128).reshape(Cshape)
        #------End Parallel Code--------------------------------------------------------------------
        
        #------Begin Sequential Code----------------------------------------------------------------
        Rh = np.empty(Rshape, dtype=np.complex128)
        Rv = np.empty(Rshape, dtype=np.complex128)
        C = np.empty(Cshape, dtype=np.complex128)
        
        for iRay, b in enumerate(pulseBoundaries):
            Xh = pulses[:,b[0]:b[1]+1,0]
            Xv = pulses[:,b[0]:b[1]+1,1]
            
            _processRaySequential(Rh, Rv, Rshape, C, Cshape, Xh, Xv, iRay)     
        #------End Sequential Code------------------------------------------------------------------

        #gstep averaging
        Rh = _averageByGstep(Rh, gstep)
        Rv = _averageByGstep(Rv, gstep)
        C = _averageByGstep(C, gstep)

        zcal = self.header["config"]["systemZCal"][0] * np.ones((len(rr), 1)) + 8
        dcal = self.header["config"]["systemDCal"] * np.ones((len(rr), 1))
        pcal = self.header["config"]["systemPCal"] * np.ones((len(rr), 1))

        #noisedB = np.tile(10*np.log10(self.header['config']['noise']), (nRay,1))
        noisedB = np.zeros((nRay, 2))
        Sh = np.real(Rh[:,:,0]) - np.power(10, 0.1*(np.tile(noisedB[:,0], (len(rr), 1))))
        Sv = np.real(Rv[:,:,0]) - np.power(10, 0.1*(np.tile(noisedB[:,1], (len(rr), 1))))
        Sh[Sh < 0] = np.nan
        Sv[Sv < 0] = np.nan

        CC = np.zeros((len(rr), nRay))
        for m in range(-N, N+1):
            CC = CC + (3*(N**2) + 3*N - 1 - 5*(m**2)) * np.log(np.abs(C[:,:,m+N]))
        CC = np.exp( 3 * CC / ((2*N - 1)*(2*N + 1)*(2*N + 3)) )
        
        wavelength = self.header['desc']['wavelength']
        prt = self.header['config']['prt']
        va = 0.25 * wavelength / prt
        
        DBZ = 10*np.log10(Sh * np.tile(np.power(rr_km, 2).reshape(-1, 1), (1, nRay))) +\
            np.tile(zcal, (1, nRay))
            
        VEL = va / np.pi * np.angle(Rh[:,:,1])
        
        WIDTH = np.sqrt(2) * va / np.pi * np.sqrt(np.abs(np.log(Sh / np.abs(Rh[:,:,1]))))
        WIDTH[WIDTH.imag > 0] = np.nan
        WIDTH = WIDTH.astype(np.float64)
        
        ZDR = 10*np.log10(Sh / Sv) + np.tile(dcal, (1, nRay))
        
        PHIDP = np.angle(C[:,:,0+N])/np.pi*180 + np.tile(pcal, (1, nRay))
        PHIDP[PHIDP < -180] += 360
        PHIDP[PHIDP > 180] -= 360
        
        if N == 1:
            RHOHV = np.abs(C[:,:,0+N]) / np.sqrt(Sh * Sv)
            
        SNRH = 10*np.log10(Sh / np.power(10, 0.1*(np.tile(noisedB[:,0], (len(rr), 1)))))
        
        SNRV = 10*np.log10(Sv / np.power(10, 0.1*(np.tile(noisedB[:,1], (len(rr), 1)))))


        pulseWidthArr = np.tile(self.header['config']['pw'], (nRay,)).astype(np.float32)
        prtArr = np.tile(prt, (nRay,)).astype(np.float32)
        wavelengthArr = np.tile(wavelength, (nRay,)).astype(np.float32)

        cf = raxpolCf()
        cf.setVolume(0)
        cf.setSweep(0)
        cf.setTime(timeDoubleArr, tzstr)
        cf.setRange(rr.astype(np.float32))
        cf.setPosition(self.header['desc']['latitude'], self.header['desc']['longitude'])
        cf.setScanningStrategy("ppi")
        cf.setTargetAngle(self.header["config"]["sweepElevation"])
        cf.setAzimuth(azUnique)
        cf.setElevation(elevations)
        cf.setPulseWidthSeconds(pulseWidthArr)
        cf.setPrtSeconds(prtArr)
        cf.setWavelengthMeters(wavelengthArr)
        
        cf.setDBZ(DBZ.T)
        cf.setVEL(VEL.T)
        cf.setWIDTH(WIDTH.T)
        cf.setZDR(ZDR.T)
        cf.setPHIDP(PHIDP.T, "degrees")
        cf.setRHOHV(RHOHV.T)
        cf.setSNRH(SNRH.T)
        cf.setSNRV(SNRV.T)
        
        cf.setPulseBoundaries(pulseBoundaries)

        cfFilename = Path("cfrad." + Path(self.filename).stem + '.nc')
        outDir = self.casePath/"out"/("s"+str(azimuthBeamWidthDeg)+"o"+str(beamOverlapDeg))
        if not outDir.exists():
            outDir.mkdir(parents=True, exist_ok=True)
        cf.saveToFile(outDir/cfFilename)

        print(f"Saved to file", outDir/cfFilename)