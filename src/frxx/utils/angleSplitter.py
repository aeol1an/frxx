import numpy as np
from numba import njit

@njit('boolean(float64, float64, float64)', inline='always', cache=True)
def inDegreeRange(val, low, high):
    if low < high:
        return val > low and val < high
    else:
        return val > low or val < high
    
from numba import njit
import numpy as np

@njit('Tuple((int64[:,:], float32[:]))(float32[:], float32, float32)', cache=True)
def findPulseBoundaries(angle, pixelWidthDeg, beamOverlapDeg):
    halfSwath = 0.5 * (pixelWidthDeg + 2 * beamOverlapDeg)
    nPulses = len(angle)
    
    # Discretize angles to group centers (offset by half spacing)
    angleSpacing = halfSwath
    angleDiscrete = np.empty(nPulses, dtype=np.float32)
    for i in range(nPulses):
        angleDiscrete[i] = (np.rint(angle[i] / angleSpacing) * angleSpacing + 0.5 * angleSpacing) % 360
    
    # Find unique consecutive values
    angleUniqueList = [angleDiscrete[0]]
    for i in range(1, nPulses):
        if angleDiscrete[i] != angleDiscrete[i - 1]:
            angleUniqueList.append(angleDiscrete[i])
    
    nGroups = len(angleUniqueList)
    angleUnique = np.empty(nGroups, dtype=np.float32)
    for i in range(nGroups):
        angleUnique[i] = angleUniqueList[i]
    
    # Precompute bounds
    lowBound = np.empty(nGroups, dtype=np.float32)
    highBound = np.empty(nGroups, dtype=np.float32)
    for j in range(nGroups):
        lowBound[j] = (angleUnique[j] - halfSwath) % 360
        highBound[j] = (angleUnique[j] + halfSwath) % 360
    
    pulseBoundaries = np.zeros((nGroups, 2), dtype=np.int64)
    
    # 0 = not entered, 1 = inside, 2 = exited
    state = np.zeros(nGroups, dtype=np.int8)
    
    # Determine direction
    increasing = np.mean(np.sign(angle[1:]-angle[0:-1])) > 0
    
    if increasing:
        for i in range(nPulses):
            currAngle = angle[i]
            for j in range(nGroups):
                if state[j] == 2:
                    continue
                
                currIn = inDegreeRange(currAngle, lowBound[j], highBound[j])
                
                if state[j] == 0 and currIn:
                    pulseBoundaries[j, 0] = i
                    state[j] = 1
                elif state[j] == 1 and not currIn:
                    pulseBoundaries[j, 1] = i - 1
                    state[j] = 2
        
        # Handle groups still inside at end of array
        for j in range(nGroups):
            if state[j] == 1:
                pulseBoundaries[j, 1] = nPulses - 1
    else:
        for i in range(nPulses - 1, -1, -1):
            currAngle = angle[i]
            for j in range(nGroups):
                if state[j] == 2:
                    continue
                
                currIn = inDegreeRange(currAngle, lowBound[j], highBound[j])
                
                if state[j] == 0 and currIn:
                    pulseBoundaries[j, 1] = i
                    state[j] = 1
                elif state[j] == 1 and not currIn:
                    pulseBoundaries[j, 0] = i + 1
                    state[j] = 2
        
        # Handle groups still inside at end of traversal
        for j in range(nGroups):
            if state[j] == 1:
                pulseBoundaries[j, 0] = 0
    
    return pulseBoundaries, angleUnique