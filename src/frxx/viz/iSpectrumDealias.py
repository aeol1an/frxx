import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import time

LEFT = 1
RIGHT = 3

#code from https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
def onSegment(p, q, r): 
    if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and 
           (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))): 
        return True
    return False
  
def orientation(p, q, r): 
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1])) 
    if (val > 0): 
        return 1
    elif (val < 0): 
        return 2
    else: 
        return 0

def doIntersect(p1,q1,p2,q2): 
    o1 = orientation(p1, q1, p2) 
    o2 = orientation(p1, q1, q2) 
    o3 = orientation(p2, q2, p1) 
    o4 = orientation(p2, q2, q1) 
  
    if ((o1 != o2) and (o3 != o4)): 
        return True
    if ((o1 == 0) and onSegment(p1, p2, q1)): 
        return True
    if ((o2 == 0) and onSegment(p1, q2, q1)): 
        return True
    if ((o3 == 0) and onSegment(p2, p1, q2)): 
        return True
    if ((o4 == 0) and onSegment(p2, q1, q2)): 
        return True

    return False
#end code from geeksforgeeks

def samePoint(p, q):
    if p[0] == q[0] and p[1] == q[1]:
        return True
    return False

def bound(val, upper, lower):
    if val > upper:
        return upper
    elif val < lower:
        return lower
    return val

def dealiasOnce(fields, currentUnfolds):
    if matplotlib.get_backend() != 'TkAgg':
        raise RuntimeError("Cannot run interactive display with current backend. "
                            "Switch to Tkagg with 'matplotlib.use(backend=\"TkAgg\")' "
                            "or if using jupyter, '%matplotlib tk.'")

    field = fields[0]
    ny, nx = field.shape
    
    dnx = nx // (currentUnfolds[0] + currentUnfolds[1] + 1)

    xx, yy = np.meshgrid(range(nx), range(ny))
    
    plt.rcParams['keymap.back'].remove('left')
    plt.rcParams['keymap.forward'].remove('right')
    
    fig = plt.figure(figsize=[8, 6])
    plt.pcolormesh(xx, yy, field, cmap='pyart_Carbone42')
    plt.title("Make selection by double clicking. Double right click to close polygon.")
    plt.xlim([-0.1*nx, nx+0.1*nx])
    plt.ylim([-0.1*ny, ny+0.1*ny])
    
    selectionDone = False
    points = []
    selectionResult = np.array([])
    path = None
    
    def addPoint(event):
        nonlocal selectionDone
        nonlocal points
        nonlocal selectionResult
        nonlocal path
        
        if (not event.dblclick) or (event.button != LEFT and event.button != RIGHT):
            return
        
        x = round(event.xdata)
        x = bound(x, nx-1, 0)
        y = round(event.ydata)
        y = bound(y, ny-1, 0)
        point = [x, y]
        
        #error finding to see if cycle or something like that
        if (not selectionDone) and (len(points) >= 3) and (not samePoint(points[0], point)) and (event.button != RIGHT):
            for i in range(len(points)-2):
                p1 = points[i]
                q1 = points[i+1]
                p2 = points[-1]
                if doIntersect(p1, q1, p2, point):
                    print("Intersecting points detected. Try again.")
                    return
                
        #logic to add to points array
        if not selectionDone:
            if event.button == RIGHT and (len(points) >= 3):
                point = points[0]
            points.append(point)
            plt.scatter(point[0], point[1], color='r')
            if len(points) >= 2:
                npPoints = np.array(points)
                npPoints = npPoints[-2:,].T
                plt.plot(npPoints[0], npPoints[1], color='r')
            event.canvas.draw()
            event.canvas.flush_events()

        #if made a circle or first and last set done to true
        if samePoint(points[0], points[-1]) and (len(points) >= 3):
            plt.title("Press left or right arrow key to execute unfold.")
            event.canvas.draw()
            event.canvas.flush_events()
            selectionDone = True
        
        #if done is true, calculate numpy boolean array with things to move over after selecting side of line
        if not selectionDone:
            return
        path = mplPath.Path(points)
        allpoints = np.array([xx,yy]).transpose(1,2,0).reshape(-1,2)
        selectionResultPos = path.contains_points(allpoints, radius=1)
        selectionResultNeg = path.contains_points(allpoints, radius=-1)
        if np.sum(selectionResultPos) > np.sum(selectionResultNeg):
            selectionResult = selectionResultPos
        else:
            selectionResult = selectionResultNeg
        selectionResult = selectionResult.reshape(ny,nx)
        fig.canvas.mpl_disconnect(boundary_handler)
        
    boundary_handler = fig.canvas.mpl_connect('button_press_event', addPoint)
    
    
    unfoldDone = False
    unfoldDirection = ""
    spaceAllocated = False
    currentUnfoldCount = np.array(currentUnfolds)
    unfoldedSelectionResult = np.array([])
    foldedSelectionResult = np.array([])
    def performUnfold(event):
        nonlocal unfoldDone
        nonlocal unfoldDirection
        nonlocal spaceAllocated
        nonlocal currentUnfoldCount
        nonlocal unfoldedSelectionResult
        nonlocal foldedSelectionResult
        
        if not selectionDone:
            return
        if not (event.key in ['left', 'right', 'enter']):
            return
        
        if unfoldDone and (event.key == 'enter'):
            plt.close()
            return
        
        if unfoldDone:
            return
        
        #execute unfolding on selectionResult
        if event.key == 'left':
            unfoldDirection = 'left'
            currentUnfoldCount += np.array([1,0])
            unfoldedSelectionResult = np.concatenate((selectionResult, np.full((ny, dnx), False, dtype=bool)), axis=1)
            foldedSelectionResult = np.concatenate((np.full((ny, dnx), False, dtype=bool), selectionResult), axis=1)
            
            if np.all(~unfoldedSelectionResult[:,0:dnx]):
                unfoldedSelectionResult = unfoldedSelectionResult[:, dnx:]
                foldedSelectionResult = foldedSelectionResult[:,dnx:]
                newField = np.array(field)
            else:
                newField = np.concatenate((np.full((ny, dnx), np.nan), field), axis=1)
                spaceAllocated = True
            
        elif event.key == 'right':
            unfoldDirection = 'right'
            currentUnfoldCount += np.array([0,1])
            unfoldedSelectionResult = np.concatenate((np.full((ny, dnx), False, dtype=bool), selectionResult), axis=1)
            foldedSelectionResult = np.concatenate((selectionResult, np.full((ny, dnx), False, dtype=bool)), axis=1)
            
            if np.all(~unfoldedSelectionResult[:, -dnx:]):
                unfoldedSelectionResult = unfoldedSelectionResult[:, 0:dnx]
                foldedSelectionResult = foldedSelectionResult[:, 0:dnx]
                newField = np.array(field)
            else:
                newField = np.concatenate((field, np.full((ny, dnx), np.nan)), axis=1)
                spaceAllocated = True
        
        newField[unfoldedSelectionResult] = newField[foldedSelectionResult]
        newField[foldedSelectionResult] = np.nan
        unfoldDone = True
            
        plt.clf()
        newxx, newyy = np.meshgrid(range(newField.shape[1]), range(newField.shape[0]))
        plt.pcolormesh(newxx, newyy, newField, cmap='pyart_Carbone42')
        plt.title("Executed Unfold. Press enter to confirm.")
        event.canvas.draw()
        event.canvas.flush_events()
    
    fig.canvas.mpl_connect('key_press_event', performUnfold)
    
    plt.show(block=True)
    
    plt.rcParams['keymap.back'].append('left')
    plt.rcParams['keymap.forward'].append('right')
    
    retFields = []
    for field in fields:
        if unfoldDirection == 'left':
            if not spaceAllocated:
                newField = np.array(field)
            else:
                newField = np.concatenate((np.full((ny, dnx), np.nan), field), axis=1)
            
        elif unfoldDirection == 'right':
            if not spaceAllocated:
                newField = np.array(field)
            else:
                newField = np.concatenate((field, np.full((ny, dnx), np.nan)), axis=1)
            
        newField[unfoldedSelectionResult] = newField[foldedSelectionResult]
        newField[foldedSelectionResult] = np.nan
        
        retFields.append(newField)
        
    return retFields, currentUnfoldCount, path, unfoldDirection

def iSpectrumDealias(fields, nyquist):
    if matplotlib.get_backend() != 'TkAgg':
        raise RuntimeError("Cannot run interactive display with current backend. "
                            "Switch to Tkagg with 'matplotlib.use(backend=\"TkAgg\")' "
                            "or if using jupyter, '%matplotlib tk.'")
    
    currentUnfoldCount = np.array([0,0])
    currentFields = fields
    paths = []
    directions = []
    xbounds = []
    
    while True:
        #menu to decide to unfold or bound x axis
        field = currentFields[0]
        ny, nx = field.shape
        
        xx, yy = np.meshgrid(range(nx), range(ny))
        
        fig = plt.figure(figsize=[8, 6])
        plt.pcolormesh(xx, yy, field, cmap='pyart_Carbone42')
        plt.xlim([-0.1*nx, nx+0.1*nx])
        plt.ylim([-0.1*ny, ny+0.1*ny])
        plt.title("Select 'u' to unfold, or 'b' to set x-axis bounds.")
        
        selectionDone = False
        def handleUnfoldOrBound(event):
            nonlocal selectionDone
            
            if not (event.key in ['b', 'u']):
                return
            
            if event.key == 'b':
                plt.title("Double click to set bounds")
                event.canvas.draw()
                event.canvas.flush_events()
                selectionDone = True
                fig.canvas.mpl_disconnect(handleUnfoldOrBoundConnection)
            if event.key == 'u':
                plt.close()
        
        handleUnfoldOrBoundConnection = fig.canvas.mpl_connect('key_press_event', handleUnfoldOrBound)
        
        
        def executeBound(event):
            nonlocal xbounds
            nonlocal selectionDone
            
            if not event.dblclick or event.button != LEFT:
                return
            if not selectionDone:
                return
            
            x = round(event.xdata)
            x = bound(x, nx-1, 0)
            
            xbounds.append(x)
            event.inaxes.axvline(x=x, color='r')
            event.canvas.draw()
            event.canvas.flush_events()
            
            if len(xbounds) == 2:
                time.sleep(1)
                plt.close()
                xbounds.sort()
        
        fig.canvas.mpl_connect('button_press_event', executeBound)
        
        plt.show(block=True)
        
        if selectionDone:
            break
        
        currentFields, currentUnfoldCount, path, direction = dealiasOnce(currentFields, currentUnfoldCount)
        paths.append(path)
        directions.append(direction)
    
    field = currentFields[0]
    ny, nx = field.shape
    xbounds[1] += 1
    
    xAxis = np.linspace(-nyquist-(2*currentUnfoldCount[0]*nyquist), nyquist+(2*currentUnfoldCount[1]*nyquist), nx)
    
    for i in range(len(currentFields)):
        currentFields[i] = currentFields[i][:,xbounds[0]:xbounds[1]]
    xAxis = xAxis[xbounds[0]:xbounds[1]]

    unfoldSaveData = {
        "unfoldData": [{"path": paths[i], "direction": directions[i]} for i in range(len(paths))],
        "xbounds": xbounds
    }
    
    return currentFields, xAxis, unfoldSaveData

def savedSpectrumDealias(fields, nyquist, unfoldSaveData):
    currentFields = fields
    unfolds = unfoldSaveData["unfoldData"]
    xbounds = unfoldSaveData["xbounds"]
    currentUnfolds = np.array([0,0])

    for unfold in unfolds:
        ny, nx = currentFields[0].shape

        dnx = nx // (currentUnfolds[0] + currentUnfolds[1] + 1)
        xx, yy = np.meshgrid(range(nx), range(ny))

        selectionResult = np.array([])

        path = unfold["path"]
        allpoints = np.array([xx,yy]).transpose(1,2,0).reshape(-1,2)
        selectionResultPos = path.contains_points(allpoints, radius=1)
        selectionResultNeg = path.contains_points(allpoints, radius=-1)
        if np.sum(selectionResultPos) > np.sum(selectionResultNeg):
            selectionResult = selectionResultPos
        else:
            selectionResult = selectionResultNeg
        selectionResult = selectionResult.reshape(ny,nx)

        direction = unfold["direction"]
        if direction == 'left':
            currentUnfolds += np.array([1,0])
            unfoldedSelectionResult = np.concatenate((selectionResult, np.full((ny, dnx), False, dtype=bool)), axis=1)
            foldedSelectionResult = np.concatenate((np.full((ny, dnx), False, dtype=bool), selectionResult), axis=1)
            
            if np.all(~unfoldedSelectionResult[:,0:dnx]):
                spaceAllocated = False
                unfoldedSelectionResult = unfoldedSelectionResult[:, dnx:]
                foldedSelectionResult = foldedSelectionResult[:,dnx:]
            else:
                spaceAllocated = True
            
        elif direction == 'right':
            currentUnfolds += np.array([0,1])
            unfoldedSelectionResult = np.concatenate((np.full((ny, dnx), False, dtype=bool), selectionResult), axis=1)
            foldedSelectionResult = np.concatenate((selectionResult, np.full((ny, dnx), False, dtype=bool)), axis=1)
            
            if np.all(~unfoldedSelectionResult[:, -dnx:]):
                spaceAllocated = False
                unfoldedSelectionResult = unfoldedSelectionResult[:, 0:dnx]
                foldedSelectionResult = foldedSelectionResult[:, 0:dnx]
            else:
                spaceAllocated = True

        newFields = []
        for field in currentFields:
            if direction == 'left':
                if not spaceAllocated:
                    newField = np.array(field)
                else:
                    newField = np.concatenate((np.full((ny, dnx), np.nan), field), axis=1)
                
            elif direction == 'right':
                if not spaceAllocated:
                    newField = np.array(field)
                else:
                    newField = np.concatenate((field, np.full((ny, dnx), np.nan)), axis=1)
                
            newField[unfoldedSelectionResult] = newField[foldedSelectionResult]
            newField[foldedSelectionResult] = np.nan
            
            newFields.append(newField)

        currentFields = list(np.array(newFields))

    ny, nx = currentFields[0].shape
        
    xAxis = np.linspace(-nyquist-(2*currentUnfolds[0]*nyquist), nyquist+(2*currentUnfolds[1]*nyquist), nx)

    for i in range(len(currentFields)):
        currentFields[i] = currentFields[i][:,xbounds[0]:xbounds[1]]
    xAxis = xAxis[xbounds[0]:xbounds[1]]

    return currentFields, xAxis