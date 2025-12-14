import numpy as np
import requests
import time
from pyart.core import cartesian_to_geographic_aeqd

def cart2polar(x: float, y: float):
    r = np.sqrt(x**2 + y**2)
    az = -np.arctan2(y, x) + (np.pi / 2)
    az = np.degrees(az)
    az = (az + 360) % 360

    return r, az

def polar2cart(r: float, az: float):
    x = r*np.cos(np.radians(-(az - 90)))
    y = r*np.sin(np.radians(-(az - 90)))

    return x, y

def getElevations(
    lats, lons, dataset="ned10m", 
    batchSize=100, maxRetries=3, retryDelay=1.1
):
    if not isinstance(lats, np.ndarray) or not isinstance(lons, np.ndarray):
        raise TypeError("Latitudes and longitudes must be numpy arrays")
    if lats.ndim != 1 or lons.ndim != 1:
        raise ValueError("Latitudes and longitudes must be 1D arrays")
    if len(lats) != len(lons):
        raise ValueError("Latitudes and longitudes must have the same length")
    
    nPoints = len(lats)
    elevations = np.full(nPoints, np.nan)
    
    for i in range(0, nPoints, batchSize):
        batchLats = lats[i:i + batchSize]
        batchLons = lons[i:i + batchSize]
        locationsStr = "|".join([f"{lat},{lon}" for lat, lon in zip(batchLats, batchLons)])
        for attempt in range(maxRetries):
            try:
                url = f"https://api.opentopodata.org/v1/{dataset}?locations={locationsStr}"
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                batchElevations = np.array([result["elevation"] for result in data["results"]])
                elevations[i:i + len(batchElevations)] = batchElevations
                break
                
            except requests.exceptions.RequestException as e:
                if attempt < maxRetries - 1:
                    print(f"Error in batch {i//batchSize + 1}, retrying in {retryDelay}s: {e}")
                    time.sleep(retryDelay)
                else:
                    print(f"Failed to get elevations for batch {i//batchSize + 1}: {e}")
        if i + batchSize < nPoints:
            time.sleep(retryDelay)
    return elevations
        

def beamHeightWithRadarHeight(rKm, az, el, radarHtM, radarLat, radarLon):
    radarLocEl = getElevations(np.array([radarLat]), np.array([radarLon]))
    
    theta_e = np.deg2rad(el)
    theta_a = np.deg2rad(az)
    R = (6370997.0 * 4.0 / 3.0) + radarLocEl[0] + radarHtM
    r = rKm.astype(np.float64) * 1000.0

    z = (r**2 + R**2 + 2.0 * r * R * np.sin(theta_e)) ** 0.5 - R
    s = R * np.arcsin(r * np.cos(theta_e) / (R + z))  # arc length in m.
    x = s * np.sin(theta_a)
    y = s * np.cos(theta_a)
    
    lons, lats = cartesian_to_geographic_aeqd(
        x, y, 
        radarLon, radarLat, 
        6370997.0+radarLocEl[0]+radarHtM
    )
    
    elArray = getElevations(lats, lons)
    
    return z + (radarLocEl - elArray) + radarHtM
    


