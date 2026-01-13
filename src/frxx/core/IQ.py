import numpy as np
import xarray as xr
import json

from .data import frxxData

class IQ(frxxData):

	def __init__(self, ds: xr.Dataset | None = None):
		super().__init__()
		
		if ds is None:
			#create new
			self.ds = self.ds.assign_coords(
				iq=np.arange(2)
			)

			self.optionalBools = {
				#None of these are require true, but are helpful for documentation
				"history": False,
				"addtl_comments": False,
			}
		else:
			# do some validation
			pass