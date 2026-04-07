from . import cmaps

_nCBTicks = 5

moments = {
	'DBZ': {
		'ranges': (0, 75, _nCBTicks),
		'cmap': 'pyart_Carbone42',
		'units': 'dBZ'
	},
	'VEL': {
		'ranges': (-100, 100, _nCBTicks),
		'cmap': 'pyart_Carbone42',
		'units': 'm/s'
	},
	'CORVEL': {
		'ranges': (-100, 100, _nCBTicks),
		'cmap': 'pyart_Carbone42',
		'units': 'm/s'
	},
	'ZDR': {
		'ranges': (-5, 8, _nCBTicks),
		'cmap': cmaps.dmap(256),
		'units': 'dB'
	},
	'RHOHV': {
		'ranges': (0.2, 1.05, _nCBTicks),
		'cmap': cmaps.rmap(256),
		'units': ''
	} 
}

spectra = {
    #DPSD
	'PSD': {
		'title': 'Power Spectra (dB)',
		'shortTitle': 'sS$_H$',
		'ranges': (0, 75, 5),
		'cmap': 'pyart_Carbone42',
		'units': 'dB'
	},
	'SNRH': {
		'title': 'Spectral Horizontal Signal to Noise Ratio (dB)',
		'shortTitle': 'sSNR$_V$',
		'ranges': (0, 75, 5),
		'cmap': 'pyart_Carbone42',
		'units': 'dB'
	},
	'ZDR': {
		'title': 'Spectral Differential Reflectivity (dB)',
		'shortTitle': 'sZDR',
		'ranges': (-5, 8, 5),
		'cmap': cmaps.dmap(256),
		'units': 'dB'
	},
	'RHOHV': {
		'title': 'Spectral Correlation Coefficient',
		'shortTitle': 's$\\rho_{HV}$',
		'ranges': (0.2, 1.05, 5),
		'cmap': cmaps.rmap(256),
		'units': ''
	},
    
	#DCA
    'ZDRVAR': {
        'title': 'Spectral Variance of\nDifferential Reflectivity (dB$^2$)',
        'shortTitle': '$\\sigma^2$sZDR',
        'ranges': (0, 50, 5),
        'cmap': 'pyart_Carbone42',
        'units': 'dB$^2$'
    },
    'RHVVAR': {
        'title': 'Spectral Variance of\nCorrelation Coefficient',
        'shortTitle': '$\\sigma^2$s$\\rho_{HV}$',
        'ranges': (0, 0.25, 5),
        'cmap': 'pyart_Carbone42',
        'units': ''
    },
    'RAGG': {
        'title': "Normalized Rain Aggregation Parameter",
        'shortTitle': 'A$_{N,rain}$',
        'ranges': (0, 1, 5),
        'cmap': 'seismic',
        'units': ''
    },
    'DCAHREF': {
        'title': lambda filterStrength: f'Power Spectra Filtered by [A$_{{rain}}$]$^{filterStrength}$ (dB)',
        'shortTitle': 'sS$_{Hf}$',
        'ranges': (0, 10, 5),
        'cmap': 'Wistia',
        'units': 'dB'
    }
}