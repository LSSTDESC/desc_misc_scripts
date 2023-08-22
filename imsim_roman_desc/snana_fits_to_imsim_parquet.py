# For format of the output parquet and hdf5 files, see:
# https://docs.google.com/document/d/1zlJ7AsFjwnVYlkqcoXq1G6YUXJDOn7kMP8Jbl6SGRxk/edit#heading=h.hlzr61r0h2en

import sys
import io
import re
import copy
import pathlib
import logging
import argparse
import numpy
import healpy
import pandas
import pyarrow
import pyarrow.parquet
import h5py

from astropy.table import Row, Table

# _rundir = pathlib.Path( __file__ ).parent

_logger = logging.getLogger( "main" )
if not _logger.hasHandlers():
    _logout = logging.StreamHandler( sys.stderr )
    _logger.addHandler( _logout )
    _formatter = logging.Formatter( f'[%(asctime)s - %(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S' )
    _logout.setFormatter( _formatter )
_logger.setLevel( logging.INFO )

# ======================================================================
# Encapsulates all the data for one healpix

class OutputFile:
    # The next mess configures translation from SNANA to the
    # schema of the output files

    general_params = { 'snid_snana': 'SNID',
                       'ra': 'RA',
                       'dec': 'DEC',
                       'zCMB': 'SIM_REDSHIFT_CMB',
                       'mwEBV': 'SIM_MWEBV',
                       'AV': 'SIM_AV',
                       'RV': 'SIM_RV',
                       'template_index': 'SIM_TEMPLATE_INDEX',
                       'vpec': 'SIM_VPEC',
                       'hostra': 'HOSTGAL_RA',
                       'hostdec': 'HOSTGAL_DEC',
                       'hostmag_g': 'HOSTGAL_MAG_g',
                       'hostmag_i': 'HOSTGAL_MAG_i',
                       'hostmag_F': 'HOSTGAL_MAG_F',
                       'snsep': 'HOSTGAL_SNSEP',
                       'peakmjd': 'PEAKMJD',
                       'peakmag_g': 'SIM_PEAKMAG_g',
                       'peakmag_i': 'SIM_PEAKMAG_i',
                       'peakmag_F': 'SIM_PEAKMAG_F',
                       }
    
    model_params = { 'SALT2.WFIRST-H17': { 'salt2x0': 'SIM_SALT2x0',
                                           'salt2x1': 'SIM_SALT2x1',
                                           'salt2c': 'SIM_SALT2c',
                                           'salt2mB': 'SIM_SALT2mB',
                                           'salt2alpha': 'SIM_SALT2alpha',
                                           'salt2beta': 'SIM_SALT2beta',
                                           'salt2gammaDM': 'SIM_SALT2gammaDM',
                                          }
                     }
    # Keep track of what warnings have already been issued
    #  about things with no entry in model_params
    seen_unknown_models = set()

    col_map = { 'id': ( pyarrow.int64(), 'SNID' ),
                'ra': ( pyarrow.float64(), 'RA' ),
                'dec': ( pyarrow.float64(), 'DEC' ),
                'host_id': ( pyarrow.int64(), 'HOSTGAL_OBJID' ),
                'model_name': ( pyarrow.string(), 'SIM_MODEL_NAME' ),
                'model_param_names': ( pyarrow.list_( pyarrow.string() ), None ),
                'model_param_values': ( pyarrow.list_( pyarrow.float32() ), None ),
                'start_mjd': ( pyarrow.float32(), None ),
                'end_mjd': ( pyarrow.float32(), None ),
               }
    trims = { 'SIM_MODEL_NAME' }

    
    def __init__( self, pix, outdir, clobber=False ):
        self.pix = pix
        self.outdir = pathlib.Path( outdir )
        self.hdf5filepath = outdir / f'snana_{pix}.hdf5'
        self.pqfilepath = outdir / f'snana_{pix}.parquet'

        if self.pqfilepath.exists():
            if clobber:
                self.pqfilepath.unlink()
            else:
                raise RuntimeError( f'{self.pqfilepath} exists, not clobbering.' )
        if self.hdf5filepath.exists():
            if clobber:
                self.hdf5filepath.unlink()
            else:
                raise RuntimeError( f'{self.hdf5filepath} exists, not clobber.' )

        self.maintable = { k: [] for k in OutputFile.col_map.keys() }
        self.hdf5file = h5py.File( self.hdf5filepath, 'w' )


    def addsn( self, headrow, spechdr, specdata ):
        objhdr = spechdr[ spechdr['SNID'] == headrow['SNID'] ]
        # Irritatingly, the last call will return different types if there
        #   is only one row vs. multiple rows
        if isinstance( objhdr, Row ):
            objhdr = Table( objhdr )
        
        nmjds = len( objhdr )
        if nmjds == 0:
            _logger.warning( f"No mjds for object {headrow['SNID']}" )
            return

        hdf5group = self.hdf5file.create_group( str( headrow['SNID'] ) )

        # Extract (most of) the main table data
        
        for key, val in OutputFile.col_map.items():
            if val[1] is not None:
                if val[1] in OutputFile.trims:
                    self.maintable[key].append( headrow[ val[1] ].strip() )
                else:
                    self.maintable[key].append( headrow[ val[1] ] )
            else:
                self.maintable[key].append( None )

        # Get out the model params

        modelname = headrow['SIM_MODEL_NAME'].strip()
        params = dict( OutputFile.general_params )
        if modelname in OutputFile.model_params.keys():
            params.update( OutputFile.model_params[ modelname ] )
        else:
            if modelname not in OutputFile.seen_unknown_models:
                _logger.warning( f"Unknown model {modelname}, just saving general parameters" )
                OutputFile.seen_unknown_models.add( modelname )
        self.maintable['model_param_names'][-1] = list( params.keys() )
        self.maintable['model_param_values'][-1] = [ headrow[v] for v in params.values() ]

        # Extract all the spectra
        # Going to assume that the lambdamin, lambdamax, and lambdabin
        # are the same for all dates for a given object

        headrow0 = objhdr[0]
        nlambdas = headrow0['NBIN_LAM']
        lambdas = numpy.arange( headrow0['LAMMIN']+headrow0['LAMBIN']/2., headrow0['LAMMAX'], headrow0['LAMBIN'] )
        flam = numpy.empty( ( nmjds, nlambdas ), dtype=numpy.float32 )

        h5mjds = hdf5group.create_dataset( 'mjd', data=objhdr['MJD'] )
        h5lambdas = hdf5group.create_dataset( 'lambda', data=lambdas )

        # The -1 is because FITS indexes are 1-offset
        for i, headrow in enumerate(objhdr):
            flam[ i, : ] = specdata[ headrow['PTRSPEC_MIN']-1 : headrow['PTRSPEC_MAX'] ]['SIM_FLAM']
        h5flam = hdf5group.create_dataset( 'flambda', data=flam )

        # Add the start and end mjd to the main (summary) table

        self.maintable['start_mjd'][-1] = objhdr[0]['MJD']
        self.maintable['end_mjd'][-1] = objhdr[-1]['MJD']


    def finalize( self ):
        maintable_schema = pyarrow.schema( { k: OutputFile.col_map[k][0] for k in OutputFile.col_map.keys() } )
        self.hdf5file.close()
        pyarrowtable = pyarrow.table( self.maintable, schema=maintable_schema )
        pyarrow.parquet.write_table( pyarrowtable, self.pqfilepath )
        

# ======================================================================

def main():
    parser = argparse.ArgumentParser( description="Convert SNANA FITS files to Parquet files for DESC+Roman imsim",
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument( '-d', '--directories', nargs='+', required=True,
                         help="Directories to find the HEAD and SPEC fits files" )
    # parser.add_argument( '-f', '--files', default=[], nargs='+',
    #                      help="Names of HEAD.fits[.[fg]z] files; default is to read all in directory" )
    parser.add_argument( '-n','--nside', default=32, type=int, help="nside for healpix (ring)" )
    parser.add_argument( '-o', '--outdir', default='.', help="Where to write output files." )
    parser.add_argument( '--verbose', action='store_true', default=False,
                         help="Set log level to DEBUG (default INFO)" )
    parser.add_argument( '-z', '--zeropoint', default=8.9, type=float,
                         help="Zeropoint to move from magnitudes to fluxes" )
    parser.add_argument( '-u', '--flux-units', default='Jy', help="Units of flux (this should match the zeropoint!)" )
    parser.add_argument( '-c', '--clobber', default=False, action='store_true', help="Overwrite existing files?" )
    args = parser.parse_args()

    if args.verbose:
        _logger.setLevel( logging.DEBUG )

    outdir = pathlib.Path( args.outdir )

    # All input files
    headfiles = []
    specfiles = []

    # Output files (OutputFile objects), indexed by healpix
    outputfiles = {}
    
    # Collect all the head and spec files

    for direc in args.directories:
        direc = pathlib.Path( direc )
        localheadfiles = list( direc.glob( '*HEAD.FITS.gz' ) )
        headfiles.extend( localheadfiles )
        
    headre = re.compile( '^(.*)HEAD\.FITS\.gz' )
    for headfile in headfiles:
        direc = headfile.parent
        match = headre.search( headfile.name )
        if match is None:
            raise ValueError( f"Failed to parse {headfile.name} for *HEAD.FITS.gz" )
        specfile = direc / f"{match.group(1)}SPEC.FITS.gz"
        if not headfile.is_file():
            raise FileNotFoundError( f"Can't read {headfile}" )
        if not specfile.is_file():
            raise FileNotFoundError( f"Can't read {specfile}" )
        specfiles.append( specfile )

    # Make one pass through the HEAD files to figure out which healpix
    # we need, and open all the HDF5 files.  (TODO: this may not
    # scale, keeping all of the HDF5 files open at once.  I'm hoping
    # it'll be fine, but if we run out of file descriptors, we'll have
    # to be more clever.  For 20 sq deg, with each healpix being (about)
    # 1.8²=~3 sq deg, this won't be a problem.)

    _logger.info( "Reading all HEAD files to determine existing healpix" )
    for headfile in headfiles:
        _logger.debug( f"Reading {headfile}" )
        head = Table.read( headfile, memmap=False )
        head['SNID'] = head['SNID'].astype( numpy.int64 )
        if len(head) == 0:
            _logger.warning( f"{headfile.name} had 0 length, skipping it" )
            continue
        for row in head:
            pix = healpy.pixelfunc.ang2pix( args.nside,
                                            numpy.radians( 90.-row['DEC'] ),
                                            numpy.radians( 360.-row['RA'] ) )
            if pix not in outputfiles.keys():
                outputfiles[pix] = OutputFile( pix, outdir, clobber=args.clobber )
    
    _logger.info( f"Found {len(outputfiles)} different healpix" )
        
    _logger.debug( f'Headfiles: {[headfiles]}' )
    _logger.debug( f'Photfiles: {[specfiles]}' )

    # Now go through all of the HEAD/SPEC files, adding the relevant
    # data for each object to the correct OutputFile (based on healpix)
    
    _logger.info( f'Reading {len(headfiles)} SNANA HEAD/SPEC files' )

    for headfile, specfile in zip( headfiles, specfiles ):
        _logger.info( f"Reading {headfile}" )

        head = Table.read( headfile, memmap=False )
        spechdr = Table.read( specfile, hdu=1, memmap=False )
        specdata = Table.read( specfile, hdu=2, memmap=False )

        head['SNID'] = head['SNID'].astype( numpy.int64 )
        spechdr['SNID'] = spechdr['SNID'].astype( numpy.int64 )

        _logger.info( f"...done reading, processing {len(head)} objects." )
        for row in head:
            phi = numpy.radians( 360. - row['RA'] )
            theta = numpy.radians( 90. - row['DEC'] )
            if ( theta < 0 ) or ( theta > numpy.pi ):
                raise ValueError( f'Bad dec for {snid} : {row["DEC"]}' )
            if ( phi >= 2*numpy.pi ): phi -= 2*numpy.pi
            if ( phi < 0 ): phi += 2*numpy.pi

            pix = healpy.pixelfunc.ang2pix( args.nside, theta, phi )
            outputfiles[pix].addsn( row, spechdr, specdata )

    # Close out all the files

    for pix, outputfile in outputfiles.items():
        outputfile.finalize()
    
    _logger.info( "Done." )
        
# ======================================================================

if __name__ == "__main__":
    main()

    
