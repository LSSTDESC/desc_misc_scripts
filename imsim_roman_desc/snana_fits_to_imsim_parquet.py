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
from astropy.table import Table
import astropy.io
import pyarrow
import pyarrow.parquet
import h5py

# _rundir = pathlib.Path( __file__ ).parent

_logger = logging.getLogger( "main" )
if not _logger.hasHandlers():
    _logout = logging.StreamHandler( sys.stderr )
    _logger.addHandler( _logout )
    _formatter = logging.Formatter( f'[%(asctime)s - %(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S' )
    _logout.setFormatter( _formatter )
_logger.setLevel( logging.INFO )

# ======================================================================

def main():
    parser = argparse.ArgumentParser( description="Convert SNANA FITS files to Parquet files for DESC+Roman imsim",
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument( '-d', '--directory', default=None, required=True,
                         help="Directory to find the HEAD and SPEC fits files" )
    parser.add_argument( '-f', '--files', default=[], nargs='+',
                         help="Names of HEAD.fits[.[fg]z] files; default is to read all in directory" )
    parser.add_argument( '-n','--nside', default=32, type=int, help="nside for healpix (ring)" )
    parser.add_argument( '-o', '--outdir', default='.', help="Where to write output files." )
    parser.add_argument( '--verbose', action='store_true', default=False,
                         help="Set log level to DEBUG (default INFO)" )
    parser.add_argument( '-c', '--clobber', default=False, action='store_true', help="Overwrite existing files?" )
    args = parser.parse_args()

    if args.verbose:
        _logger.setLevel( logging.DEBUG )

    outdir = pathlib.Path( args.outdir )
        
    # Read the dump file

    direc = pathlib.Path( args.directory )
    if not direc.is_dir():
        raise RuntimeError( f"{str(direc)} isn't a directory" )
    dumpfile = direc / f"{direc.name}.DUMP"
    if not dumpfile.is_file():
        raise RuntimeError( f"Can't read {dumpfile}" )
    dump = astropy.io.ascii.read( dumpfile )

    if len( args.files ) == 0:
        headfiles = list( direc.glob( '*HEAD.FITS.gz' ) )
    else:
        headfiles = args.files
        headfiles = [ direc / h for h in headfiles ]

    # Make sure all HEAD.FITS.gz and spec.FITS.gz files exist

    headre = re.compile( '^(.*)HEAD\.FITS\.gz' )
    specfiles = []
    for headfile in headfiles:
        match = headre.search( headfile.name )
        if match is None:
            raise ValueError( f"Failed to parse {headfile.name} for *.HEAD.FITS.gz" )
        specfile = direc / f"{match.group(1)}SPEC.FITS.gz"
        if not headfile.is_file():
            raise FileNotFoundError( f"Can't read {headfile}" )
        if not specfile.is_file():
            raise FileNotFoundError( f"Can't read {specfile}" )
        specfiles.append( specfile )

    _logger.debug( f'Headfiles: {[headfiles]}' )
    _logger.debug( f'Photfiles: {[specfiles]}' )

    # Set up the parameters we read for each model
    # general_params are read for every model
    
    general_params = { 'zCMB': 'SIM_REDSHIFT_CMB',
                       'mwEBV': 'SIM_MWEBV',
                       'AV': 'SIM_AV',
                       'RV': 'SIM_RV' }
    
    model_params = { 'SALT2.WFIRST-H17': { 'salt2x0': 'SIM_SALT2x0',
                                           'salt2x1': 'SIM_SALT2x1',
                                           'salt2c': 'SIM_SALT2c',
                                           'salt2mB': 'SIM_SALT2mB',
                                           'salt2alpha': 'SIM_SALT2alpha',
                                           'salt2beta': 'SIM_SALT2beta',
                                           'salt2gammaDM': 'SIM_SALT2gammaDM',
                                          }
                     }
    seen_unknown_models = set()

    # Set up the columns we map from the HEAD file to the main parquet file
    
    maintables = {}
    hdf5files = {}
    col_map = { 'id': ( pyarrow.int64(), 'SNID' ),
                'ra': ( pyarrow.float64(), 'RA' ),
                'dec': ( pyarrow.float64(), 'DEC' ),
                'host_id': ( pyarrow.int64(), 'HOSTGAL_OBJID' ),
                'model name': ( pyarrow.string(), 'SIM_MODEL_NAME' ),
                'model params (names)': ( pyarrow.list_( pyarrow.string() ), None ),
                'model params (values)': ( pyarrow.list_( pyarrow.float32() ), None ),
                'start_mjd': ( pyarrow.float32(), None ),
                'end_mjd': ( pyarrow.float32(), None ),
               }
    trims = { 'SIM_MODEL_NAME' }
    maintable_schema = pyarrow.schema( { k: col_map[k][0] for k in col_map.keys() } )

    # Make one pass through the HEAD files to figure out which healpix
    # we need; create the spectroscopy HDF5 files.  (TODO: this may not
    # scale, keeping all of the HDF5 files open at once.  I'm hoping
    # it'll be fine, but if we run out of file descriptors, we'll have
    # to be more clever.  For 20 sq deg, with each healpix being (about)
    # 1.8²=~3 sq deg, this won't be a problem.)

    _logger.info( "Reading all HEAD files to determine existing healpix" )
    for headfile in headfiles:
        head = Table.read( headfile )
        if len(head) == 0:
            _logger.warning( f"{headfile.name} had 0 length, skipping it" )
            continue
        for row in head:
            pix = healpy.pixelfunc.ang2pix( args.nside,
                                            numpy.radians( 90.-row['DEC'] ),
                                            numpy.radians( 360.-row['RA'] ) )
            if pix not in maintables:
                # Make sure files don't already exist
                # TODO : abillity to append
            
                pqfile = outdir / f'snana_{pix}.parquet'
                h5file = outdir / f'snana_{pix}.hdf5'
                if pqfile.exists():
                    if args.clobber:
                        pqfile.unlink()
                    else:
                        raise RuntimeError( f'{pqfile} (at least) exists, not clobberng.' )
                if h5file.exists():
                    if args.clobber:
                        hdf5file.unlink()
                    else:
                        raise RuntimeError( f'{h5file} (at least) exists, not clobbering.' )

                # Make the variable that will hold the parquet information,
                # and hopen the hdf5 file
                
                maintables[pix] = { k: [] for k in col_map.keys() }
                hdf5files[pix] = h5py.File( outdir / f'snana_{pix}.hdf5', 'w' )

    _logger.info( f"Found {len(maintables)} different healpix" )

    # Now read the HEAD and SPEC files, augmenting the data above

    for headfile, specfile in zip( headfiles, specfiles ):
        head = Table.read( headfile )
        if len(head) == 0:
            continue
        # SNID was written as a string, we need it to be a bigint
        head['SNID'] = head['SNID'].astype( numpy.int64 )

        spechdr = Table.read( specfile, hdu=1 )
        spechdr['SNID'] = spechdr['SNID'].astype( numpy.int64 )
        specdata = Table.read( specfile, hdu=2 )
        
        for row in head:
            modelname = row['SIM_MODEL_NAME'].strip()
            
            phi = numpy.radians( 360. - row['RA'] )
            theta = numpy.radians( 90. - row['DEC'] )
            if ( theta < 0 ) or ( theta > numpy.pi ):
                raise ValueError( f'Bad dec for {row["SNID"]} : {row["DEC"]}' )
            if ( phi >= 2*numpy.pi ): phi -= 2*numpy.pi
            if ( phi < 0 ): phi += 2*numpy.pi

            pix = healpy.pixelfunc.ang2pix( args.nside, theta, phi )

            hdf5group = hdf5files[pix].create_group( str( row['SNID'] ) )

            for key, val in col_map.items():
                if val[1] is not None:
                    if val[1] in trims:
                        maintables[pix][key].append( row[ val[1] ].strip() )
                    else:
                        maintables[pix][key].append( row[ val[1] ] )
                else:
                    maintables[pix][key].append( None )

            # Get out the model params

            params = dict( general_params )
            if modelname in model_params.keys():
                params.update( model_params[ modelname ] )
            else:
                if modelname not in seen_unknown_models:
                    _logger.warning( f"Unknown model {modelname}, just saving general parameters" )
                    seen_unknown_models.add( modelname )
            maintables[pix]['model params (names)'][-1] = list( params.keys() )
            maintables[pix]['model params (values)'][-1] = [ row[v] for v in params.values() ]
                    
            # Extract all the spectra
            # Going to assume that the lambdamin, lambdamax, and lambdabin
            # are the same for all dates for a given object

            objhdr = spechdr[ spechdr['SNID'] == row['SNID'] ]
            nmjds = len( objhdr )
            row0 = objhdr[0]
            nlambdas = row0['NBIN_LAM']
            lambdas = numpy.arange( row0['LAMMIN']+row0['LAMBIN']/2., row0['LAMMAX'], row0['LAMBIN'] )
            flam = numpy.empty( ( nmjds, nlambdas ), dtype=numpy.float32 )

            h5mjds = hdf5group.create_dataset( 'mjd', data=objhdr['MJD'] )
            h5lambdas = hdf5group.create_dataset( 'lambda', data=lambdas )

            # The -1 is because FITS indexes are 1-offset
            for i, row in enumerate(objhdr):
                flam[ i, : ] = specdata[ row['PTRSPEC_MIN']-1 : row['PTRSPEC_MAX'] ]['SIM_FLAM']
            h5flam = hdf5group.create_dataset( 'flambda', data=flam )

            # Add the start and end mjd to the main (summary) table
            
            maintables[pix]['start_mjd'][-1] = objhdr[0]['MJD']
            maintables[pix]['end_mjd'][-1] = objhdr[-1]['MJD']
            
    # Close the hdf5 files and create the main parquet files

    for healpix, data in maintables.items():
        hdf5files[pix].close()

        arrowdata = copy.deepcopy( data )
        # import pdb; pdb.set_trace()
        # for key, val in arrowdata.items():
        #     arrowdata[key] = pyarrow.array( arrowdata[key], type=col_map[key][0] )
        maintable = pyarrow.table( arrowdata, schema=maintable_schema )
        ofpath = outdir / f'snana_{healpix}.parquet'
        pyarrow.parquet.write_table( maintable, ofpath )
        
# ======================================================================

if __name__ == "__main__":
    main()

    
