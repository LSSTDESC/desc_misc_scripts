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

import astropy.table
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

    general_params = { 'template_index': 'SIM_TEMPLATE_INDEX',
                       }

    model_params = { 'SALT2.WFIRST-H17': { 'salt2_x0': 'SIM_SALT2x0',
                                           'salt2_x1': 'SIM_SALT2x1',
                                           'salt2_c': 'SIM_SALT2c',
                                           'salt2_mB': 'SIM_SALT2mB',
                                           'salt2_alpha': 'SIM_SALT2alpha',
                                           'salt2_beta': 'SIM_SALT2beta',
                                           'salt2_gammaDM': 'SIM_SALT2gammaDM',
                                          },
                     'SALT3.NIR_WAVEEXT': { 'salt2_x0': 'SIM_SALT2x0',
                                            'salt2_x1': 'SIM_SALT2x1',
                                            'salt2_c': 'SIM_SALT2c',
                                            'salt2_mB': 'SIM_SALT2mB',
                                            'salt2_alpha': 'SIM_SALT2alpha',
                                            'salt2_beta': 'SIM_SALT2beta',
                                            'salt2_gammaDM': 'SIM_SALT2gammaDM',
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
                'start_mjd': ( pyarrow.float32(), None ),
                'end_mjd': ( pyarrow.float32(), None ),
                'z_CMB': ( pyarrow.float32(), 'SIM_REDSHIFT_CMB' ),
                'mw_EBV': ( pyarrow.float32(), 'SIM_MWEBV' ),
                'mw_extinction_applied': ( pyarrow.bool_(), '_no_header_keyword', False ),
                # Also a do not apply flag?
                'AV': ( pyarrow.float32(), 'SIM_AV' ),
                'RV': ( pyarrow.float32(), 'SIM_RV' ),
                'v_pec': ( pyarrow.float32(), 'SIM_VPEC' ),
                'host_ra': ( pyarrow.float64(), 'HOSTGAL_RA' ),
                'host_dec': ( pyarrow.float64(), 'HOSTGAL_DEC' ),
                'host_mag_g': ( pyarrow.float32(), 'HOSTGAL_MAG_g' ),
                'host_mag_i': ( pyarrow.float32(), 'HOSTGAL_MAG_i' ),
                'host_mag_F': ( pyarrow.float32(), 'HOSTGAL_MAG_F' ),
                'host_sn_sep': ( pyarrow.float32(), 'HOSTGAL_SNSEP' ),
                'peak_mjd': ( pyarrow.float32(), 'PEAKMJD' ),
                'peak_mag_g': ( pyarrow.float32(), 'SIM_PEAKMAG_g' ),
                'peak_mag_i': ( pyarrow.float32(), 'SIM_PEAKMAG_i' ),
                'peak_mag_F': ( pyarrow.float32(), 'SIM_PEAKMAG_F' ),
                # ROB, FIGURE OUT KAPPA
                'lens_dmu' : ( pyarrow.float32(), 'SIM_LENSDMU' ),
                'lens_dmu_applied': ( pyarrow.bool_(), '_no_header_keywrod', False ),
                'model_param_names': ( pyarrow.list_( pyarrow.string() ), None ),
                'model_param_values': ( pyarrow.list_( pyarrow.float32() ), None ),
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


    def addsn( self, headrow, spechdr, specdata, photdata ):
        spechdrrows = spechdr[ spechdr['SNID'] == headrow['SNID'] ]
        # Irritatingly, the last call will return different types if there
        #   is only one row vs. multiple rows
        if isinstance( spechdrrows, Row ):
            spechdrrows = Table( spechdrrows )

        nmjds = len( spechdrrows )
        if nmjds == 0:
            _logger.warning( f"No mjds for object {headrow['SNID']}" )
            return

        hdf5group = self.hdf5file.create_group( str( headrow['SNID'] ) )
        photgroup = hdf5group.create_group( 'photometry' )
        
        # Extract (most of) the main table data

        for key, val in OutputFile.col_map.items():
            if val[1] is not None:
                if val[1] not in headrow.keys():
                    if len(val) < 3:
                        raise RuntimeError( f'{val[1]} not in header row and no default supplied' )
                    actualval = val[2]
                else:
                    actualval = headrow[ val[1] ]
                if val[1] in OutputFile.trims:
                    self.maintable[key].append( actualval.strip() )
                else:
                    self.maintable[key].append( actualval )
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

        # Extract all the photometry

        pd0 = headrow[ 'PTROBS_MIN' ] - 1   # -1 because FITS ranges are all 1-offset
        pd1 = headrow[ 'PTROBS_MAX' ]       # No -1 since numpy ranges go one past the end
        subphot = photdata[ pd0 : pd1 ]
        # Make sure the type is what we want
        if isinstance( subphot, Row ):
            subphot = Table( subphot )
        bands = astropy.table.unique( subphot, keys=['BAND'] )[ 'BAND' ]
        photmjds = { b: numpy.array( subphot[ subphot['BAND'] == b ][ 'MJD' ] ) for b in bands }
        photdata = { b: numpy.array( subphot[ subphot['BAND'] == b ][ 'SIM_MAGOBS' ] ) for b in bands }

        h5mags = {}
        for band in photdata.keys():
            h5mags[band] = photgroup.create_group( band.strip() )
            h5mags[band].create_dataset( 'MJD', data=photmjds[ band ] )
            h5mags[band].create_dataset( 'MAG', data=photdata[ band ] )
            
        # Extract all the spectra
        # Going to assume that the lambdamin, lambdamax, and lambdabin
        # are the same for all dates for a given object

        headrow0 = spechdrrows[0]
        nlambdas = headrow0['NBIN_LAM']
        lambdas = numpy.arange( headrow0['LAMMIN']+headrow0['LAMBIN']/2., headrow0['LAMMAX'], headrow0['LAMBIN'] )
        flam = numpy.empty( ( nmjds, nlambdas ), dtype=numpy.float32 )

        # The -1 is because FITS indexes are 1-offset
        for i, headrow in enumerate(spechdrrows):
            flam[ i, : ] = specdata[ headrow['PTRSPEC_MIN']-1 : headrow['PTRSPEC_MAX'] ]['SIM_FLAM']

        h5mjds = hdf5group.create_dataset( 'mjd', data=spechdrrows['MJD'] )
        h5lambdas = hdf5group.create_dataset( 'lambda', data=lambdas )
        h5lambdas.attrs.create( 'units', 'Angstroms' )

        h5flam = hdf5group.create_dataset( 'flambda', data=flam )
        # These are the standard SNANA units
        h5flam.attrs.create( 'units', 'erg/s/Å/cm²' )

        # Add the start and end mjd to the main (summary) table

        self.maintable['start_mjd'][-1] = spechdrrows[0]['MJD']
        self.maintable['end_mjd'][-1] = spechdrrows[-1]['MJD']


    def finalize( self ):
        maintable_schema = pyarrow.schema( { k: OutputFile.col_map[k][0] for k in OutputFile.col_map.keys() } )
        self.hdf5file.close()
        pyarrowtable = pyarrow.table( self.maintable, schema=maintable_schema )
        pyarrow.parquet.write_table( pyarrowtable, self.pqfilepath )


# ======================================================================

class CustomFormatter( argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter ):
    pass

def main():
    parser = argparse.ArgumentParser( formatter_class=CustomFormatter,
                                      description="""
Convert SNANA FITS files to Parquet+HDF5 files for DESC+Roman imsim.

It will read all the *.HEAD.FITS.gz, *.PHOT.FITS.gz, and *.SPEC.FITS.gz
files from all the directories given with the -d argument.  It will
write files to the directory given in the -o argument.  There will be
two files in the output directory for each healpix that had at least one
object, snana_<healpix>.parquet and snana_<healpix>.hdf5.

The .parquet file has summary information about what objects there, one
row for each obgject.

The .hdf5 file has the actual data.  At the top level, it will have one
HDF5 group for each id (corresponding to the "id" column in the .parquet
file).  The structure under that is:

<id>: a group whose name is the object
  lambda: a ( nlambda ) array with the wavelengths
     attribute: units, string, the units ( always Angstroms )

  mjd: a ( nmjd ) array with the MJDs

  flambda: dataset, a ( nmjd , lambda ) array with fluxes
     attribute: units, string, the units ( always erg/s/Å/cm² )

  photometry: a group that contains one subgroup for each band
     <band>: a group whose name specifies the band of the photometrey
        <mjd>: a dataset, a ( nphot ) array with mjds
        <mag>: a dataset, a ( nphot ) array with magnitudes

The mjds of photometry are not necessarily the same as the mjds of
spectroscopy (which is why the group structure is what it is); in
general, there will be more epochs with photometry, as epochs that have
effectively 0 flux (indicated by a very large magnitude, over 90) don't
have spectra at all.

""" )
    parser.add_argument( '-d', '--directories', nargs='+', required=True,
                         help="Directories to find the SNANA data files (HEAD, PHOT, SPEC)" )
    # parser.add_argument( '-f', '--files', default=[], nargs='+',
    #                      help="Names of HEAD.fits[.[fg]z] files; default is to read all in directory" )
    parser.add_argument( '-n','--nside', default=32, type=int, help="nside for healpix (ring)" )
    parser.add_argument( '-o', '--outdir', default='.', help="Where to write output files." )
    parser.add_argument( '--verbose', action='store_true', default=False,
                         help="Set log level to DEBUG (default INFO)" )
    parser.add_argument( '-c', '--clobber', default=False, action='store_true', help="Overwrite existing files?" )
    args = parser.parse_args()

    if args.verbose:
        _logger.setLevel( logging.DEBUG )

    outdir = pathlib.Path( args.outdir )

    # All input files
    headfiles = []
    specfiles = []
    photfiles = []

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
        photfile = direc / f"{match.group(1)}PHOT.FITS.gz"
        if not headfile.is_file():
            raise FileNotFoundError( f"Can't read {headfile}" )
        if not specfile.is_file():
            raise FileNotFoundError( f"Can't read {specfile}" )
        if not photfile.is_file():
            raise FileNotFoundError( f"Can't read {photfile}" )
        specfiles.append( specfile )
        photfiles.append( photfile )

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
            pix = healpy.pixelfunc.ang2pix( args.nside, row['RA'], row['DEC'], lonlat=True )
            if pix not in outputfiles.keys():
                outputfiles[pix] = OutputFile( pix, outdir, clobber=args.clobber )

    _logger.info( f"Found {len(outputfiles)} different healpix" )

    _logger.debug( f'Headfiles: {[headfiles]}' )
    _logger.debug( f'Photfiles: {[specfiles]}' )

    # Now go through all of the HEAD/SPEC files, adding the relevant
    # data for each object to the correct OutputFile (based on healpix)

    _logger.info( f'Reading {len(headfiles)} SNANA HEAD/SPEC files' )

    for headfile, specfile, photfile in zip( headfiles, specfiles, photfiles ):
        _logger.info( f"Reading {headfile}" )

        head = Table.read( headfile, memmap=False )
        spechdr = Table.read( specfile, hdu=1, memmap=False )
        specdata = Table.read( specfile, hdu=2, memmap=False )
        photdata = Table.read( photfile, hdu=1, memmap=False )

        head['SNID'] = head['SNID'].astype( numpy.int64 )
        spechdr['SNID'] = spechdr['SNID'].astype( numpy.int64 )

        _logger.info( f"...done reading, processing {len(head)} objects." )
        for row in head:
            # phi = numpy.radians( 360. - row['RA'] )
            # theta = numpy.radians( 90. - row['DEC'] )
            # if ( theta < 0 ) or ( theta > numpy.pi ):
            #     raise ValueError( f'Bad dec for {snid} : {row["DEC"]}' )
            # if ( phi >= 2*numpy.pi ): phi -= 2*numpy.pi
            # if ( phi < 0 ): phi += 2*numpy.pi
            # pix = healpy.pixelfunc.ang2pix( args.nside, theta, phi )

            pix = healpy.pixelfunc.ang2pix( args.nside, row['RA'], row['DEC'], lonlat=True )

            outputfiles[pix].addsn( row, spechdr, specdata, photdata )

    # Close out all the files

    for pix, outputfile in outputfiles.items():
        outputfile.finalize()

    _logger.info( "Done." )

# ======================================================================

if __name__ == "__main__":
    main()


