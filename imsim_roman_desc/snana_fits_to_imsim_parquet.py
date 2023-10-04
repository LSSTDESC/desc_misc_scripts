# For format of the output parquet and hdf5 files, see:
# https://docs.google.com/document/d/1zlJ7AsFjwnVYlkqcoXq1G6YUXJDOn7kMP8Jbl6SGRxk/edit#heading=h.hlzr61r0h2en

import sys
import io
import re
import copy
import pathlib
import logging
import multiprocessing
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

class HealPixProcessor:
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


    def __init__( self, pix, nside, outdir, headfiles, photfiles, specfiles, clobber=False ):
        self.pix = pix
        self.nside = nside
        self.outdir = pathlib.Path( outdir )
        self.hdf5filepath = outdir / f'snana_{pix}.hdf5'
        self.pqfilepath = outdir / f'snana_{pix}.parquet'

        self.headfiles = headfiles
        self.photfiles = photfiles
        self.specfiles = specfiles
        self.clobber = clobber

        self.logger = logging.getLogger( str(pix) )
        logout = logging.FileHandler( f'{pix}.log' )
        formatter = logging.Formatter( f'[%(asctime)s - %(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S' )
        logout.setFormatter( formatter )
        self.logger.addHandler( logout )
        self.logger.setLevel( logging.INFO )


    def go( self ):
        self.logger.info( f"Starting process for heaplix {self.pix}" )

        if self.pqfilepath.exists():
            if self.clobber:
                self.pqfilepath.unlink()
            else:
                raise RuntimeError( f'{self.pqfilepath} exists, not clobbering.' )
        if self.hdf5filepath.exists():
            if self.clobber:
                self.hdf5filepath.unlink()
            else:
                raise RuntimeError( f'{self.hdf5filepath} exists, not clobber.' )

        self.maintable = { k: [] for k in self.col_map.keys() }
        self.hdf5file = h5py.File( self.hdf5filepath, 'w' )

        for n, headfile, specfile, photfile in zip( range(len(self.headfiles)),
                                                    self.headfiles,
                                                    self.specfiles,
                                                    self.photfiles ):
            self.logger.info( f"Reading {n+1} of {len(self.headfiles)} : {headfile}" )
            head = Table.read( headfile, memmap=False )
            spechdr = Table.read( specfile, hdu=1, memmap=False )
            photdata = Table.read( photfile, hdu=1, memmap=True )

            head['SNID'] = head['SNID'].astype( numpy.int64 )
            spechdr['SNID'] = spechdr['SNID'].astype( numpy.int64 )

            self.logger.info( f"...done reading {len(head)} objects from header" )
            self.nwrote = 0

            for row in head:
                pix = healpy.pixelfunc.ang2pix( self.nside, row['RA'], row['DEC'], lonlat=True )
                if pix == self.pix:
                    self.addsn( row, spechdr, specfile, photdata )
                    self.nwrote += 1

            self.logger.info( f"...{self.nwrote} of {len(head)} objects were in healpix {self.pix}" )

        self.logger.info( f"Writing Parquet file for healpix {self.pix}" )
        self.finalize()
        self.logger.info( f"Healpix {self.pix} Done" )


    def addsn( self, headrow, spechdr, specfile, photdata ):
        spechdrrows = spechdr[ spechdr['SNID'] == headrow['SNID'] ]
        # Irritatingly, the last call will return different types if there
        #   is only one row vs. multiple rows
        if isinstance( spechdrrows, Row ):
            spechdrrows = Table( spechdrrows )

        nmjds = len( spechdrrows )
        if nmjds == 0:
            self.logger.warning( f"No mjds for object {headrow['SNID']}" )
            return

        hdf5group = self.hdf5file.create_group( str( headrow['SNID'] ) )
        photgroup = hdf5group.create_group( 'photometry' )

        # Extract (most of) the main table data

        for key, val in self.col_map.items():
            if val[1] is not None:
                if val[1] not in headrow.keys():
                    if len(val) < 3:
                        raise RuntimeError( f'{val[1]} not in header row and no default supplied' )
                    actualval = val[2]
                else:
                    actualval = headrow[ val[1] ]
                if val[1] in self.trims:
                    self.maintable[key].append( actualval.strip() )
                else:
                    self.maintable[key].append( actualval )
            else:
                self.maintable[key].append( None )

        # Get out the model params

        modelname = headrow['SIM_MODEL_NAME'].strip()
        params = dict( self.general_params )
        if modelname in self.model_params.keys():
            params.update( self.model_params[ modelname ] )
        else:
            if modelname not in self.seen_unknown_models:
                self.logger.warning( f"Unknown model {modelname}, just saving general parameters" )
                self.seen_unknown_models.add( modelname )
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
        specdata = Table.read( specfile, hdu=2, memmap=True )
        for i, headrow in enumerate(spechdrrows):
            specsubdata = specdata[ headrow['PTRSPEC_MIN']-1 : headrow['PTRSPEC_MAX'] ]
            flam[ i, : ] = specsubdata['SIM_FLAM']

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
        maintable_schema = pyarrow.schema( { k: self.col_map[k][0] for k in self.col_map.keys() } )
        self.hdf5file.close()
        pyarrowtable = pyarrow.table( self.maintable, schema=maintable_schema )
        pyarrow.parquet.write_table( pyarrowtable, self.pqfilepath )


# ======================================================================

def collect_files( directories ):
    headfiles = []
    specfiles = []
    photfiles = []

    for direc in directories:
        direc = pathlib.Path( direc )
        localheadfiles = list( direc.glob( '*HEAD.FITS.gz' ) )
        headfiles.extend( localheadfiles )

    headre = re.compile( '^(.*)HEAD\.FITS\.gz' )
    for headfile in headfiles:
        direc = headfile.parent
        match = headre.search( headfile.name )
        if match is None:
            raise ValueError( f"Failed to parse {headfile.name} for *HEAD.FITS.gz" )
        specfile = direc / f"{match.group(1)}SPEC.FITS"
        photfile = direc / f"{match.group(1)}PHOT.FITS.gz"
        if not headfile.is_file():
            raise FileNotFoundError( f"Can't read {headfile}" )
        if not specfile.is_file():
            raise FileNotFoundError( f"Can't read {specfile}" )
        if not photfile.is_file():
            raise FileNotFoundError( f"Can't read {photfile}" )
        specfiles.append( specfile )
        photfiles.append( photfile )

    return headfiles, photfiles, specfiles

# ======================================================================

def find_healpix( headfiles, nside ):
    headfiledata = {}
    allpix = []

    _logger.info( f'Reading {len(headfiles)} SNANA HEAD files to find healpix' )

    for headfile in headfiles:
        _logger.info( f"Reading {headfile}" )

        head = Table.read( headfile, memmap=False )
        _logger.info( f"...done reading, processing {len(head)} objects." )
        for row in head:
            pix = healpy.pixelfunc.ang2pix( nside, row['RA'], row['DEC'], lonlat=True )
            if pix not in allpix:
                allpix.append( pix )

    return allpix

# ======================================================================

class CustomFormatter( argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter ):
    pass

def main():
    parser = argparse.ArgumentParser( formatter_class=CustomFormatter,
                                      description="""
Convert SNANA FITS files to Parquet+HDF5 files for DESC+Roman imsim.

Will launch one process for each healpix that it needs to write, so make
sure that you have at least that many plus one (for the master process)
CPUs available.  Run with -j to just count the healpix (a relatively
fast operation) to figure this number out.

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
    parser.add_argument( '-j', '--just-count-healpix', default=False, action='store_true',
                         help="Only read HEAD files to count healpix, don't do translation" )
    args = parser.parse_args()

    if args.verbose:
        _logger.setLevel( logging.DEBUG )

    outdir = pathlib.Path( args.outdir )

    headfiles, photfiles, specfiles = collect_files( args.directories )

    # Make one pass through the HEAD files to figure out which healpix
    # we need.  

    healpix = find_healpix( headfiles, args.nside ) # , sharedmanager )

    if args.just_count_healpix:
        _logger.info( f"Found {len(healpix)} different healpix; stopping." )
        return
    else:
        _logger.info( f"Found {len(healpix)} different healpix, launching that many processes." )

    # Launch a process for each healpix
    # TODO: make this a pool instead and make the
    # number of processes an argument.

    procs = []
    for pix in healpix:
        hpp = HealPixProcessor( pix, args.nside, outdir, headfiles, photfiles, specfiles, args.clobber )
        proc = multiprocessing.Process( target=hpp.go )
        proc.start()
        procs.append( proc )

    # Wait for all the processes to finish

    for proc in procs:
        proc.join()

    _logger.info( "All done." )

# ======================================================================

if __name__ == "__main__":
    main()


