# For format of the output parquet and hdf5 files, see:
# https://docs.google.com/document/d/1zlJ7AsFjwnVYlkqcoXq1G6YUXJDOn7kMP8Jbl6SGRxk/edit#heading=h.hlzr61r0h2en

import sys
import io
import re
import copy
import pathlib
import logging
import traceback
import multiprocessing
import multiprocessing.pool
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
#
# Will be run in parallel

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

    # Make of columns from the HEAD file to the Parquet file
    col_map = { 'id': ( pyarrow.int64(), 'SNID' ),
                'ra': ( pyarrow.float64(), 'RA' ),
                'dec': ( pyarrow.float64(), 'DEC' ),
                'host_id': ( pyarrow.int64(), 'HOSTGAL_OBJID' ),
                'gentype': ( pyarrow.int16(), 'SIM_TYPE_INDEX' ),
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
    # Columns we have to do a string trim on
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
        self.logger.propagate = False
        logout = logging.FileHandler( f'snana_to_pq+hdf5_healpix_{pix}.log' )
        formatter = logging.Formatter( f'[%(asctime)s - %(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S' )
        logout.setFormatter( formatter )
        self.logger.addHandler( logout )
        self.logger.setLevel( logging.INFO )


    def go( self ):
        self.logger.info( f"Starting process for heaplix {self.pix}" )
        try:
            if self.pqfilepath.exists():
                if self.clobber:
                    self.pqfilepath.unlink()
                else:
                    self.logger.error( f'{self.pqfilepath} exists, not clobbering.' )
                    return False
            if self.hdf5filepath.exists():
                if self.clobber:
                    self.hdf5filepath.unlink()
                else:
                    self.logger.error( f'{self.hdf5filepath} exists, not clobber.' )
                    return False

            self.maintable = { k: [] for k in self.col_map.keys() }
            self.hdf5file = h5py.File( self.hdf5filepath, 'w' )

            for n, headfile, specfile, photfile in zip( range(len(self.headfiles)),
                                                        self.headfiles,
                                                        self.specfiles,
                                                        self.photfiles ):
                self.logger.info( f"Reading {n+1} of {len(self.headfiles)} : {headfile}" )

                head = Table.read( headfile, memmap=False )
                spechdr = Table.read( specfile, hdu=1, memmap=False )
                head['SNID'] = head['SNID'].astype( numpy.int64 )
                spechdr['SNID'] = spechdr['SNID'].astype( numpy.int64 )

                self.logger.info( f"...done reading {len(head)} objects from header" )
                self.nwrote = 0

                for row in head:
                    pix = healpy.pixelfunc.ang2pix( self.nside, row['RA'], row['DEC'], lonlat=True )
                    if pix == self.pix:
                        self.addsn( row, spechdr, specfile, photfile )
                        self.nwrote += 1

                self.logger.info( f"...{self.nwrote} of {len(head)} objects were in healpix {self.pix}" )

            self.logger.info( f"Writing Parquet file for healpix {self.pix}" )
            self.finalize()
            self.logger.info( f"Healpix {self.pix} Done" )
            return True

        except Exception as e:
            self.logger.error( f"Exception: {traceback.format_exc()}" )
            return False

    def addsn( self, headrow, spechdr, specfile, photfile ):
        """Add a single SN (specified by a header row) to hdf5 files and self.maintable"""

        # Extract (most of) the main table data, appending to self.maintable

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

        # MAGNITUDE CORRECTION STUFF
        # 
        # Spectra are only binned at 100Å, which is too course for the
        # precision we want.  As such, we need to apply a binning
        # correction.  The PHOT files have magnitudes (where? top of
        # atmosphere?) that are calculated from a much finer binning
        # (that doesn't show up in the spec file).  Those are in the
        # SIM_MAGOBS column in the phot file, indexed by MJD and BAND.
        # The SPEC file has synthetic photometry that comes from
        # integrating the coarse spectra; those show up in the
        # SIM_SYNMAG_[band} fields.
        #
        # The correction that needs to be applied is
        #   magcor = SIM_MAGOBS - SIM_SYNMAG_[band]

        # Pull out the spectrum header rows for further processing

        spechdrrows = spechdr[ spechdr['SNID'] == headrow['SNID'] ]
        if len( spechdrrows ) == 0:
            self.logger.warning( f"No mjds for object {headrow['SNID']}" )
            return
        # Irritatingly, subsetting of an astropy Table will return
        # different things based on whether the subset has only one row
        # vs. multiple rows
        if isinstance( spechdrrows, Row ):
            spechdrrows = Table( spechdrrows )
        spechdrrows = spechdrrows.to_pandas().set_index( 'MJD' )

        # Extract photometry

        pd0 = headrow[ 'PTROBS_MIN' ] - 1   # -1 because FITS ranges are all 1-offset
        pd1 = headrow[ 'PTROBS_MAX' ]       # No -1 since numpy ranges go one past the end
        photdata = Table.read( photfile, hdu=1, memmap=True )
        subphot = photdata[ pd0 : pd1 ]
        photdata = None
        # Make sure the type is what we want
        if isinstance( subphot, Row ):
            subphot = Table( subphot )
        subphot = subphot.to_pandas()
        subphot['BAND'] = subphot['BAND'].apply( lambda x: x.decode('ASCII').strip() )
        bands = subphot['BAND'].unique()
        subphot = subphot.pivot( index='MJD', columns='BAND' )
        subphot.columns = subphot.columns.map( "_".join )

        # Join to spectroscopy so that we only have MJDs where there is
        # both photometry and spectroscopy

        magdata = spechdrrows.merge( subphot, left_index=True, right_index=True, how='inner' )

        # Extract all the spectra
        # Going to assume that the lambdamin, lambdamax, and lambdabin
        # are the same for all dates for a given object

        # Pandas can be so annoying: doing .iloc turns all integers into 64-bit floats
        # headrow0 = magdata.iloc[0]
        # lambdas = numpy.arange( headrow0['LAMMIN']+headrow0['LAMBIN']/2., headrow0['LAMMAX'], headrow0['LAMBIN'] )
        mjd0 = magdata.index.values[0]
        nlambdas = magdata.loc[ mjd0, 'NBIN_LAM' ]
        lammin = magdata.loc[ mjd0, 'LAMMIN' ]
        lammax = magdata.loc[ mjd0, 'LAMMAX' ]
        lambin = magdata.loc[ mjd0, 'LAMBIN' ]
        lambdas = numpy.arange( lammin + lambin/2., lammax, lambin )
        flam = numpy.empty( ( len(magdata), nlambdas ), dtype=numpy.float32 )

        # The -1 is because FITS indexes are 1-offset
        specdata = Table.read( specfile, hdu=2, memmap=True )
        for i, row in enumerate( magdata.itertuples() ):
            specsubdata = specdata[ row.PTRSPEC_MIN-1 : row.PTRSPEC_MAX ]
            flam[ i, : ] = specsubdata['SIM_FLAM']
        specdata = None

        # ****
        # HACK ALERT -- Some of the flams were coming up with NaN.
        # This was causing trouble down the line.  For now, we're going
        # to assume that these are small enough fluxes that we can
        # just ignore them, and set them to 0
        wnan = ( numpy.isnan( flam ) )
        if wnan.any():
            _logger.warning( f"Got {wnan.sum()} NaN flam values for SN {headrow['SNID']}, setting them to zero" )
            flam[wnan] = 0.
        # ****

        # Turn these into the HDF5 data structures

        hdf5group = self.hdf5file.create_group( str( headrow['SNID'] ) )
        h5mags = {}
        h5synmags = {}
        h5magcor = {}
        h5mjds = hdf5group.create_dataset( 'mjd', data=magdata.index.values )
        for band in bands:
            h5mags[band] = hdf5group.create_dataset( f'mag_{band}', data=magdata[f'SIM_MAGOBS_{band}'] )
            h5synmags[band] = hdf5group.create_dataset( f'synmag_{band}', data=magdata[f'SIM_SYNMAG_{band}'] )
            magcor = magdata[f'SIM_MAGOBS_{band}'] - magdata[f'SIM_SYNMAG_{band}']
            if numpy.any( ( magcor < -0.2 ) | ( magcor > 0.2 ) ):
                # w = numpy.where( ( magcor < -0.2 ) | ( magcor > 0.2 ) )
                self.logger.warning( f'Band {band}, SNID {headrow["SNID"]} has some extreme magcors!' )
            h5magcor[band] = hdf5group.create_dataset(
                f'magcor_{band}',
                data=magdata[f'SIM_MAGOBS_{band}'] - magdata[f'SIM_SYNMAG_{band}']
            )
        h5lambdas = hdf5group.create_dataset( 'lambda', data=lambdas )
        h5lambdas.attrs.create( 'units', 'Angstroms' )
        h5flam = hdf5group.create_dataset( 'flambda', data=flam )
        h5flam.attrs.create( 'units', 'erg/s/Å/cm²' )    # Standard SNANA units

        # Add the start and end mjd to the main (summary) table

        self.maintable['start_mjd'][-1] = magdata.index[0]
        self.maintable['end_mjd'][-1] = magdata.index[-1]


    def finalize( self ):
        """Write the parquet file from self.maintable"""

        maintable_schema = pyarrow.schema( { k: self.col_map[k][0] for k in self.col_map.keys() } )
        self.hdf5file.close()
        pyarrowtable = pyarrow.table( self.maintable, schema=maintable_schema )
        pyarrow.parquet.write_table( pyarrowtable, self.pqfilepath )


# ======================================================================

def collect_files( directories ):
    headfiles = []
    specfiles = []
    photfiles = []

    headre = re.compile( '^(?P<filebase>.*)HEAD\.FITS(?P<gzip>\.gz)?' )

    for direc in directories:
        _logger.info( f"Looking for files in {direc}" )
        direc = pathlib.Path( direc )
        if not direc.is_dir():
            _logger.warning( f"{direc} isn't a directory, skipping it." )
            continue

        # Collect HEAD files
        for localfile in direc.iterdir():
            match = headre.search( localfile.name )
            if match is not None:
                if localfile.is_file():
                    headfiles.append( localfile )
                else:
                    _logger.warning( f"{localfile} isn't a regular file, skipping it" )

    # Weed out gzipped equivalents of non-gzipped files
    keptheadfiles = []
    for headfile in headfiles:
        match = headre.search( headfile.name )
        if match.group( 'gzip' ) == '.gz':
            if ( headfile.parent / f"{match.group('filebase')}HEAD.FITS" ) not in headfiles:
                keptheadfiles.append( headfile )
        else:
            keptheadfiles.append( headfile )
    headfiles = keptheadfiles

    # Sort for sanity
    headfiles.sort()

    headfiles = keptheadfiles
    _logger.info( f"Found {len(headfiles)} HEAD files." )
    _logger.debug( headfiles )


    # Make sure we have SPEC and PHOT files
    missingspec = []
    missingphot = []
    for headfile in headfiles:
        direc = headfile.parent
        match = headre.search( headfile.name )
        filebase = match.group( 'filebase' )

        specfile = direc / f"{filebase}SPEC.FITS"
        if not specfile.is_file():
            _logger.error( f"Couldn't find {specfile}" )
            missingspec.append( specfile )
        specfiles.append( specfile )

        nongzphotfile = direc / f"{filebase}PHOT.FITS"
        if nongzphotfile.is_file():
            photfiles.append( nongzphotfile )
        else:
            gzphotfile = direc / f"{filebase}PHOT.FITS.gz"
            if not gzphotfile.is_file():
                _logger.error( f"Couldn't find {nongzphotfile}[.gz]" )
                missingphot.append( nongzphotfile )
            photfiles.append( gzphotfile )

    if ( len( missingphot ) != 0 ) or ( len( missingspec ) != 0 ):
        raise FileNotFoundError( f"Failed to find {len(missingspec)} SPEC files and "
                                 f"{len(missingphot)} PHOT files." )

    return headfiles, photfiles, specfiles

# ======================================================================

def find_healpix( headfiles, nside ):
    headfiledata = {}
    allpix = []

    _logger.info( f'Reading {len(headfiles)} SNANA HEAD files to find healpix' )

    for headfile in headfiles:
        _logger.info( f"Finding healpix in {headfile}" )

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

def proclauncher( pix, nside, outdir, headfiles, photfiles, specfiles, clobber ):
    hpp = HealPixProcessor( pix, nside, outdir, headfiles, photfiles, specfiles, clobber )
    return hpp.go()

def main():
    parser = argparse.ArgumentParser( formatter_class=CustomFormatter,
                                      description="""
Convert SNANA FITS files to Parquet+HDF5 files for DESC+Roman imsim.

Will launch one process for each healpix that it needs to write (unless
you explicitly specify the number of processes with --processes), so
make sure that you have at least that many plus one (for the master
process) CPUs available.  Run with -j to just count the healpix (a
relatively fast operation) to figure out how many heal pix will be written.

It will read all the *.HEAD.FITS[.gz], *.PHOT.FITS[.gz], and
*.SPEC.FITS[.gz] files from all the directories given with the -d
argument.  It will write files to the directory given in the -o
argument.  There will be two files in the output directory for each
healpix that had at least one object, snana_<healpix>.parquet and
snana_<healpix>.hdf5.

The .parquet file has summary information about what objects there, one
row for each obgject.

The .hdf5 file has the actual data.  At the top level, it will have one
HDF5 group for each id (corresponding to the "id" column in the .parquet
file).  The structure under that is:

<id>: a group whose name is the object
  mjd: a ( nmjd ) array with the MJDs

  lambda: a ( nlambda ) array with the wavelengths at the *center* of the bins
     attribute: units, string, the units ( always Angstroms )

  flambda: dataset, a ( nmjd , lambda ) array with fluxes
     attribute: units, string, the units ( always erg/s/Å/cm² )

  mag_<band>: multiple datasets, an ( nmjd ) array for each photometric
     band.  Has the top-of-atmosphere (CHECK THIS) magnitudes calculated
     by SNANA (using high resolution spectroscopic binning).

  synmag_<band>: multiple datasets, an ( nmjd ) array for each
     photometric band.  Has synthetic photometry calculated from the
     coarse-λ SED as it appears in flambda.

  magcor_<band>: multiple datasets, an ( nmjd ) array for each
     photometric band.  Has the magnitude correction that should be
     applied for this band to any synthetic photometry calculated from
     any processing of the SED.  This field is redundant with the
     previous two files, as magcor_<band> = mag_<band> - synmag_<band>

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
    parser.add_argument( '-p', '--processes', default=0, type=int,
                         help=( "Number of subprocesses to launch; default is number of healpix that will "
                                "be written.  Number of CPUs available must be at least this number plus 1." ) )
    args = parser.parse_args()

    if args.verbose:
        _logger.setLevel( logging.DEBUG )

    headfiles, photfiles, specfiles = collect_files( args.directories )

    # Make one pass through the HEAD files to figure out which healpix
    # we need.  

    healpix = find_healpix( headfiles, args.nside ) # , sharedmanager )

    if args.just_count_healpix:
        strio = io.StringIO()
        strio.write( f"Found {len(healpix)} different healpix; stopping.\n" )
        strio.write( f"Run with {len(healpix)+1} cpus per task and {3*(len(healpix)+1)} GB of memory\n" )
        strio.write( f"#SBATCH --cpus-per-task={len(healpix)+1}\n" )
        strio.write( f"#SBATCH --mem={3*(len(healpix)+1)}G\n" )
        _logger.info( strio.getvalue() )
        return
    else:
        _logger.info( f"Found {len(healpix)} different healpix, launching that many processes." )

    outdir = pathlib.Path( args.outdir )
    if outdir.exists() and not outdir.is_dir():
        raise FileExistsError( f"outdir {outdir} exists and is not a directory" )
    if not outdir.exists():
        outdir.mkdir( exist_ok=True, parents=True )

    # Launch subprocesses to process healpix
    nprocs = len(healpix) if args.processes == 0 else args.processes
    pool = multiprocessing.pool.Pool( nprocs )

    poolres = []
    for pix in healpix:
        poolres.append(
            pool.apply_async(
                proclauncher, args=(pix, args.nside, outdir, headfiles, photfiles, specfiles, args.clobber)
            )
        )

    # Wait for all the processes to finish and check for errors
    pool.close()
    pool.join()

    exceptions = []
    fails = []
    for pix, res in zip( healpix, poolres ):
        if not res.successful():
            exceptions.append( pix )
        if not res.get():
            fails.append( pix )

    if len(exceptions) > 0:
        _logger.error( f"The following healpix threw an exception: {exceptions}" )
    if len(fails) > 0:
        _logger.error( f"The following healpix failed: {fails}" )

    _logger.info( "All done." )

# ======================================================================

if __name__ == "__main__":
    main()


