import sys
import re
import gzip
import math
import logging
import argparse
import pathlib

import numpy
import pandas

import healpy
import astropy.units
import astropy.table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

_logger = logging.getLogger( __name__ )
if not _logger.hasHandlers():
    _logout = logging.StreamHandler( sys.stderr )
    _logger.addHandler( _logout )
    _logout.setFormatter( logging.Formatter( f'[%(asctime)s - %(levelname)s] - %(message)s',
                                             datefmt='%Y-%m-%d %H:%M:%S' ) )
_logger.setLevel( logging.INFO )

class Image:
    def __init__( self, imagepath, centroid, snana_dirs ):
        self.hdul = None
        self.imagepath = imagepath
        self.centroid = centroid
        self.snana_dirs = snana_dirs
        self.nside = 32

        self.hdul = fits.open( self.imagepath )
        self.mjd = float( self.hdul[0].header['MJD'] )
        self.band = self.hdul[0].header['FILTER'].strip()
        self.wcs = WCS( self.hdul[0].header )
        self.xsize = self.hdul[0].data.shape[1]
        self.ysize = self.hdul[0].data.shape[0]
        corners = self.wcs.pixel_to_world( ( 0, 0, self.xsize-1, self.xsize-1 ),
                                           ( 0, self.ysize-1, 0, self.ysize-1 ) )
        self.ra00 = corners[0].ra.to(astropy.units.deg).value
        self.ra01 = corners[1].ra.to(astropy.units.deg).value
        self.ra10 = corners[2].ra.to(astropy.units.deg).value
        self.ra11 = corners[3].ra.to(astropy.units.deg).value
        self.dec00 = corners[0].dec.to(astropy.units.deg).value
        self.dec01 = corners[1].dec.to(astropy.units.deg).value
        self.dec10 = corners[2].dec.to(astropy.units.deg).value
        self.dec11 = corners[3].dec.to(astropy.units.deg).value
        self.minra = min( self.ra00, self.ra01, self.ra10, self.ra11 )
        self.maxra = max( self.ra00, self.ra01, self.ra10, self.ra11 )
        self.mindec = min( self.dec00, self.dec01, self.dec10, self.dec11 )
        self.maxdec = max( self.dec00, self.dec01, self.dec10, self.dec11 )
        
        pix00 = healpy.pixelfunc.ang2pix( self.nside, self.ra00, self.dec00, lonlat=True )
        pix01 = healpy.pixelfunc.ang2pix( self.nside, self.ra01, self.dec01, lonlat=True )
        pix10 = healpy.pixelfunc.ang2pix( self.nside, self.ra10, self.dec10, lonlat=True )
        pix11 = healpy.pixelfunc.ang2pix( self.nside, self.ra11, self.dec11, lonlat=True )

        if ( pix00 != pix01 ) or ( pix00 != pix10 ) or ( pix00 != pix11 ):
            _logger.warning( f"Multiple healpix on image!" )

        if centroid:
            match = re.search( '^(amp|eimage)(.*)\.fits(\.fz)?$', imagepath.name )
            if match is None:
                raise ValueError( f"Failed to parse {imagepath.name}" )
            centroidfile = imagepath.parent / f"centroid{match.group(2)}.txt"
            if not centroidfile.is_file():
                centroidfile = imagepath.parent / f"centroid{match.group(2)}.txt.gz"
                if not centroidfile.is_file():
                    raise FileNotFoundError( f"Failed to find centroidfile {centroidfile.name} (with and without .gz" )
            else:
                raise NotImplementedError( "I'm foolishly assuming gzipped centroid files right now." )

            # Gotta strip the # from the begining of the
            # header line.  I'm kind of boggled that
            # pandas.read_csv doesn't have an option to just
            # do this.  Maybe it does, and I failed to find it?
            with gzip.open( centroidfile, 'rt' ) as ifp:
                header = ifp.readline().strip()
            match = re.search('^ *#? *(.*)$', header )
            if match is None:
                raise ValueError( f"Failed to parse header line {header}" )
            columns = match.group(1).split()
            self.centroiddata = pandas.read_csv( centroidfile, delim_whitespace=True,
                                                 header=None, skiprows=1, names=columns )

        else:
            raise NotImplementedError( "only know centroid files so far" )

    def load_snana_mags( self ):
        heads = []
        headre = re.compile( "^(?P<base>.*)_HEAD.FITS(?P<gz>\.gz)?$" )
        for direc in self.snana_dirs:
            direc = pathlib.Path( direc )
            for f in direc.glob( "*_HEAD.FITS*" ):
                match = headre.search( f.name )
                if match is None:
                    raise ValueError( f"Failed to parse {f.name} for .*_HEAD.FITS(\.gz)?" )
                if match.group('gz') is not None:
                    raise ValueError( f"OMG gzip file {match.group('gz')}" )
                heads.append( f )

        self.snanahead = None
        for headfile in heads:
            _logger.info( f"Reading {headfile}..." )
            tab = astropy.table.Table.read( headfile, hdu=1 )
            tab = tab.to_pandas()
            tab.SNID = tab.SNID.astype( numpy.int64 )
            tab = tab[ ( tab['RA'] >= self.minra ) & ( tab['RA'] <= self.maxra ) &
                       ( tab['DEC'] >= self.mindec ) & ( tab['DEC'] <= self.maxdec ) ]
            if len(tab) == 0:
                continue
            tab['healpix'] = tab.apply( lambda row: healpy.ang2pix( 32, row.RA, row.DEC, lonlat=True ), axis=1 )
            sc = SkyCoord( tab.RA.values, tab.DEC.values, frame='icrs', unit='deg' )
            x, y = self.wcs.world_to_pixel( sc )
            tab['x'] = x
            tab['y'] = y
            tab = tab[ ( tab.x >= 0 ) & ( tab.x < self.xsize ) & ( tab.y >=0 ) & ( tab.y <= self.ysize ) ]
            if len( tab ) > 0:
                tab['headfile'] = str( headfile )
                if self.snanahead is None:
                    self.snanahead = tab
                else:
                    self.snanahead = pandas.concat( [ self.snanahead, tab ] )

        _logger.info( f'{len(self.snanahead)} SNANA objects are on this field' )
        
        # Now go through and read all the corresponding PHOT files,
        # extracting all the magnitudes at the time of observation

        mags = {}
        hasmags = numpy.repeat( False, len(self.snanahead) )
        rowdex = 0
        for row in self.snanahead.itertuples():
            photfile = row.headfile.replace( "_HEAD.FITS", "_PHOT.FITS" )
            tab = astropy.table.Table.read( photfile, hdu=1, memmap=True )
            tabrows = tab[ row.PTROBS_MIN-1 : row.PTROBS_MAX ]
            tab = None
            tabrows = tabrows.to_pandas()
            tabrows['BAND'] = tabrows['BAND'].apply( lambda x : x.decode('utf-8').strip() )
            bands = tabrows['BAND'].unique()
            for band in bands:
                if band not in mags.keys():
                    mags[band] = numpy.repeat( -9., len(self.snanahead) )
                magrows = tabrows[ tabrows.BAND == band ]
                if ( magrows.MJD.min() <= self.mjd ) and ( magrows.MJD.max() >= self.mjd ):
                    mag = None
                    for i in range(len(magrows)-1):
                        if ( magrows.iloc[i].MJD <= self.mjd ) and ( magrows.iloc[i+1].MJD > self.mjd ):
                            # ... I bet pandas has some nice built in linear interpolation thing
                            # Or, I could be clever and write a few pandas lines to do this in
                            # all a vetor way.  But, I'm not clever, not now.
                            mag = ( magrows.iloc[i].SIM_MAGOBS +
                                    ( ( magrows.iloc[i+1].SIM_MAGOBS - magrows.iloc[i].SIM_MAGOBS ) /
                                      ( magrows.iloc[i+1].MJD - magrows.iloc[i].MJD ) )
                                    * ( self.mjd - magrows.iloc[i].MJD  ) )
                            break
                    if mag is None:
                        if magrows.iloc[-1].MJD != self.mjd:
                            _logger.error( "This should never happen." )
                            import pdb; pdb.set_trace()
                            pass
                        mag = magrows.iloc[-1].SIM_MAGOBS
                    hasmags[rowdex] = True
                    mags[band][rowdex] = mag
            rowdex += 1

        for band in mags.keys():
            self.snanahead[f'mag_{band}'] = mags[band]

        self.snanahead = self.snanahead[ hasmags ]
        _logger.info( f'{len(self.snanahead)} SNANA objects on this field have a mag at this MJD' )
        
    def print_centroid_comparison( self ):
        print( f'File {self.imagepath.name} , filter {self.band}, mjd {self.mjd}' )
        print( f'     SNID  SIMOBS_MAG  realized_flux     zp' )
        for row in self.snanahead.itertuples():
            snid = row.SNID
            mag = getattr( row, f'mag_{self.band}' )
            w = ( self.centroiddata.object_id == snid )
            if w.sum() == 0:
                print( f'{snid:9d}    {mag:7.3f}            ---' )
            else:
                flux = self.centroiddata[ w ].iloc[0].realized_flux
                zp = mag + 2.5*math.log10( flux )
                print( f'{snid:9d}    {mag:7.3f}   {flux:12.3g}   {zp:5.2f}' )

        snana_missing = []
        lowobjid = self.centroiddata[ self.centroiddata.object_id < 100000000 ]
        for row in lowobjid.itertuples():
            if not ( self.snanahead.SNID == row.object_id ).any():
                if ( row.x >= 0 ) and ( row.x < self.xsize ) and ( row.y >= 0 ) and ( row.y < self.ysize ):
                    snana_missing.append( row.object_id )

        if len(snana_missing) > 0:
            print( f"\nMissing from SNANA list: {snana_missing}" )
                
    def finalize( self ):
        if self.hdul is not None:
            self.hdul.close()
            self.hdul = None

    def __del__( self ):
        self.finalize()

# ----------------------------------------------------------------------
    
class CustomFormatter( argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter ):
    pass

def main():
    parser = argparse.ArgumentParser( formatter_class=CustomFormatter,
                                      description="Do things" )
    parser.add_argument( '-s', '--snana-dirs', nargs='+', required=True,
                         help="Directories with SNANA data files (HEAD, PHOT, SPEC)" )
    parser.add_argument( '-d', '--image-dirs', nargs='+', required=True,
                         help="Directories with images" )
    parser.add_argument( '-i', '--image', required=True,
                         help="Name of the image (somewhere in image-dirs) to process" )
    parser.add_argument( '--skycat-dir', default='/global/cfs/cdirs/descssim/imSim/skyCatalogs_v2',
                         help="Where the galaxy flux and pointsource parquet files are" )
    parser.add_argument( '-c', '--centroid', action='store_true', default=False,
                         help=( "If true, truth is in Jim-style centroid files.  "
                                "If false, truth is in Troxel-style truth files." ) )
    args = parser.parse_args()

    for direc in args.image_dirs:
        direc = pathlib.Path( direc )
        impath = pathlib.Path( direc / args.image )
        if not impath.is_file():
            raise FileNotFoundError( f"Can't find file {impath}" )

    imager = Image( impath, args.centroid, args.snana_dirs )
    imager.load_snana_mags()
    imager.print_centroid_comparison()
    
# ----------------------------------------------------------------------

if __name__ == "__main__":
    main()
