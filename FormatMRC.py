"""Classes for Formats that recognise data stored in the MRC format, an open
standard used in electron microscopy
(http://www.ccpem.ac.uk/mrc_format/mrc2014.php)"""

from __future__ import absolute_import, division, print_function

import logging
import os
import struct
from math import sqrt
from scitbx import matrix
from scitbx.array_family import flex
from dxtbx.format.Format import Format
from dxtbx.model import ScanFactory
from dxtbx.format.FormatMultiImage import FormatMultiImage
import mrcfile

logger = logging.getLogger("dials")

class FormatMRC(Format):
    @staticmethod
    def understand(image_file):
        try:
            mrc = mrcfile.mmap(image_file)
        except ValueError:
            return False
        return True

    def _start(self):
        """Open the MRC file, read the metadata into an internal dictionary
        self._header_dictionary, add FEI extended metadata if available"""

        with mrcfile.mmap(self._image_file) as mrc:
            h = mrc.header
            #xh = mrc.extended_header
        self._header_dictionary = self._unpack_header(h)

        if h['exttyp'].tobytes() == b'FEI1':
            xh = self._read_ext_header(self._image_file)
            self._header_dictionary.update(xh)

        # Add a positive pedestal level to images to avoid negative
        # pixel values if a value is set by the environment variable
        # ADD_PEDESTAL. This is a nasty workaround! Suitable values might be
        # +128 for Ceta and +8 for Falcon III (https://doi.org/10.1101/615484)
        self.pedestal = float(os.environ.get("ADD_PEDESTAL", 0))

        # Set all negative pixel values to zero (after applying the pedestal).
        # Another nasty hack, while we explore what is the best practice for
        # images that have negative-valued pixels
        self.truncate = "TRUNCATE_PIXELS" in os.environ

    @staticmethod
    def _unpack_header(header):
        hd = {}
        # What do we need from the header?
        fields = ("nx", "ny", "nz", "mx", "my", "mz")
        for key in fields:
            hd[key] = int(header[key])

        # For image stacks, NX==MX etc. should always be true. Assert this
        # to ensure we fail on an MRC file of the wrong type.
        assert hd["nx"] == hd["mx"]
        assert hd["ny"] == hd["my"]
        assert hd["nz"] == hd["mz"]

        return hd

    @staticmethod
    def _read_ext_header(fileName):
        """
        Read FEI1 format extended metadata. See FeiMrc2Img.py from
        https://github.com/fei-company/FeiImageFileIO/ courtesy of Lingbo Yu.
        """

        ext_header_def = [
            ('alphaTilt',         1, 'f'),  # 4-byte floating point   Alpha tilt, in degrees.
            ('integrationTime',       1, 'f'),  # 4-byte floating point   Exposure time in seconds.
            ('tilt_axis',      1, 'f'),  # 4-byte floating point   The orientation of the tilt axis in the image in degrees. Vertical to the top is 0 [deg], the direction of positive rotation is anti-clockwise.
            ('pixelSpacing',     1, 'f'),  # 4-byte floating point   The pixel size of the images in SI units (meters).
            ('acceleratingVoltage', 1, 'f'),  # 4-byte floating point   Value of the high tension in SI units (volts).
            ('cameraLength', 1, 'f'), #4-byte floating point The calibrated camera length
            ('camera', 16, 'c'),
            ('physicalPixel', 1, 'f' ),
            ('dim', 1, 'i'),
            ('binning', 1, 'i'),
            ('wavelength', 1, 'f'),
            ('noiseReduction',1,'?')
        ]
        _sizeof_dtypes =  {
            "i":4, "f":4, "d":8, "?":1,"s":16}
        ext_header_offset = {
            'alphaTilt':(100,'d'),
            'integrationTime': (419,'d'),
            'tilt_axis':(140, 'd'),
            'pixelSpacing':(156, 'd'),
            'acceleratingVoltage':(84, 'd'),
            'camera':(435, 's'),
            'binning':(427, 'i'),
            'noiseReduction':(467,'?')}

        def cal_wavelength(V0):
            h=6.626e-34 #Js, Planck's constant
            m=9.109e-31 #kg, electron mass
            e=1.6021766208e-19 #C, electron charge
            c=3e8 #m/s^2, speed

            return h/sqrt(2*m*e*V0*(1+e*V0/(2*m*c*c)))*1e10 #return wavelength in Angstrom

        ext_header = {}
        with open(fileName,'rb') as a:
            ext_header['dim'] = struct.unpack('i',a.read(4))[0]
            for key, offset in ext_header_offset.items():
                a.seek(1024+offset[0])
                if 's' not in offset[1] :
                    ext_header[key] = struct.unpack(offset[1],a.read(_sizeof_dtypes[offset[1]]))[0]
                else:
                    ext_header[key] = b''.join(struct.unpack(offset[1]*_sizeof_dtypes[offset[1]], a.read(_sizeof_dtypes[offset[1]]))).rstrip(b'\x00')
            if b'Ceta' in ext_header['camera']:
                ext_header['binning'] = 4096/ext_header['dim']
                ext_header['physicalPixel'] = 14e-6
            ext_header['wavelength'] = cal_wavelength(ext_header['acceleratingVoltage'])
            ext_header['cameraLength'] = (ext_header['physicalPixel']*ext_header['binning'])/(ext_header['pixelSpacing']*ext_header['wavelength']*1e-10)
            #print ext_header

        return ext_header

    def get_raw_data(self):

        # Use mrcfile to open the dataset and the image.
        # Note MRC files use z, y, x ordering
        with mrcfile.mmap(self._image_file) as mrc:
            image = flex.double(mrc.data.astype("double"))
            image += self.pedestal

            if self.truncate:
                image.set_selected((image < 0), 0)

            return image

    def _goniometer(self):
        """Return a model for a simple single-axis goniometer."""

        direction = matrix.col((0.0, -1.0, 0.0))
        rot_by_deg = self._header_dictionary.get('tilt_axis', 0.0)
        direction.rotate(axis=matrix.col((0.0, 0.0, 1.0)),
            angle=rot_by_deg, deg=True)

        return self._goniometer_factory.known_axis(direction)

    def _detector(self):
        """Dummy detector"""

        image_size = (self._header_dictionary["nx"], self._header_dictionary["ny"])

        # Get pixel size, defaulting to 14 um for the Ceta if unknown
        physical_pixel = self._header_dictionary.get('physicalPixel', 1.4e-5)
        binning = self._header_dictionary.get('binning')
        if binning is None:
            if image_size == (2048, 2048):
                binning = 2.0
            else:
                binning = 1.0
        pixel_size = physical_pixel * 1000.0 * binning

        # The best distance measure is calculated from the calibrated pixel
        # size. If this is not available then default to the nominal camera
        # length, or finally to 2.0m
        distance = self._header_dictionary.get('cameraLength', 2.0) * 1000
        try:
            calibrated_pixel_size = self._header_dictionary['pixelSpacing'] # 1/m
            wavelength = self._header_dictionary['wavelength'] * 1e-10 # m
            distance = pixel_size / (calibrated_pixel_size * wavelength) # mm
        except KeyError:
            pass

        # Get detector-specific details for TF detectors as discussed with
        # Lingbo Yu. Ceta has gain of > 26 and Ceta and Falcon III both saturate
        # at about 8000.0 for binning=1
        camera = self._header_dictionary.get('camera', '').lower()
        if b'ceta' in camera:
            gain = 26.0
            saturation = 8000 * binning**2
        elif b'falcon' in camera:
            gain = 1.0
            saturation = 8000 * binning**2
        else:
            gain = 1.0
            saturation = 1e6
        saturation += self.pedestal
        trusted_range = (-1000, saturation)

        # Beam centre not in the header - set to the image centre
        beam_centre = [(pixel_size * i) / 2 for i in image_size]
        detector = self._detector_factory.simple(
            "PAD",
            distance,
            beam_centre,
            "+x",
            "-y",
            (pixel_size, pixel_size),
            image_size,
            trusted_range,
        )

        for panel in detector: panel.set_gain(gain)
        return detector

    def _beam(self):
        """Unpolarized beam model"""

        # Default to 200 keV
        wavelength = self._header_dictionary.get('wavelength', 0.02508)
        return self._beam_factory.make_polarized_beam(
            sample_to_source=(0.0, 0.0, 1.0),
            wavelength=wavelength,
            polarization=(0, 1, 0),
            polarization_fraction=0.5,
        )

class FormatMRCimages(FormatMRC):

    @staticmethod
    def understand(image_file):
        mrc = mrcfile.mmap(image_file)
        return not mrc.is_image_stack()

    def __init__(self, image_file, **kwargs):
        from dxtbx import IncorrectFormatError

        if not self.understand(image_file):
            raise IncorrectFormatError(self, image_file)
        Format.__init__(self, image_file, **kwargs)

    def _scan(self):
        """Dummy scan for this image"""

        alpha = self._header_dictionary.get('alphaTilt', 0.0)
        dalpha = 1.0
        exposure = self._header_dictionary.get('integrationTime', 0.0)
        fname = os.path.split(self._image_file)[-1]
        index = int(fname.split("_")[-1].split(".")[0])
        return ScanFactory.make_scan((index, index), exposure, (alpha, dalpha), {index: 0})

class FormatMRCstack(FormatMultiImage, FormatMRC):

    @staticmethod
    def understand(image_file):
        mrc = mrcfile.mmap(image_file)
        return mrc.is_image_stack()

    def __init__(self, image_file, **kwargs):
        from dxtbx import IncorrectFormatError

        if not self.understand(image_file):
            raise IncorrectFormatError(self, image_file)
        FormatMultiImage.__init__(self, **kwargs)
        Format.__init__(self, image_file, **kwargs)

    def get_num_images(self):
        return self._header_dictionary["nz"]

    def get_goniometer(self, index=None):
        return Format.get_goniometer(self)

    def get_detector(self, index=None):
        return Format.get_detector(self)

    def get_beam(self, index=None):
        return Format.get_beam(self)

    def get_scan(self, index=None):
        if index == None:
            return Format.get_scan(self)
        else:
            scan = Format.get_scan(self)
            return scan[index]

    def get_image_file(self, index=None):
        return Format.get_image_file(self)

    def get_raw_data(self, index):
        # Use mrcfile to open the dataset and extract slice index from the stack.
        # Note MRC files use z, y, x ordering
        with mrcfile.mmap(self._image_file) as mrc:
            raw_data = mrc.data[index, ...]

        return flex.double(raw_data.astype("double")) + self.pedestal

    def _scan(self):
        """Dummy scan for this stack"""

        nframes = self.get_num_images()
        image_range = (1, nframes)

        # Dummy values, not known from the header
        exposure_times = 0.0
        oscillation = (0, 1.0)
        epochs = [0] * nframes

        return self._scan_factory.make_scan(
            image_range, exposure_times, oscillation, epochs, deg=True
        )
