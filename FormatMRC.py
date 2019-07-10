"""Classes for Formats that recognise data stored in the MRC format, an open
standard used in electron microscopy
(http://www.ccpem.ac.uk/mrc_format/mrc2014.php)"""

from __future__ import absolute_import, division, print_function

import logging
import os
import struct
from math import sqrt
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

        if h['exttyp'].tostring() == 'FEI1':
            xh = self._read_ext_header(self._image_file)
            self._header_dictionary.update(xh)

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
            for key, offset in ext_header_offset.iteritems():
                a.seek(1024+offset[0])
                if 's' not in offset[1] :
                    ext_header[key] = struct.unpack(offset[1],a.read(_sizeof_dtypes[offset[1]]))[0]
                else:
                    ext_header[key] = ''.join(struct.unpack(offset[1]*_sizeof_dtypes[offset[1]], a.read(_sizeof_dtypes[offset[1]]))).rstrip('\x00')
            if 'Ceta' in ext_header['camera']:
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
            return flex.double(mrc.data.astype("double"))

    def _goniometer(self):
        """Return a model for a simple single-axis goniometer."""

        return self._goniometer_factory.known_axis((1.0, 0.0, 0.0))

    def _detector(self):
        """Dummy detector"""

        image_size = (self._header_dictionary["nx"], self._header_dictionary["ny"])

        # Pixel sizes for the CETA camera
        if image_size == (4096, 4096):
            pixel_size = 0.014, 0.014
        if image_size == (2048, 2048):
            pixel_size = 0.028, 0.028

        # Dummy values, not stored in the header
        distance = 2000
        trusted_range = (-4, 65535)  # Unsure what is appropriate here
        beam_centre = [(p * i) / 2 for p, i in zip(pixel_size, image_size)]
        d = self._detector_factory.simple(
            "PAD",
            distance,
            beam_centre,
            "+x",
            "-y",
            pixel_size,
            image_size,
            trusted_range,
        )
        # Default to gain = 1
        # for p in d: p.set_gain(8)
        return d

    def _beam(self):
        """Dummy unpolarized beam, energy 200 keV"""

        wavelength = 0.02508
        return self._beam_factory.make_polarized_beam(
            sample_to_source=(0.0, 0.0, 1.0),
            wavelength=wavelength,
            polarization=(0, 1, 0),
            polarization_fraction=0.5,
        )

    def _scan(self):
        """Dummy scan for this image"""

        fname = os.path.split(self._image_file)[-1]
        index = int(fname.split("_")[-1].split(".")[0])
        return ScanFactory.make_scan((index, index), 0.0, (0, 0.5), {index: 0})

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

    def get_raw_data(self):

        # Use mrcfile to open the dataset and the image.
        # Note MRC files use z, y, x ordering
        with mrcfile.mmap(self._image_file) as mrc:
            return flex.double(mrc.data.astype("double"))

    def _scan(self):
        """Dummy scan for this image"""

        fname = os.path.split(self._image_file)[-1]
        index = int(fname.split("_")[-1].split(".")[0])
        return ScanFactory.make_scan((index, index), 0.0, (0, 0.5), {index: 0})

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

        return flex.double(raw_data.astype("double"))

    def _scan(self):
        """Dummy scan for this stack"""

        nframes = self.get_num_images()
        image_range = (1, nframes)

        # Dummy values, not known from the header
        exposure_times = 0.0
        oscillation = (0, 0.5)
        epochs = [0] * nframes

        return self._scan_factory.make_scan(
            image_range, exposure_times, oscillation, epochs, deg=True
        )
