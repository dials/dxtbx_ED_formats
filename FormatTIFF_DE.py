"""Experimental implementation of a format class to recognise images in TIFF
format recorded on an electron microscope with a Direct Electron (DE) detector."""

from __future__ import absolute_import, division, print_function

import os
import io
from PIL import Image
from dxtbx.format.FormatTIFF import FormatTIFF
from dxtbx.format.FormatTIFFHelpers import read_basic_tiff_header
from dxtbx.format.FormatTIFFHelpers import BIG_ENDIAN
from dxtbx.model import ScanFactory
from boost.python import streambuf
from scitbx.array_family import flex
from dxtbx.ext import (
    read_uint8,
    read_uint16,
    read_uint16_bs,
)
from dxtbx.model.beam import Probe


class FormatTIFF_DE(FormatTIFF):
    """An image reading class for TIFF images from a Direct Electron detector.
    We have limited information about the data format at present.

    The header does not contain useful information about the geometry, therefore
    we will construct dummy objects and expect to override on import using
    site.phil.
    """

    @staticmethod
    def understand(image_file):
        """Check to see if this looks like a TIFF format image of the type
        expected from the DE detector"""

        btf = read_basic_tiff_header(image_file)
        if btf[3] is None:
            return True
        return False

    def __init__(self, image_file, **kwargs):
        """Initialise the image structure from the given file, including a
        proper model of the experiment."""

        assert self.understand(image_file)

        FormatTIFF.__init__(self, image_file, **kwargs)

    def detectorbase_start(self):
        pass

    def get_raw_data(self):
        """Get the pixel intensities"""

        if self._tiff_depth == 2:
            if self._tiff_byte_order == BIG_ENDIAN:
                read_pixel = read_uint16_bs
            else:
                read_pixel = read_uint16
        else:
            from dxtbx.ext import read_uint8 as read_pixel

        im = Image.open(self._image_file)
        raw_data = read_pixel(streambuf(io.BytesIO(im.tobytes())), 4096 * 4096)

        image_size = (4096, 4096)
        raw_data.reshape(flex.grid(image_size[1], image_size[0]))

        return raw_data

    def _goniometer(self):
        """Dummy goniometer, 'vertical' as the images are viewed. Not completely
        sure about the handedness yet"""

        return self._goniometer_factory.known_axis((0, 1, 0))

    def _detector(self):
        """Dummy detector"""

        pixel_size = 0.013, 0.013
        image_size = 4096, 4096
        trusted_range = (0, 65535)
        beam_centre = [(p * i) / 2 for p, i in zip(pixel_size, image_size)]
        d = self._detector_factory.simple(
            "PAD", 700, beam_centre, "+x", "-y", pixel_size, image_size, trusted_range
        )
        # Not sure what the gain should be, but spot-finding works when it is ~70
        for p in d:
            p.set_gain(70)
        return d

    def _beam(self):
        """Return an unpolarized beam model, energy 200 keV"""

        wavelength = 0.02508
        return self._beam_factory.make_polarized_beam(
            sample_to_source=(0.0, 0.0, 1.0),
            wavelength=wavelength,
            polarization=(0, 1, 0),
            polarization_fraction=0.5,
            probe=Probe.electron,
        )

    def _scan(self):
        """Dummy scan for this image"""

        fname = os.path.split(self._image_file)[-1]
        index = int(fname.split("_")[-1].split(".")[0])
        return ScanFactory.make_scan((index, index), 0.0, (0, 0.5), {index: 0})
