#!/usr/bin/env python
# FormatFalconIIRaw.py
#  Copyright (C) (2016) STFC Rutherford Appleton Laboratory, UK.
#
#  Author: David Waterman.
#
#  This code is distributed under the BSD license. For dxtbx licensing see
#  https://github.com/cctbx/cctbx_project/blob/master/dxtbx/license.txt
#
"""Experimental implementation of a format class to recognise raw images
from an FEI Falcon II detector"""

from __future__ import absolute_import, division, print_function

import os

from dxtbx.format.Format import Format
from dxtbx.model import ScanFactory
from dxtbx.model.detector import Detector


class FormatFalconIIRaw(Format):
    @staticmethod
    def understand(image_file):
        """Check to see if this looks like a Falcon II raw format image. Not much
        to go on here. We use the file size and common bytes from the header that
        appear to be the same for all the images. Note, this appears to be for
        2x2 binning, as there are really 4096^2 pixels, but these images have
        2048^2"""

        with open(image_file, "rb") as f:
            if os.fstat(f.fileno()).st_size != 8388785:
                return False
            header = f.read(177)
            data = f.read()

        if header != (
            "FEI RawImage\x00\x01\x00\x00\x00\x00\x08\x00\x00\x00\x08"
            "\x00\x00\x01\x00\x00\x00\x10\x00\x00\x00\x01\x00\x00\x00\x80\x00"
            "\x00\x00\x02\x00\x00\x00\x00\x10" + 130 * "\x00"
        ):
            return False

        return True

    def detectorbase_start(self):
        pass

    def _start(self):
        """Open the image file and read the image header"""

        self._header_size = 177
        self._header_bytes = FormatFalconIIRaw.open_file(self._image_file, "rb").read(
            self._header_size
        )

    def get_raw_data(self):
        """Get the pixel intensities"""

        from boost.python import streambuf
        try:
            from dxtbx.ext import read_int16
        except ImportError:
            from dxtbx import read_int16
        from scitbx.array_family import flex

        f = FormatFalconIIRaw.open_file(self._image_file, "rb")
        f.read(self._header_size)

        raw_data = read_int16(streambuf(f), 2048 * 2048)
        image_size = (2048, 2048)
        raw_data.reshape(flex.grid(image_size[1], image_size[0]))

        return raw_data

    def _goniometer(self):
        """Dummy goniometer, 'vertical' as the images are viewed. Not completely
        sure about the handedness yet"""

        return self._goniometer_factory.known_axis((0, -1, 0))

    def _detector(self):
        """Dummy detector"""

        pixel_size = 0.028, 0.028
        image_size = 2048, 2048
        trusted_range = (-1, 65535)
        beam_centre = [(p * i) / 2 for p, i in zip(pixel_size, image_size)]
        d = self._detector_factory.simple(
            "PAD", 2440, beam_centre, "+x", "-y", pixel_size, image_size, trusted_range
        )
        # Not sure what the gain is
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
        return ScanFactory.make_scan((index, index), 0.0, (0, 1), {index: 0})
