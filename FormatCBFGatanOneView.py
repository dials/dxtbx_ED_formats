#!/usr/bin/env python
# FormatCBFGatanOneView.py
#  Copyright (C) (2017) STFC Rutherford Appleton Laboratory, UK.
#
#  Author: David Waterman.
#
#  This code is distributed under the BSD license. For dxtbx licensing see
#  https://github.com/cctbx/cctbx_project/blob/master/dxtbx/license.txt
#

from __future__ import absolute_import, division, print_function

import os

from dxtbx.format.FormatCBF import FormatCBF
from dxtbx.model import ScanFactory
from dxtbx.model.beam import Probe

if "DXTBX_OVERLOAD_SCALE" in os.environ:
    dxtbx_overload_scale = float(os.environ["DXTBX_OVERLOAD_SCALE"])
else:
    dxtbx_overload_scale = 1


class FormatCBFGatanOneView(FormatCBF):
    """Work-in-progress image reading class for CBF format images with a
    minimal header from an electron microscope with a Gatan OneView detector
    at RIKEN SPring-8. Largely based upon FormatCBFMini.py"""

    @staticmethod
    def understand(image_file):
        """Check to see if this looks like an CBF format image, i.e. we can
        make sense of it."""

        header = FormatCBF.get_cbf_header(image_file)

        # The header is minimal, but we want to avoid 'understanding' CBFs from
        # other detectors. Use whatever we can in the header that might uniquely
        # identify the right images
        if "data_clip:_flipx" not in header:
            return False
        if "_array_data.header_convention" in header:
            return False
        if "Detector" in header:
            return False

        return True

    def __init__(self, image_file, **kwargs):
        """Initialise the image structure from the given file."""

        from dxtbx import IncorrectFormatError

        if not self.understand(image_file):
            raise IncorrectFormatError(self, image_file)

        FormatCBF.__init__(self, image_file, **kwargs)

        self._raw_data = None

    def _start(self):
        """Read the image header and copy it into a dictionary for future reference.

        In this case the header is useless, so we populate the dictionary manually
        with dummy values"""

        FormatCBF._start(self)
        self._cif_header_dictionary = {}
        self._cif_header_dictionary["Detector_distance"] = "3.523 m"
        self._cif_header_dictionary["Beam_xy"] = "(1024.0, 1024.0) pixels"
        self._cif_header_dictionary["Wavelength"] = "0.02508 A"
        self._cif_header_dictionary["Pixel_size"] = "30e-6 m x 30e-6 m"
        self._cif_header_dictionary["X-Binary-Size-Fastest-Dimension"] = "2048"
        self._cif_header_dictionary["X-Binary-Size-Second-Dimension"] = "2048"
        self._cif_header_dictionary["Count_cutoff"] = "65535 counts"
        self._cif_header_dictionary["Phi"] = "0.0000 deg."

    def _detector(self):
        """Return a dummy model for the detector"""

        distance = float(self._cif_header_dictionary["Detector_distance"].split()[0])

        beam_xy = (
            self._cif_header_dictionary["Beam_xy"]
            .replace("(", "")
            .replace(")", "")
            .replace(",", "")
            .split()[:2]
        )

        wavelength = float(self._cif_header_dictionary["Wavelength"].split()[0])

        beam_x, beam_y = map(float, beam_xy)

        pixel_xy = (
            self._cif_header_dictionary["Pixel_size"]
            .replace("m", "")
            .replace("x", "")
            .split()
        )

        pixel_x, pixel_y = map(float, pixel_xy)

        nx = int(self._cif_header_dictionary["X-Binary-Size-Fastest-Dimension"])
        ny = int(self._cif_header_dictionary["X-Binary-Size-Second-Dimension"])

        max_trusted = dxtbx_overload_scale * int(
            self._cif_header_dictionary["Count_cutoff"].split()[0]
        )
        min_trusted = 0

        detector = self._detector_factory.simple(
            "PAD",
            distance * 1000.0,
            (beam_x * pixel_x * 1000.0, beam_y * pixel_y * 1000.0),
            "+x",
            "-y",
            (1000 * pixel_x, 1000 * pixel_y),
            (nx, ny),
            (min_trusted, max_trusted),
            [],
        )

        return detector

    def _goniometer(self):
        """Return a model for a simple single-axis goniometer."""

        return self._goniometer_factory.single_axis()

    def _beam(self):
        """Dummy unpolarized beam"""

        wavelength = float(self._cif_header_dictionary["Wavelength"].split()[0])

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
        return ScanFactory.make_scan((index, index), 0.0, (0, 1), {index: 0})

    def read_cbf_image(self, cbf_image):
        from cbflib_adaptbx import uncompress
        import binascii

        start_tag = binascii.unhexlify("0c1a04d5")

        data = self.open_file(cbf_image, "rb").read()
        data_offset = data.find(start_tag) + 4
        cbf_header = data[: data_offset - 4]

        fast = 0
        slow = 0
        length = 0
        byte_offset = False
        no_compression = False

        for record in cbf_header.split("\n"):
            if "X-Binary-Size-Fastest-Dimension" in record:
                fast = int(record.split()[-1])
            elif "X-Binary-Size-Second-Dimension" in record:
                slow = int(record.split()[-1])
            elif "X-Binary-Number-of-Elements" in record:
                length = int(record.split()[-1])
            elif "X-Binary-Size:" in record:
                size = int(record.split()[-1])
            elif "conversions" in record:
                if "x-CBF_BYTE_OFFSET" in record:
                    byte_offset = True
                elif "x-CBF_NONE" in record:
                    no_compression = True

        assert length == fast * slow

        if byte_offset:
            pixel_values = uncompress(
                packed=data[data_offset : data_offset + size], fast=fast, slow=slow
            )
        elif no_compression:
            from boost.python import streambuf

            try:
                from dxtbx.ext import read_int32
            except ImportError:
                from dxtbx import read_int32
            from scitbx.array_family import flex

            assert len(self.get_detector()) == 1
            f = self.open_file(self._image_file)
            f.read(data_offset)
            pixel_values = read_int32(streambuf(f), int(slow * fast))
            pixel_values.reshape(flex.grid(slow, fast))

        else:
            raise ValueError(
                "Uncompression of type other than byte_offset or none "
                " is not supported (contact authors)"
            )

        return pixel_values

    def get_raw_data(self):
        if self._raw_data is None:
            data = self.read_cbf_image(self._image_file)
            self._raw_data = data

        return self._raw_data


if __name__ == "__main__":

    import sys

    for arg in sys.argv[1:]:
        print(FormatCBFGatanOneView.understand(arg))
