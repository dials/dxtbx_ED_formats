#!/usr/bin/env python
# FormatSMV_TVIPS.py
#  Copyright (C) (2017) STFC Rutherford Appleton Laboratory, UK.
#
#  Author: David Waterman.
#
#  This code is distributed under the BSD license, a copy of which is
#  included in the root directory of this package.

"""Implementation of a format class for images recorded on a TVIPS TemCam-F416
and converted to SMV by img2px, part of the tvips-tools-jiffies package.
This is intended specifically for processing of the Trypsin datasets available
at https://data.sbgrid.org/dataset/288/, but could be adapted for other
datasets produced in a similar manner."""

from __future__ import absolute_import, division, print_function

import os
import time

from dxtbx.format.FormatSMVADSC import FormatSMVADSC
from dxtbx.model.detector import Detector


class FormatSMV_TVIPS(FormatSMVADSC):
    """TVIPS TemCam-F416 images converted to SMV externally. We have to inherit
    from FormatSMVADSC rather than FormatSMV, as FormatSMVADSC is promiscuous and
    recognises most SMV files"""

    @staticmethod
    def understand(image_file):

        size, header = FormatSMVADSC.get_smv_header(image_file)

        # Not much to go on with these headers. Check we have exactly the keys
        # expected and no others
        if sorted(header.keys()) != [
            "BEAM_CENTER_X",
            "BEAM_CENTER_Y",
            "BIN",
            "BYTE_ORDER",
            "DATE",
            "DETECTOR_SN",
            "DIM",
            "DISTANCE",
            "HEADER_BYTES",
            "OSC_RANGE",
            "OSC_START",
            "PHI",
            "PIXEL_SIZE",
            "SIZE1",
            "SIZE2",
            "TIME",
            "TWOTHETA",
            "TYPE",
            "WAVELENGTH",
        ]:
            return False

        # Check a few values are as expected
        if header.get("DETECTOR_SN") != "unknown":
            return False
        if header.get("BIN") != "2x2":
            return False
        if header.get("BYTE_ORDER") != "little_endian":
            return False
        if header.get("HEADER_BYTES") != "512":
            return False
        if header.get("PIXEL_SIZE") != "0.0311999992":
            return False
        if header.get("SIZE1") != "2048":
            return False
        if header.get("SIZE2") != "2048":
            return False
        if header.get("TYPE") != "unsigned_short":
            return False

        return True

    def _detector(self):
        pixel_size = 0.0311999992, 0.0311999992
        image_size = 2048, 2048
        trusted_range = (-2, 65535)
        gain = 5  # As suggested for processing with MOSFLM - unsure how right this is
        distance = float(self._header_dictionary["DISTANCE"])
        beam_x = float(self._header_dictionary["BEAM_CENTER_X"])
        beam_y = float(self._header_dictionary["BEAM_CENTER_Y"])
        d = self._detector_factory.simple(
            "PAD",
            distance,
            (beam_y, beam_x),
            "+x",
            "-y",
            pixel_size,
            image_size,
            trusted_range,
            gain=gain,
        )

        return d

    def _beam(self):
        """Return an unpolarized beam model."""

        wavelength = float(self._header_dictionary["WAVELENGTH"])

        return self._beam_factory.make_polarized_beam(
            sample_to_source=(0.0, 0.0, 1.0),
            wavelength=wavelength,
            polarization=(0, 1, 0),
            polarization_fraction=0.5,
        )

    def get_raw_data(self):

        # hack IMAGE_PEDESTAL into the header dictionary, then we can use the method
        # from the base class. Set this to values for specific datasets based on
        # the processing instructions at https://data.sbgrid.org/dataset/288/.
        try:
            image_pedestal = os.environ["IMAGE_PEDESTAL"]
        except KeyError:
            from libtbx.utils import Sorry

            raise Sorry(
                "This format expects an environment variable IMAGE_PEDESTAL "
                "to be set explicitly (which could be IMAGE_PEDESTAL=0) as the "
                "required information is not in the image headers "
            )
        self._header_dictionary["IMAGE_PEDESTAL"] = image_pedestal
        return super(FormatSMV_TVIPS, self).get_raw_data()


if __name__ == "__main__":

    import sys

    for arg in sys.argv[1:]:
        print(FormatSMV_TVIPS.understand(arg))
