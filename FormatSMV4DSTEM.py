"""Format class to recognise images from a camera used for 4D-STEM measurements
(https://doi.org/10.1017/S1431927620019753), which have been converted to SMV"""

from __future__ import absolute_import, division, print_function
import os

from dxtbx.format.FormatSMVADSC import FormatSMVADSC

class FormatSMV4DSTEM(FormatSMVADSC):
    @staticmethod
    def understand(image_file):

        # Allow this class to override FormatSMVADSC with an environment variable
        if "FORCE_SMV_AS_4DSTEM" in os.environ:
            return True

        return False

    def _detector(self):

        distance = float(self._header_dictionary["DISTANCE"])
        beam_x = float(self._header_dictionary["BEAM_CENTER_X"])
        beam_y = float(self._header_dictionary["BEAM_CENTER_Y"])
        pixel_size = float(self._header_dictionary["PIXEL_SIZE"])
        image_size = (
            float(self._header_dictionary["SIZE1"]),
            float(self._header_dictionary["SIZE2"]),
        )
        # Assume GAIN=1 as counting
        binning = {"1x1": 1, "2x2": 2}.get(self._header_dictionary.get("BIN"), 1)
        gain = 1.0
        saturation = 65535 #?
        trusted_range = (-1, saturation)
        pedestal = float(self._header_dictionary.get("IMAGE_PEDESTAL", 0))

        return self._detector_factory.simple(
            "PAD",
            distance,
            (beam_y, beam_x),
            "+x",
            "-y",
            (pixel_size, pixel_size),
            image_size,
            trusted_range,
            [],
            gain=gain,
            pedestal=pedestal,
        )

    def _beam(self):
        """Return a simple model for an unpolarised beam."""

        wavelength = float(self._header_dictionary["WAVELENGTH"])
        return self._beam_factory.make_polarized_beam(
            sample_to_source=(0.0, 0.0, 1.0),
            wavelength=wavelength,
            polarization=(0, 1, 0),
            polarization_fraction=0.5,
        )
