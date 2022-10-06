"""Format class to read SMV images where the wavelength has been wrongly
set to zero. This will instead be taken from an environment variable"""

from __future__ import absolute_import, division, print_function
import os

from dxtbx.format.FormatSMVADSC import FormatSMVADSC


class FormatSMVwavelengthZero(FormatSMVADSC):
    @staticmethod
    def understand(image_file):

        # Allow this class to override FormatSMVADSC with an environment variable
        if "FORCE_SMV_WAVELENGTH" in os.environ:
            return True

        return False


    def _beam(self):
        """Return a simple model for an unpolarised beam."""

        wavelength = float(os.environ["FORCE_SMV_WAVELENGTH"])
        return self._beam_factory.make_polarized_beam(
            sample_to_source=(0.0, 0.0, 1.0),
            wavelength=wavelength,
            polarization=(0, 1, 0),
            polarization_fraction=0.5,
        )
