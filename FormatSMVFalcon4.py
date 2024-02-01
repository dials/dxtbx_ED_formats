"""Format class to recognise images from a ThermoFisher Falcon 4 detector that
have been converted to SMV with partial metadata. Actually, not much is specific
here to the Falcon 4 detector, apart from a suggested value of the gain given
by Max Clabbers. This is apparently relevant for the detector in 'electron
counting' mode. This Format class must be activated by an environment variable
to avoid clashes with other formats."""

from __future__ import annotations

import os

from dxtbx.format.FormatSMVADSC import FormatSMVADSC
from dxtbx.model.beam import Probe


class FormatSMVFalcon4(FormatSMVADSC):
    @staticmethod
    def understand(image_file):

        # Allow this class to override FormatSMVADSC with an environment variable
        if "FORCE_SMV_AS_FALCON4" in os.environ:
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
        # Ceta has gain of > 26 and saturates at about 8000.0 for binning=1
        # according to Thermo Fisher
        binning = {"1x1": 1, "2x2": 2}.get(self._header_dictionary.get("BIN"), 1)
        gain = float(self._header_dictionary.get("GAIN", 32.0))
        saturation = 8000 * binning**2  # Guesswork
        trusted_range = (0, saturation)
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
        # Some images have wavelength incorrectly set to zero. In those cases
        # default to 0.0197 (300 keV electrons)
        if wavelength == 0.0:
            wavelength = 0.0197

        return self._beam_factory.make_polarized_beam(
            sample_to_source=(0.0, 0.0, 1.0),
            wavelength=wavelength,
            polarization=(0, 1, 0),
            polarization_fraction=0.5,
            probe=Probe.electron,
        )
