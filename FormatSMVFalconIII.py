"""Experimental implementation of a format class to recognise images from a
ThermoFisher Falcon III detector that have been converted to SMV with useful
metadata. We want to override the beam model to produce an unpolarised beam
and to set the detector gain to something sensible"""

from __future__ import absolute_import, division, print_function

from dxtbx.format.FormatSMVADSC import FormatSMVADSC
from dxtbx.model.beam import Probe


class FormatSMVFalconIII(FormatSMVADSC):
    @staticmethod
    def understand(image_file):

        size, header = FormatSMVADSC.get_smv_header(image_file)
        if header.get("DETECTOR_SN") == "falcon3":
            return True

        return False

    def _detector(self):
        d = super(FormatSMVFalconIII, self)._detector()

        # Set gain. Detector is used in integrating mode. In a single frame the
        # gain is 390 counts/pe (pe = primary electron of 300 keV). But each image is
        # a summation of 40 frames, and the total counts are averaged. So the gain is
        # then 390/40 ~10.

        # But spotfinding results are better with gain set to 1.0!
        # for p in d:
        #    p.set_gain(10)

        return d

    def _beam(self):
        """Return a simple model for an unpolarised beam."""

        wavelength = float(self._header_dictionary["WAVELENGTH"])
        return self._beam_factory.make_polarized_beam(
            sample_to_source=(0.0, 0.0, 1.0),
            wavelength=wavelength,
            polarization=(0, 1, 0),
            polarization_fraction=0.5,
            probe=Probe.electron,
        )


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:]:
        print(FormatSMVFalconIII.understand(arg))
