"""Experimental format class for miniCBF files from an ELDICO diffractometer,
which uses a DECTRIS QUADRO detector (EIGER technology)"""

from __future__ import absolute_import, division, print_function
import os

from dxtbx.format.FormatCBFMiniEiger import FormatCBFMiniEiger
from dxtbx.model import SimplePxMmStrategy


class FormatCBFMiniEigerQuadro(FormatCBFMiniEiger):
    """We have to inherit from FormatCBFMiniEiger, as this class understands
    these files. This is not ideal, as that class is geared towards EIGER
    X-ray detectors"""

    @staticmethod
    def understand(image_file):

        header = FormatCBFMiniEiger.get_cbf_header(image_file)

        # This is not very specific for the QUADRO and would probably also match
        # CBFs from e.g. the SINGLA.
        for record in header.split("\n"):

            if "# wavelength" in record.lower():
                try:
                    wl = float(record.split()[-2])
                except ValueError:
                    return False
                if wl > 0.05:
                    return False

        return True

    def _goniometer(self):
        """Axis as determined by a single run of dials.find_rotation_axis, so
        probably not exact"""

        return self._goniometer_factory.known_axis((-0.0540788,0.998537,0))

    def _beam(self):
        """Ensure an unpolarised beam"""

        wavelength = float(self._cif_header_dictionary["Wavelength"].split()[0])
        return self._beam_factory.make_polarized_beam(
            sample_to_source=(0.0, 0.0, 1.0),
            wavelength=wavelength,
            polarization=(0, 1, 0),
            polarization_fraction=0.5,
        )

    def _detector(self):
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

        if "Silicon" in self._cif_header_dictionary:
            thickness = (
                float(self._cif_header_dictionary["Silicon"].split()[2]) * 1000.0
            )
            material = "Si"
        elif "CdTe" in self._cif_header_dictionary:
            thickness = float(self._cif_header_dictionary["CdTe"].split()[2]) * 1000.0
            material = "CdTe"
        else:
            thickness = 0.450
            material = "Si"

        nx = int(self._cif_header_dictionary["X-Binary-Size-Fastest-Dimension"])
        ny = int(self._cif_header_dictionary["X-Binary-Size-Second-Dimension"])

        if "Count_cutoff" in self._cif_header_dictionary:
            overload = int(self._cif_header_dictionary["Count_cutoff"].split()[0])
        else:
            # missing from data transformed with GPhL converter - dials#376
            overload = 1048576
        if overload == 0:
            overload = 1048576

        minimum_trusted_value = 0

        try:
            identifier = self._cif_header_dictionary["Detector"].encode()
        except KeyError:
            identifier = "Unknown Eiger"

        detector = self._detector_factory.simple(
            sensor="PAD",
            distance=distance * 1000.0,
            beam_centre=(beam_x * pixel_x * 1000.0, beam_y * pixel_y * 1000.0),
            fast_direction="+x",
            slow_direction="-y",
            pixel_size=(1000 * pixel_x, 1000 * pixel_y),
            image_size=(nx, ny),
            trusted_range=(minimum_trusted_value, overload),
            mask=[],
        )

        # Here we set specifics, notably gain=3 and parallax correction and
        # QE correction are effectively disabled by setting the simple
        # pixel-to-millimetre strategy and a very high mu value.
        for panel in detector:
            panel.set_gain(3)
            panel.set_thickness(thickness)
            panel.set_material(material)
            panel.set_identifier(identifier)
            panel.set_px_mm_strategy(SimplePxMmStrategy())
            panel.set_mu(1e10)

        return detector



