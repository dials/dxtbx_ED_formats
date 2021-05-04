"""Experimental implementation of a format class to recognise images in TIFF
format produced from a Quantum Detectors Merlin camera."""

import os
import io
from dxtbx.format.Format import Format
from scitbx.array_family import flex
import re

try:
    import tifffile
except ImportError:
    tifffile = None


class FormatTIFF_Merlin(Format):
    """An image reading class for TIFF images from a Quantum Detectors detector.
    We have limited information about the data format at present.

    The header does not contain useful information about the geometry, therefore
    we will construct dummy objects and expect to override on import using
    site.phil.
    """

    @staticmethod
    def understand(image_file):
        """Check to see if this looks like a TIFF format image of the type
        expected from the Merlin detector"""

        if not tifffile:
            return False

        try:
            tif = tifffile.TiffFile(image_file)
        except tifffile.TiffFileError:
            return False

        try:
            assert len(tif.pages) == 1
            assert len(tif.series) == 1
            page = tif.pages[0]
            assert page.shape == (512, 512)
            assert "SerialEM" in page.tags["ImageDescription"].value
        except (AssertionError, KeyError):
            return False
        finally:
            tif.close()

        return True

    def get_raw_data(self):
        """Get the pixel intensities"""

        raw_data = tifffile.imread(self._image_file)
        return flex.double(raw_data.astype(float))

    def _goniometer(self):
        """Dummy goniometer, 'vertical' as the images are viewed. Not completely
        sure about the handedness yet"""

        return self._goniometer_factory.known_axis((0, 1, 0))

    def _detector(self):
        """Dummy detector"""

        pixel_size = 0.055, 0.055
        image_size = (512, 512)
        dyn_range = 12
        trusted_range = (-1, 2 ** dyn_range - 1)
        beam_centre = [(p * i) / 2 for p, i in zip(pixel_size, image_size)]
        d = self._detector_factory.simple(
            "PAD", 2440, beam_centre, "+x", "-y", pixel_size, image_size, trusted_range
        )
        return d

    def _beam(self):
        """Dummy beam, energy 200 keV"""

        wavelength = 0.02508
        return self._beam_factory.simple(wavelength)

    def _scan(self):
        """Dummy scan for this image"""

        fname = os.path.split(self._image_file)[-1]
        # assume that the final number before the extension is the image number
        s = fname.split("_")[-1].split(".")[0]
        try:
            index = int(re.match(".*?([0-9]+)$", s).group(1))
        except AttributeError:
            index = 1
        exposure_times = 0.0
        frame = index - 1
        # Dummy scan with a 0.5 deg image
        oscillation = (frame * 0.5, 0.5)
        epochs = [0]
        return self._scan_factory.make_scan(
            (index, index), exposure_times, oscillation, epochs, deg=True
        )
