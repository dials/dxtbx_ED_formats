"""A format class for generic TIFF images plus implementations for specific
detectors producing electron diffraction data"""

import os
import io
from dxtbx.format.Format import Format
from scitbx.array_family import flex
import re
from dxtbx import flumpy

try:
    import tifffile
except ImportError:
    tifffile = None


class FormatTIFFgeneric(Format):
    """General-purpose TIFF image reader using tifffile. This will clash with
    the dxtbx FormatTIFF tree for Rigaku/Bruker TIFFs."""

    @staticmethod
    def understand(image_file):
        """Check to see if this looks like a TIFF format image with a single page"""

        if not tifffile:
            print(
                "FormatTIFFgeneric is installed but the required library tifffile is not available"
            )
            return False

        try:
            tif = tifffile.TiffFile(image_file)
        except tifffile.TiffFileError:
            return False

        try:
            assert len(tif.pages) == 1
            assert len(tif.series) == 1
            page = tif.pages[0]
        except (AssertionError, KeyError):
            return False
        finally:
            tif.close()

        return True

    def get_raw_data(self):
        """Get the pixel intensities"""

        raw_data = tifffile.imread(self._image_file)
        return flumpy.from_numpy(raw_data.astype(float))

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


class FormatTIFFgeneric_Merlin(FormatTIFFgeneric):
    """An experimental image reading class for TIFF images from a Quantum
    Detectors Merlin detector. We have limited information about the data format
    at present.

    The header does not contain useful information about the geometry, therefore
    we will construct dummy objects and expect to override on import using
    site.phil.

    WARNING: this format is not very specific and will pick up *any* TIFF file
    containing a single 512x512 pixel image.
    """

    @staticmethod
    def understand(image_file):
        """Check to see if this looks like a TIFF format image with a single page"""

        with tifffile.TiffFile(image_file) as tif:
            page = tif.pages[0]
            if page.shape != (512, 512):
                return False

        return True

    def _goniometer(self):
        """Dummy goniometer, 'vertical' as the images are viewed. Not completely
        sure about the handedness yet"""

        return self._goniometer_factory.known_axis((0, 1, 0))

    def _beam(self):
        """Dummy beam, energy 200 keV"""

        wavelength = 0.02508
        return self._beam_factory.simple(wavelength)

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


class FormatTIFFgeneric_ASI(FormatTIFFgeneric):
    """Format reader for the PETS2 Glycine example, which was recorded on an
    ASI hybrid pixel detector.

    """

    @staticmethod
    def understand(image_file):
        """Check to see if this looks like a TIFF format 516*516 image with
        an expected string in the ImageDescription tag"""

        with tifffile.TiffFile(image_file) as tif:

            page = tif.pages[0]
            if page.shape != (516, 516):
                return False
            ImageDescription = page.tags[270]
            if not "ImageCameraName: timepix" in ImageDescription.value:
                return False

        return True

    def _goniometer(self):
        """Dummy goniometer, 'vertical' as the images are viewed. Not completely
        sure about the handedness yet"""

        return self._goniometer_factory.known_axis((0, 1, 0))

    def _beam(self):
        """Dummy beam, energy 200 keV"""

        wavelength = 0.02508
        return self._beam_factory.simple(wavelength)

    def _detector(self):
        """Dummy detector"""

        pixel_size = 0.055, 0.055
        image_size = (516, 516)
        dyn_range = 20 # XXX ?
        trusted_range = (-1, 2 ** dyn_range - 1)
        beam_centre = [(p * i) / 2 for p, i in zip(pixel_size, image_size)]
        d = self._detector_factory.simple(
            "PAD", 2440, beam_centre, "+x", "-y", pixel_size, image_size, trusted_range
        )
        return d
