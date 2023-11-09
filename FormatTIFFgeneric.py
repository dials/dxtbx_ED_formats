"""A format class for generic TIFF images plus implementations for specific
detectors producing electron diffraction data"""

import os
import io
from dxtbx.format.Format import Format
from dxtbx.format.FormatStill import FormatStill
from scitbx.array_family import flex
import re
from dxtbx import flumpy
from dxtbx.model.beam import Probe
from dxtbx.model.detector import Detector

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

    WARNING: this format is not very specific so an environment variable,
    QD_MERLIN_TIFF, must be set, otherwise this will pick up *any* TIFF file
    containing a single 512x512 pixel image.
    """

    @staticmethod
    def understand(image_file):
        """Check to see if this looks like a TIFF format image with a single page"""

        if os.getenv("QD_MERLIN_TIFF") is None:
            return False

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
        return self._beam_factory.make_polarized_beam(
            sample_to_source=(0.0, 0.0, 1.0),
            wavelength=wavelength,
            polarization=(0, 1, 0),
            polarization_fraction=0.5,
            probe=Probe.electron,
        )

    def _detector(self):
        """Dummy detector"""

        pixel_size = 0.055, 0.055
        image_size = (512, 512)
        dyn_range = 12
        trusted_range = (0, 2 ** dyn_range - 1)
        beam_centre = [(p * i) / 2 for p, i in zip(pixel_size, image_size)]
        d = self._detector_factory.simple(
            "PAD", 2440, beam_centre, "+x", "-y", pixel_size, image_size, trusted_range
        )
        return d


class FormatTIFFgeneric_Timepix512(FormatTIFFgeneric):
    """An experimental image reading class for TIFF images from a Timepix
    detectors with 512x512 pixels where the central cross is excluded and the
    image is separated into 4 panels.

    TIMEPIX512
    WARNING: this format is not very specific so an environment variable,
    QD_MERLIN_TIFF, must be set, otherwise this will pick up *any* TIFF file
    containing a single 512x512 pixel image.
    """

    @staticmethod
    def understand(image_file):
        """Check to see if this looks like a TIFF format image with a single page"""

        if os.getenv("TIMEPIX512_TIFF") is None:
            return False

        with tifffile.TiffFile(image_file) as tif:
            page = tif.pages[0]
            if page.shape != (512, 512):
                return False

        return True

    def _detector(self):
        """Dummy detector"""
        from scitbx import matrix

        # 55 mu pixels
        pixel_size = 0.055, 0.055
        trusted_range = (-1, 65535)
        material = "Si"
        thickness = 0.3  # assume 300 mu thick. This is actually in the header too
        # so could take it from there

        # Initialise detector frame - dummy origin to place detector at the header
        # distance along the canonical beam direction. Dummy distance
        distance = 1000
        fast = matrix.col((1.0, 0.0, 0.0))
        slow = matrix.col((0.0, -1.0, 0.0))
        cntr = matrix.col((0.0, 0.0, -100.0))

        # shifts to go from the centre to the origin - outer pixels are 0.165 mm
        self._array_size = (512,512)
        off_x = (self._array_size[0] / 2 - 2) * pixel_size[0]
        off_x += 2 * 0.165
        shift_x = -1.0 * fast * off_x
        off_y = (self._array_size[1] / 2 - 2) * pixel_size[1]
        off_y += 2 * 0.165
        shift_y = -1.0 * slow * off_y
        orig = cntr + shift_x + shift_y

        d = Detector()

        root = d.hierarchy()
        root.set_local_frame(fast.elems, slow.elems, orig.elems)

        self.coords = {}
        panel_idx = 0

        # set panel extent in pixel numbers and x, y mm shifts. Note that the
        # outer pixels are 0.165 mm in size. These are excluded from the panel
        # extents.
        pnl_data = []
        pnl_data.append(
            {
                "xmin": 1,
                "ymin": 1,
                "xmax": 255,
                "ymax": 255,
                "xmin_mm": 1 * 0.165,
                "ymin_mm": 1 * 0.165,
            }
        )
        pnl_data.append(
            {
                "xmin": 257,
                "ymin": 1,
                "xmax": 511,
                "ymax": 255,
                "xmin_mm": 3 * 0.165 + (511 - 257) * pixel_size[0],
                "ymin_mm": 1 * 0.165,
            }
        )
        pnl_data.append(
            {
                "xmin": 1,
                "ymin": 257,
                "xmax": 255,
                "ymax": 511,
                "xmin_mm": 1 * 0.165,
                "ymin_mm": 3 * 0.165 + (511 - 257) * pixel_size[1],
            }
        )
        pnl_data.append(
            {
                "xmin": 257,
                "ymin": 257,
                "xmax": 511,
                "ymax": 511,
                "xmin_mm": 3 * 0.165 + (511 - 257) * pixel_size[0],
                "ymin_mm": 3 * 0.165 + (511 - 257) * pixel_size[1],
            }
        )

        # redefine fast, slow for the local frame
        fast = matrix.col((1.0, 0.0, 0.0))
        slow = matrix.col((0.0, 1.0, 0.0))

        for ipanel, pd in enumerate(pnl_data):
            xmin = pd["xmin"]
            xmax = pd["xmax"]
            ymin = pd["ymin"]
            ymax = pd["ymax"]
            xmin_mm = pd["xmin_mm"]
            ymin_mm = pd["ymin_mm"]

            origin_panel = fast * xmin_mm + slow * ymin_mm

            panel_name = "Panel%d" % panel_idx
            panel_idx += 1

            p = d.add_panel()
            p.set_type("SENSOR_PAD")
            p.set_name(panel_name)
            p.set_raw_image_offset((xmin, ymin))
            p.set_image_size((xmax - xmin, ymax - ymin))
            p.set_trusted_range(trusted_range)
            p.set_pixel_size((pixel_size[0], pixel_size[1]))
            p.set_thickness(thickness)
            p.set_material("Si")
            # p.set_mu(mu)
            # p.set_px_mm_strategy(ParallaxCorrectedPxMmStrategy(mu, t0))
            p.set_local_frame(fast.elems, slow.elems, origin_panel.elems)
            p.set_raw_image_offset((xmin, ymin))
            self.coords[panel_name] = (xmin, ymin, xmax, ymax)

        return d

    def _goniometer(self):
        """Dummy goniometer, 'vertical' as the images are viewed. Not completely
        sure about the handedness yet"""

        return self._goniometer_factory.known_axis((0, 1, 0))

    def _beam(self):
        """Dummy beam, energy 200 keV"""

        wavelength = 0.02508
        return self._beam_factory.make_polarized_beam(
            sample_to_source=(0.0, 0.0, 1.0),
            wavelength=wavelength,
            polarization=(0, 1, 0),
            polarization_fraction=0.5,
            probe=Probe.electron,
        )

    def get_raw_data(self):

        raw_data = tifffile.imread(self._image_file)
        raw_data = flumpy.from_numpy(raw_data.astype(float))
        raw_data.reshape(flex.grid(self._array_size[1], self._array_size[0]))

        self._raw_data = []

        d = self.get_detector()
        for panel in d:
            xmin, ymin, xmax, ymax = self.coords[panel.get_name()]
            self._raw_data.append(raw_data[ymin:ymax, xmin:xmax])

        return tuple(self._raw_data)

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
        return self._beam_factory.make_polarized_beam(
            sample_to_source=(0.0, 0.0, 1.0),
            wavelength=wavelength,
            polarization=(0, 1, 0),
            polarization_fraction=0.5,
        )

    def _detector(self):
        """Dummy detector"""

        pixel_size = 0.055, 0.055
        image_size = (516, 516)
        dyn_range = 20  # XXX ?
        trusted_range = (-1, 2 ** dyn_range - 1)
        beam_centre = [(p * i) / 2 for p, i in zip(pixel_size, image_size)]
        d = self._detector_factory.simple(
            "PAD", 2440, beam_centre, "+x", "-y", pixel_size, image_size, trusted_range
        )
        return d


class FormatTIFFgeneric_FEI_Tecnai_G2(FormatTIFFgeneric):
    """Format reader for the PETS2 Quartz SiO2 example, which was recorded on
    an FEI Tecnai G2 microscope with a CCD detector.
    """

    @staticmethod
    def understand(image_file):
        """Check to see if this looks like a TIFF format 516*516 image with
        an expected string in the ImageDescription tag"""

        with tifffile.TiffFile(image_file) as tif:

            page = tif.pages[0]
            if page.shape != (1024, 1024):
                return False
            OlympusSIS = page.tags[33560]
            if not "Veleta" in OlympusSIS.value["cameraname"]:
                return False

        return True

    def _goniometer(self):
        """Dummy goniometer, 'vertical' as the images are viewed. Not completely
        sure about the handedness yet"""

        return self._goniometer_factory.known_axis((0, 1, 0))

    def _beam(self):
        """Dummy beam, energy 200 keV"""

        wavelength = 0.02508
        return self._beam_factory.make_polarized_beam(
            sample_to_source=(0.0, 0.0, 1.0),
            wavelength=wavelength,
            polarization=(0, 1, 0),
            polarization_fraction=0.5,
        )

    def _detector(self):
        """Dummy detector"""

        # 2x2 binning https://cfim.ku.dk/equipment/electron_microscopy/cm100/Veleta.pdf
        pixel_size = 0.026, 0.026
        image_size = (1024, 1024)
        dyn_range = 14  # XXX ?
        trusted_range = (-1, 2 ** dyn_range - 1)
        beam_centre = [(p * i) / 2 for p, i in zip(pixel_size, image_size)]
        d = self._detector_factory.simple(
            "PAD", 2440, beam_centre, "+x", "-y", pixel_size, image_size, trusted_range
        )
        return d


class FormatTIFFgeneric_Medipix(FormatTIFFgeneric):
    """An experimental image reading class for TIFF images from a Medipix
    detector which have been converted to 16 bits, have 514*514 pixels and
    have geometry and flat field corrections applied.

    The header does not contain useful information about the geometry, therefore
    we will construct dummy objects and expect to override on import using
    site.phil.

    WARNING: this format is not very specific and will pick up *any* TIFF file
    containing a single 514x514 pixel image.
    """

    @staticmethod
    def understand(image_file):
        """Check to see if this looks like a TIFF format image with a single page"""

        with tifffile.TiffFile(image_file) as tif:
            page = tif.pages[0]
            if page.shape != (514, 514):
                return False

        return True

    def _goniometer(self):
        """Dummy goniometer, 'vertical' as the images are viewed. Not completely
        sure about the handedness yet"""

        return self._goniometer_factory.known_axis((0, 1, 0))

    def _beam(self):
        """Dummy beam, energy 200 keV"""

        wavelength = 0.02508
        return self._beam_factory.make_polarized_beam(
            sample_to_source=(0.0, 0.0, 1.0),
            wavelength=wavelength,
            polarization=(0, 1, 0),
            polarization_fraction=0.5,
        )

    def _detector(self):
        """Dummy detector"""

        pixel_size = 0.055, 0.055
        image_size = (514, 514)
        dyn_range = 16
        trusted_range = (-1, 2 ** dyn_range - 1)
        beam_centre = [(p * i) / 2 for p, i in zip(pixel_size, image_size)]
        d = self._detector_factory.simple(
            "PAD", 2440, beam_centre, "+x", "-y", pixel_size, image_size, trusted_range
        )
        return d

class FormatTIFFgeneric_BlochwaveSim(FormatTIFFgeneric):
    """Format class to process headerless TIFF images produced by Tarik Drevon's
    Bloch wave simulation. Use environment variable BLOCHWAVE_TIFF to activate,
    and assumes 2048^2 pixels.
    """

    @staticmethod
    def understand(image_file):
        if os.getenv("BLOCHWAVE_TIFF") is None:
            return False

        with tifffile.TiffFile(image_file) as tif:
            page = tif.pages[0]
            if page.shape != (2048, 2048):
                return False

        return True

    def _goniometer(self):
        return self._goniometer_factory.known_axis((1, 0, 0))

    def _beam(self):
        wavelength = 0.02508
        return self._beam_factory.make_polarized_beam(
            sample_to_source=(0.0, 0.0, 1.0),
            wavelength=wavelength,
            polarization=(0, 1, 0),
            polarization_fraction=0.5,
        )

    def _detector(self):
        pixel_size = 0.028, 0.028
        image_size = (2048, 2048)
        dyn_range = 16
        trusted_range = (-1, 2 ** dyn_range - 1)
        beam_centre = [(p * i) / 2 for p, i in zip(pixel_size, image_size)]
        d = self._detector_factory.simple(
            "PAD", 834, beam_centre, "+x", "-y", pixel_size, image_size, trusted_range
        )
        return d

class FormatTIFF_UED(FormatTIFFgeneric, FormatStill):
    """An experimental image reading class for TIFF images from a UED
    instrument. Most of this is probably incorrect.
    """

    def __init__(self, image_file, **kwargs):

        FormatTIFFgeneric.__init__(self, image_file, **kwargs)
        FormatStill.__init__(self, image_file, **kwargs)

        return

    @staticmethod
    def understand(image_file):
        """Check to see if this looks like a TIFF format image with a single page"""

        with tifffile.TiffFile(image_file) as tif:
            page = tif.pages[0]
            if page.shape != (1300, 1340):
                return False

        return True

    def _beam(self):
        """Dummy beam, energy 200 keV"""

        wavelength = 0.02508
        return self._beam_factory.make_polarized_beam(
            sample_to_source=(0.0, 0.0, 1.0),
            wavelength=wavelength,
            polarization=(0, 1, 0),
            polarization_fraction=0.5,
        )

    def _detector(self):
        """Dummy detector"""

        pixel_size = 0.060, 0.060
        image_size = (1300, 1340)
        dyn_range = 20  # No idea what is correct
        trusted_range = (-1, 2 ** dyn_range - 1)
        beam_centre = [(p * i) / 2 for p, i in zip(pixel_size, image_size)]
        d = self._detector_factory.simple(
            "PAD", 2440, beam_centre, "+x", "-y", pixel_size, image_size, trusted_range
        )
        return d


class FormatTIFF_UED_BNL(FormatTIFFgeneric, FormatStill):
    """An experimental image reading class for TIFF images from a UED
    instrument at BNL: https://www.bnl.gov/atf/capabilities/ued.php.

    Set environment variable UED_BNL_TIFF to use.
    """

    def __init__(self, image_file, **kwargs):

        FormatTIFFgeneric.__init__(self, image_file, **kwargs)
        FormatStill.__init__(self, image_file, **kwargs)

        return

    @staticmethod
    def understand(image_file):
        """Check to see if this looks like a TIFF format image with a single page"""

        if os.getenv("UED_BNL_TIFF") is None:
            return False

        with tifffile.TiffFile(image_file) as tif:
            page = tif.pages[0]
            if page.shape != (512, 512):
                return False

        return True

    def _beam(self):
        """Dummy beam, energy 200 keV"""

        wavelength = 0.03569
        return self._beam_factory.make_polarized_beam(
            sample_to_source=(0.0, 0.0, 1.0),
            wavelength=wavelength,
            polarization=(0, 1, 0),
            polarization_fraction=0.5,
        )

    def _detector(self):
        """Dummy detector"""

        pixel_size = 0.016, 0.016
        image_size = (512, 512)
        dyn_range = 20  # No idea what is correct
        trusted_range = (-1, 2 ** dyn_range - 1)
        beam_centre = [(p * i) / 2 for p, i in zip(pixel_size, image_size)]
        d = self._detector_factory.simple(
            "CCD", 3480, beam_centre, "+x", "-y", pixel_size, image_size, trusted_range
        )
        return d
