"""Format class to recognise images stored in an MRC stack, an open standard
used in electron microscopy (http://www.ccpem.ac.uk/mrc_format/mrc2014.php)"""

from __future__ import absolute_import, division, print_function

import logging

from scitbx.array_family import flex
from dxtbx.format.Format import Format
from dxtbx.format.FormatMultiImage import FormatMultiImage
import mrcfile

logger = logging.getLogger("dials")


class FormatMRCstack(FormatMultiImage, Format):
    @staticmethod
    def understand(image_file):
        try:
            mrc = mrcfile.mmap(image_file)
        except ValueError:
            return False

        return mrc.is_image_stack()

    def __init__(self, image_file, **kwargs):
        from dxtbx import IncorrectFormatError

        if not self.understand(image_file):
            raise IncorrectFormatError(self, image_file)
        FormatMultiImage.__init__(self, **kwargs)
        Format.__init__(self, image_file, **kwargs)

    def _start(self):
        """Open the MRC file, read the metadata into an internal dictionary
        self._header_dictionary"""

        with mrcfile.mmap(self._image_file) as mrc:
            h = mrc.header
        self._header_dictionary = self._unpack_header(h)

    @staticmethod
    def _unpack_header(header):
        hd = {}
        # What do we need from the header?
        fields = ("nx", "ny", "nz", "mx", "my", "mz")
        for key in fields:
            hd[key] = int(header[key])

        # For image stacks, NX==MX etc. should always be true. Assert this
        # to ensure we fail on an MRC file of the wrong type.
        assert hd["nx"] == hd["mx"]
        assert hd["ny"] == hd["my"]
        assert hd["nz"] == hd["mz"]

        return hd

    def get_num_images(self):
        return self._header_dictionary["nz"]

    def get_goniometer(self, index=None):
        return Format.get_goniometer(self)

    def get_detector(self, index=None):
        return Format.get_detector(self)

    def get_beam(self, index=None):
        return Format.get_beam(self)

    def get_scan(self, index=None):
        if index == None:
            return Format.get_scan(self)
        else:
            scan = Format.get_scan(self)
            return scan[index]

    def get_image_file(self, index=None):
        return Format.get_image_file(self)

    def get_raw_data(self, index):
        # Use mrcfile to open the dataset and extract slice index from the stack.
        # Note MRC files use z, y, x ordering
        with mrcfile.mmap(self._image_file) as mrc:
            raw_data = mrc.data[index, ...]

        return flex.double(raw_data.astype("double"))

    def _goniometer(self):
        """Return a model for a simple single-axis goniometer."""

        return self._goniometer_factory.known_axis((1.0, 0.0, 0.0))

    def _detector(self):
        """Dummy detector"""

        image_size = (self._header_dictionary["nx"], self._header_dictionary["ny"])

        # Dummy values, not stored in the header
        pixel_size = 0.014, 0.014
        distance = 2000
        trusted_range = (-4, 65535)  # Unsure what is appropriate here
        beam_centre = [(p * i) / 2 for p, i in zip(pixel_size, image_size)]
        d = self._detector_factory.simple(
            "PAD",
            distance,
            beam_centre,
            "+x",
            "-y",
            pixel_size,
            image_size,
            trusted_range,
        )
        # Default to gain = 1
        # for p in d: p.set_gain(8)
        return d

    def _beam(self):
        """Dummy unpolarized beam, energy 200 keV"""

        wavelength = 0.02508
        return self._beam_factory.make_polarized_beam(
            sample_to_source=(0.0, 0.0, 1.0),
            wavelength=wavelength,
            polarization=(0, 1, 0),
            polarization_fraction=0.5,
        )

    def _scan(self):
        """Dummy scan for this stack"""

        nframes = self.get_num_images()
        image_range = (1, nframes)

        # Dummy values, not known from the header
        exposure_times = 0.0
        oscillation = (0, 0.5)
        epochs = [0] * nframes

        return self._scan_factory.make_scan(
            image_range, exposure_times, oscillation, epochs, deg=True
        )


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:]:
        print(FormatMRCstack.understand(arg))
