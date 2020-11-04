from __future__ import absolute_import, division, print_function

import os

from dxtbx.format.FormatCBF import FormatCBF
from dxtbx.model import ScanFactory


class FormatCBFJungfrauVIE01(FormatCBF):
    """Work-in-progress image reading class for CBF format images with a
    minimal header from an electron microscope with a Jungfrau detector
    at University of Vienna. Largely based upon FormatCBFMini.py"""

    @staticmethod
    def understand(image_file):
        """Check to see if this looks like an CBF format image, i.e. we can
        make sense of it."""

        header = FormatCBF.get_cbf_header(image_file)
        if "VIE-01" not in header:
            return False

        return True

    def __init__(self, image_file, **kwargs):
        """Initialise the image structure from the given file."""

        from dxtbx import IncorrectFormatError

        if not self.understand(image_file):
            raise IncorrectFormatError(self, image_file)

        FormatCBF.__init__(self, image_file, **kwargs)

        self._raw_data = None

    def _start(self):
        """Open the image file, read the image header, copy it into a
        dictionary for future reference."""

        FormatCBF._start(self)
        cif_header = FormatCBF.get_cbf_header(self._image_file)

        self._header_dictionary = {}

        # Get the few items that are in the header
        for record in cif_header.split("\n"):
            if "Wavelength" in record:
                self._header_dictionary["Wavelength"] = " ".join(record.split()[-2:])
            if "Pixel_size" in record:
                tokens = record.replace("=", "").replace(":", "").split()
                self._header_dictionary["Pixel_size"] = " ".join(tokens[2:])

        # Populate other items manually with dummy values
        self._header_dictionary["Detector_distance"] = "0.2 m"
        self._header_dictionary["Count_cutoff"] = "1000000 counts"
        self._header_dictionary["Phi"] = "0.0000 deg."

        for record in self._mime_header.split("\n"):
            if not record.strip():
                continue
            token, value = record.split(":")
            self._header_dictionary[token.strip()] = value.strip()

    def _detector(self):
        """Return a dummy model for the detector"""

        distance = float(self._header_dictionary["Detector_distance"].split()[0])

        wavelength = float(self._header_dictionary["Wavelength"].split()[0])

        nx = int(self._header_dictionary["X-Binary-Size-Fastest-Dimension"])
        ny = int(self._header_dictionary["X-Binary-Size-Second-Dimension"])

        beam_x, beam_y = nx / 2.0, ny / 2.0

        pixel_xy = (
            self._header_dictionary["Pixel_size"]
            .replace("m", "")
            .replace("x", "")
            .split()
        )

        pixel_x, pixel_y = map(float, pixel_xy)

        overload = int(self._header_dictionary["Count_cutoff"].split()[0])
        underload = -1

        detector = self._detector_factory.simple(
            "PAD",
            distance * 1000.0,
            (beam_x * pixel_x * 1000.0, beam_y * pixel_y * 1000.0),
            "+x",
            "-y",
            (1000 * pixel_x, 1000 * pixel_y),
            (nx, ny),
            (underload, overload),
            [],
        )

        for panel in detector:
            panel.set_gain(200)

        return detector

    def _goniometer(self):
        """Return a model for a simple single-axis goniometer."""

        # From T. Gruene:
        # >the rotation axis is
        # >0.4983416  -0.8668326   0.0060932
        # >when the detector distance is about 725mm ('D500', low res data) and
        # >0.56652  -0.82390   0.00335
        # >when the detector distance is about 279mm ('D190', high res data).
        # Here we take the mean of these directions.

        direction = (0.5329242414415575, -0.846149760997591, 0.004725975842100904)
        return self._goniometer_factory.known_axis(direction)

    def _beam(self):
        """Dummy unpolarized beam"""

        wavelength = float(self._header_dictionary["Wavelength"].split()[0])

        return self._beam_factory.make_polarized_beam(
            sample_to_source=(0.0, 0.0, 1.0),
            wavelength=wavelength,
            polarization=(0, 1, 0),
            polarization_fraction=0.5,
        )

    def _scan(self):
        """Dummy scan for this image"""

        fname = os.path.split(self._image_file)[-1]
        index = int(fname.split("_")[-1].split(".")[0])
        return ScanFactory.make_scan((index, index), 0.0, (0, 1), {index: 0})

    def read_cbf_image(self, cbf_image):
        from cbflib_adaptbx import uncompress
        import binascii

        start_tag = binascii.unhexlify("0c1a04d5")

        data = self.open_file(cbf_image, "rb").read()
        data_offset = data.find(start_tag) + 4
        cbf_header = data[: data_offset - 4].decode()

        fast = 0
        slow = 0
        length = 0
        byte_offset = False
        no_compression = False

        for record in cbf_header.split("\n"):
            if "X-Binary-Size-Fastest-Dimension" in record:
                fast = int(record.split()[-1])
            elif "X-Binary-Size-Second-Dimension" in record:
                slow = int(record.split()[-1])
            elif "X-Binary-Number-of-Elements" in record:
                length = int(record.split()[-1])
            elif "X-Binary-Size:" in record:
                size = int(record.split()[-1])
            elif "conversions" in record:
                if "x-CBF_BYTE_OFFSET" in record:
                    byte_offset = True
                elif "x-CBF_NONE" in record:
                    no_compression = True

        assert length == fast * slow

        if byte_offset:
            pixel_values = uncompress(
                packed=data[data_offset : data_offset + size], fast=fast, slow=slow
            )
        elif no_compression:
            from boost.python import streambuf

            try:
                from dxtbx.ext import read_int32
            except ImportError:
                from dxtbx import read_int32
            from scitbx.array_family import flex

            assert len(self.get_detector()) == 1
            f = self.open_file(self._image_file)
            f.read(data_offset)
            pixel_values = read_int32(streambuf(f), int(slow * fast))
            pixel_values.reshape(flex.grid(slow, fast))

        else:
            raise ValueError(
                "Uncompression of type other than byte_offset or none "
                " is not supported (contact authors)"
            )

        return pixel_values

    def get_raw_data(self):
        if self._raw_data is None:
            data = self.read_cbf_image(self._image_file)
            self._raw_data = data

        return self._raw_data
