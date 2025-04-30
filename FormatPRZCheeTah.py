"""A Format class to recognise images from a CheeTah M3 electron detector
with a 2x2 array of Timepix modules, using the raw .prz format"""


from __future__ import annotations

import calendar
import os
import time
from math import sqrt

import numpy as np
from dxtbx import flumpy
from dxtbx.format.Format import Format
from dxtbx.format.FormatMultiImage import FormatMultiImage
from dxtbx.masking import mask_untrusted_rectangle
from dxtbx.model import SimplePxMmStrategy
from dxtbx.model.beam import Probe
from dxtbx.model.detector import Detector
from iotbx.detectors import SMVImage
from scitbx import matrix
from scitbx.array_family import flex


class FormatPRZCheeTahM3(Format):
    """ASI CheeTah M3"""

    @staticmethod
    def understand(image_file):

        # The understand method of subclasses is expensive as it loads
        # the whole data set, so only go that far if we have the right
        # extension
        if not image_file.lower().endswith(".prz"):
            return False

        return True

    def _start(self):
        """Open the file and cache the header and data arrays"""

        prz = np.load(self._image_file, mmap_mode="r", allow_pickle=True)
        self._header_dictionary = prz["meta_data"][0]
        self._raw_data = prz["data"]

    def _goniometer(self):
        """Return a model for a simple single-axis goniometer."""

        return self._goniometer_factory.known_axis((0, 1, 0))

    def _beam(self):
        """Return an unpolarized beam model"""

        h = 6.626e-34  # Js, Planck's constant
        m = 9.109e-31  # kg, electron mass
        e = 1.6021766208e-19  # C, electron charge
        c = 3e8  # m/s^2, speed

        V0 = self._header_dictionary.get("electron_gun.voltage", 0)
        # Default to e- wavelength at 200 keV if voltage set to zero
        if V0 == 0:
            V0 = 200000
        wavelength = h / sqrt(2 * m * e * V0 * (1 + e * V0 / (2 * m * c * c))) * 1e10

        return self._beam_factory.make_polarized_beam(
            sample_to_source=(0.0, 0.0, 1.0),
            wavelength=wavelength,
            polarization=(0, 1, 0),
            polarization_fraction=0.5,
            probe=Probe.electron,
        )

    def _detector(self):
        """4 panel detector, 55 micron pixels. The wider central cross
        pixels have already been split, so that the full image has 514×514
        pixels"""

        pixel_size = 1000 * self._header_dictionary["camera.pixel_size"]
        pixel_size = (pixel_size, pixel_size)
        image_size = self._header_dictionary["camera.detector_size"]
        panel_size = tuple([int(e / 2) for e in image_size])
        beam_centre = [(p * i) / 2 for p, i in zip(pixel_size, image_size)]
        trusted_range = (0, 65535)
        thickness = 0.3  # assume 300 mu thick

        # We want to be sure we are taking into account the module gap
        # correctly. Only proceed for 514×514 images, where we don't
        # worry about wide pixels
        assert image_size == (514, 514)

        d = self._detector_factory.simple(
            "PAD", 2440, beam_centre, "+x", "-y", pixel_size, image_size, trusted_range
        )
        p = d[0]
        p.set_type("SENSOR_PAD")
        p.set_thickness(thickness)
        p.set_material("Si")
        # Parallax and QE corrections are effectively disabled by setting
        # the simple pixel-to-millimetre strategy and a very high mu value.
        p.set_px_mm_strategy(SimplePxMmStrategy())
        p.set_mu(1e10)

        return d

    def get_static_mask(self):
        """Return the static mask that excludes the central cross of pixels."""

        mask = flex.bool(flex.grid((514, 514)), True)
        mask_untrusted_rectangle(mask, 0, 514, 255, 259)
        mask_untrusted_rectangle(mask, 255, 259, 0, 514)

        return (mask,)


class FormatPRZCheeTahM3images(FormatPRZCheeTahM3):
    """ASI CheeTah M3, single file per image"""

    @staticmethod
    def understand(image_file):

        prz = np.load(image_file, mmap_mode="r", allow_pickle=True)
        if prz["meta_data"][0]["type"].lower() == "stack":
            return False

        return True

    def _scan(self):
        """Return the scan information for this image."""

        exposure_time = self._header_dictionary["camera.exposure_time"]
        epoch = 0
        date_str = self._header_dictionary.get("camera.start_time", "")
        try:
            epoch = calendar.timegm(time.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f"))
        except ValueError:
            pass

        # Not sure this information is in the header
        osc_start = 0
        osc_range = 0.1

        return self._scan_factory.single_file(
            self._image_file, exposure_time, osc_start, osc_range, epoch
        )

    def get_raw_data(self):

        raw_data = flumpy.from_numpy(self._raw_data).as_double()
        return raw_data


class FormatPRZCheeTahM3stack(FormatMultiImage, FormatPRZCheeTahM3):
    """ASI CheeTah M3, image stack"""

    @staticmethod
    def understand(image_file):

        prz = np.load(image_file, allow_pickle=True)
        if prz["meta_data"][0]["type"].lower() != "stack":
            return False

        return True

    def __init__(self, image_file, **kwargs):
        from dxtbx import IncorrectFormatError

        if not self.understand(image_file):
            raise IncorrectFormatError(self, image_file)
        FormatMultiImage.__init__(self, **kwargs)
        Format.__init__(self, image_file, **kwargs)

    def get_num_images(self):
        return self._raw_data.shape[0]

    def get_goniometer(self, index=None):
        return Format.get_goniometer(self)

    def get_detector(self, index=None):
        return Format.get_detector(self)

    def get_beam(self, index=None):
        return Format.get_beam(self)

    def get_scan(self, index=None):
        if index is None:
            return Format.get_scan(self)
        else:
            scan = Format.get_scan(self)
            return scan[index]

    def get_image_file(self, index=None):
        return Format.get_image_file(self)

    def get_raw_data(self, index):

        raw_data = flumpy.from_numpy(self._raw_data[index, ...]).as_double()
        return raw_data

    def _scan(self):
        """Return the scan information for this stack."""

        exposure_time = self._header_dictionary["camera.exposure_time"]
        epoch = 0
        date_str = self._header_dictionary.get("camera.start_time", "")
        # epoch handling seems broken, so comment this out
        # try:
        #    epoch = calendar.timegm(time.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%f'))
        # except ValueError:
        #    pass

        # Not sure this information is in the header
        osc_start = 0
        osc_range = 0.1

        nframes = self.get_num_images()
        image_range = (1, nframes)
        epochs = [epoch] * nframes
        return self._scan_factory.make_scan(
            image_range, exposure_time, (osc_start, osc_range), epochs, deg=True
        )
