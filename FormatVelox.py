"""Format class for data stored in the FEI EMD (Velox) format"""

from __future__ import absolute_import, division, print_function

import os
import h5py
import json
from scitbx.array_family import flex
from dxtbx.format.Format import Format
from dxtbx.format.FormatHDF5 import FormatHDF5
from dxtbx.format.FormatMultiImage import FormatMultiImage
from dxtbx import IncorrectFormatError

class FormatVelox(FormatHDF5):

    @staticmethod
    def understand(image_file):
        with h5py.File(image_file, "r") as h5_handle:
            try:
                version = h5_handle["Version"]
            except KeyError:
                return False
            v = json.loads(version[()][0].decode('utf-8'))
            if "Velox" in v["format"]:
                return True

        return False

    def __init__(self, image_file, **kwargs):
        if not self.understand(image_file):
            raise IncorrectFormatError(self, image_file)
        FormatMultiImage.__init__(self, **kwargs)
        Format.__init__(self, image_file, **kwargs)

    @staticmethod
    def _extract_lone_item(group):
        """Extract a sub-group or dataset from a group with a single key"""
        assert len(group) == 1
        k = list(group.keys())[0]
        return group[k]

    def _start(self):
        self._h5_handle = h5py.File(self.get_image_file(), "r")
        image_group = self._extract_lone_item(self._h5_handle["Data/Image"])
        self._data = image_group["Data"]
        self._image_size = self._data.shape[0:2]
        self._num_images = self._data.shape[2]

    def get_raw_data(self, index):
        d = self._data[:, :, index].astype("int32")
        return flex.int(d)

    def _goniometer(self):
        """Dummy goniometer"""

        return self._goniometer_factory.known_axis((0, 1, 0))


    def _detector(self):
        """Dummy detector"""

        # Dummy pixel size: assume 14 um for the Ceta unbinned
        physical_pixel = 1.4e-5
        if self._image_size == (2048, 2048):
            binning = 2.0
        else:
            binning = 1.0
        pixel_size = physical_pixel * 1000.0 * binning

        # Dummy distance of 2.0m
        distance = 2000.0

        # Get detector-specific details for TF detectors as discussed with
        # Lingbo Yu. Ceta has gain of > 26 and Ceta and Falcon III both saturate
        # at about 8000.0 for binning=1
        camera = self._extract_lone_item(self._h5_handle["Operations/CameraInputOperation"])
        camera = json.loads(camera[()][0].decode('utf-8'))
        camera = camera["cameraName"].lower()

        if 'ceta' in camera:
            gain = 26.0
            saturation = 8000 * binning**2
        elif 'falcon' in camera:
            gain = 1.0
            saturation = 8000 * binning**2
        else:
            gain = 1.0
            saturation = 1e6
        trusted_range = (-1000, saturation)

        # Beam centre not in the header - set to the image centre
        beam_centre = [(pixel_size * i) / 2 for i in self._image_size]
        detector = self._detector_factory.simple(
            "PAD",
            distance,
            beam_centre,
            "+x",
            "-y",
            (pixel_size, pixel_size),
            self._image_size,
            trusted_range,
        )

        for panel in detector: panel.set_gain(gain)
        return detector

    def _beam(self):
        """Unpolarized beam model"""

        # Default to 200 keV
        wavelength = 0.02508
        return self._beam_factory.make_polarized_beam(
            sample_to_source=(0.0, 0.0, 1.0),
            wavelength=wavelength,
            polarization=(0, 1, 0),
            polarization_fraction=0.5,
        )

    def get_num_images(self):
        return self._num_images

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

    def _scan(self):
        """Dummy scan for this stack"""

        image_range = (1, self._num_images)

        # Dummy values, not known from the header
        alpha = 0.0
        exposure_times = 0.0
        oscillation = (alpha, 1.0)
        epochs = [0] * self._num_images

        return self._scan_factory.make_scan(
            image_range, exposure_times, oscillation, epochs, deg=True
        )
