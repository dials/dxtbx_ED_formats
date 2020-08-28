"""Format class for data stored in the FEI EMD (Velox) format"""

from __future__ import absolute_import, division, print_function

import os
import h5py
import json
from scitbx.array_family import flex
import numpy
from dxtbx.format.Format import Format
from dxtbx.format.FormatHDF5 import FormatHDF5
from dxtbx.format.FormatMultiImage import FormatMultiImage
from dxtbx import IncorrectFormatError


def get_metadata(metadata):
    mds = []
    for i in range(metadata.shape[1]):
        metadata_array = metadata[:, i].T
        mdata_string = metadata_array.tostring().decode("utf-8")
        mds.append(json.loads(mdata_string.rstrip("\x00")))

    return mds


# get_metadata()


def analyse_angle(metadata):
    alphas = []

    for i, md in enumerate(metadata):
        alpha = numpy.rad2deg(float(md["Stage"]["AlphaTilt"]))
        alphas.append(alpha)

    if len(alphas) < 2:
        return [0, 0], 0.0

    d_alphas = numpy.diff(alphas)
    q25, q50, q75 = numpy.percentile(d_alphas, [25, 50, 75])
    iqr = q75 - q25
    iqrc = 1.5
    lowlim, highlim = q25 - iqrc * iqr, q75 + iqrc * iqr
    d_alphas2 = d_alphas[
        numpy.where(numpy.logical_and(d_alphas > lowlim, d_alphas < highlim))
    ]  # outlier rejected
    d_alpha_z = abs(d_alphas - numpy.mean(d_alphas2)) / numpy.std(d_alphas2)

    valid_range = [0, len(metadata) - 1]
    for i in range(len(metadata) - 1):
        if d_alpha_z[i] < 3:
            break
        valid_range[0] = i + 1

    for i in reversed(range(len(metadata) - 1)):
        if d_alpha_z[i] < 3:
            break
        valid_range[1] = i

    if valid_range[0] > valid_range[1]:
        valid_range = [0, len(metadata) - 1]  # reset

    mean_alpha_step = (alphas[valid_range[1]] - alphas[valid_range[0]]) / (
        valid_range[1] - valid_range[0]
    )

    return valid_range, mean_alpha_step


class FormatVelox(FormatHDF5):
    @staticmethod
    def understand(image_file):
        with h5py.File(image_file, "r") as h5_handle:
            try:
                version = h5_handle["Version"]
            except KeyError:
                return False
            v = json.loads(version[()][0].decode("utf-8"))
            if "Velox" in v["format"]:
                return True

        return False

    def __init__(self, image_file, **kwargs):
        if not self.understand(image_file):
            raise IncorrectFormatError(self, image_file)
        FormatMultiImage.__init__(self, **kwargs)
        Format.__init__(self, image_file, **kwargs)

    @staticmethod
    def _read_metadata(image_file):
        h = h5py.File(image_file, "r")
        ret = {}

        image_path = h["/Data/Image"]
        assert len(image_path.keys()) == 1
        k = list(image_path.keys())[0]

        ret["image_file"] = image_file
        ret["file_handle"] = h
        ret["data_path"] = "/Data/Image/%s/Data" % k
        ret["metadata_path"] = "/Data/Image/%s/Metadata" % k

        metadata = get_metadata(h[ret["metadata_path"]])
        valid_range, mean_alpha_step = analyse_angle(metadata)
        data = h[ret["data_path"]]

        ret["n_frames"] = data.shape[2]
        ret["valid_range"] = valid_range
        ret["mean_alpha_step"] = mean_alpha_step
        ret["width"], ret["height"] = data.shape[:2]
        ret["binning"] = (
            int(metadata[0]["BinaryResult"]["ImageSize"]["width"]) // ret["width"]
        )

        h, m0, e, c = 6.62607004e-34, 9.10938356e-31, 1.6021766208e-19, 299792458.0
        voltage = float(metadata[0]["Optics"]["AccelerationVoltage"])
        ret["wavelength"] = (
            h
            / numpy.sqrt(2 * m0 * e * voltage * (1.0 + e * voltage / 2.0 / m0 / c ** 2))
            * 1.0e10
        )

        return ret

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
        # self._image_size = self._data.shape[0:2]
        # self._num_images = self._data.shape[2]
        self._header_dictionary = self._read_metadata(self._image_file)

    def get_raw_data(self, index):
        d = self._data[:, :, index].astype("int32")
        return flex.int(d)

    def _goniometer(self):
        """Dummy goniometer, 'vertical' as the images are viewed. Not completely
        sure about the handedness yet"""

        if self._header_dictionary["mean_alpha_step"] > 0:  # XXX is this really OK??
            return self._goniometer_factory.known_axis((0, -1, 0))
        else:
            return self._goniometer_factory.known_axis((0, 1, 0))

    def _detector(self):
        """Dummy detector"""

        image_size = (
            self._header_dictionary["width"],
            self._header_dictionary["height"],
        )
        binning = self._header_dictionary["binning"]

        # Dummy pixel size: assume 14 um for the Ceta unbinned
        pixel_size = 0.014 * binning

        # Dummy distance of 2.0m
        distance = 2000.0

        # Get detector-specific details for TF detectors as discussed with
        # Lingbo Yu. Ceta has gain of > 26 and Ceta and Falcon III both saturate
        # at about 8000.0 for binning=1
        camera = self._extract_lone_item(
            self._h5_handle["Operations/CameraInputOperation"]
        )
        camera = json.loads(camera[()][0].decode("utf-8"))
        camera = camera["cameraName"].lower()

        if "ceta" in camera:
            gain = 26.0
            saturation = 8000 * binning ** 2
        elif "falcon" in camera:
            gain = 1.0
            saturation = 8000 * binning ** 2
        else:
            gain = 1.0
            saturation = 1e6
        trusted_range = (-1000, saturation)

        # Beam centre not in the header - set to the image centre
        beam_centre = [(pixel_size * i) / 2 for i in image_size]
        detector = self._detector_factory.simple(
            "PAD",
            distance,
            beam_centre,
            "+x",
            "-y",
            (pixel_size, pixel_size),
            image_size,
            trusted_range,
        )

        for panel in detector:
            panel.set_gain(gain)
        return detector

    def _beam(self):

        return self._beam_factory.make_polarized_beam(
            sample_to_source=(0.0, 0.0, 1.0),
            wavelength=self._header_dictionary["wavelength"],
            polarization=(0, 1, 0),
            polarization_fraction=0.5,
        )

    def get_num_images(self):
        return self._header_dictionary["n_frames"]

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

        image_range = (1, self.get_num_images())

        # Dummy values, not known from the header
        alpha = 0.0
        exposure_times = 0.0
        osc_step = abs(self._header_dictionary["mean_alpha_step"])
        oscillation = (alpha, osc_step)
        epochs = [0] * self.get_num_images()

        return self._scan_factory.make_scan(
            image_range, exposure_times, oscillation, epochs, deg=True
        )
