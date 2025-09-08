"""Experimental implementation of a format class to recognise images
from a Quantum Detectors Merlin device in HDF5 file format"""

from __future__ import annotations

import h5py
from dxtbx.format.Format import Format
from dxtbx.format.FormatHDF5 import FormatHDF5
from dxtbx.masking import mask_untrusted_rectangle
from dxtbx.model.beam import Probe
from scitbx.array_family import flex


class FormatHDF5Merlin(FormatHDF5):
    @staticmethod
    def understand(image_file):
        with h5py.File(image_file, "r") as h5_handle:
            try:
                data = h5_handle["data"]
            except KeyError:
                return False

            if data.shape[1:3] == (515, 515):
                return True

        return False

    def _start(self):
        self._h5_handle = h5py.File(self.get_image_file(), "r")

    def _scan(self):
        """Scan model for this stack, filling out any unavailable items with
        dummy values. Set oscillation to 0.0, as these are still images."""

        alpha = 0.0
        dalpha = 0.0
        exposure = 0.0

        oscillation = (alpha, dalpha)
        nframes = self.get_num_images()
        image_range = (1, nframes)
        epochs = [0] * nframes

        return self._scan_factory.make_scan(
            image_range, exposure, oscillation, epochs, deg=True
        )

    def _beam(self):
        """Dummy unpolarized beam, energy 300 keV"""

        wavelength = 0.0197
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
        image_size = 515, 515
        dyn_range = 12
        trusted_range = (0, 2**dyn_range - 1)
        beam_centre = [(p * i) / 2 for p, i in zip(pixel_size, image_size)]
        # Following discussion with QD, I think the origin of the image array
        # is at the bottom left (as viewed by dials.image_viewer), with fast
        # increasing +X and slow increasing +Y. See https://github.com/dials/dxtbx_ED_formats/pull/11
        d = self._detector_factory.simple(
            "PAD", 1348, beam_centre, "+x", "+y", pixel_size, image_size, trusted_range
        )
        return d

    def _goniometer(self):
        """Dummy goniometer. These are still images, so the goniometer has no
        practical effect on processing. However some tools, such as dials.show
        appear to need a model here. A way around that would be to inherit from
        FormatStill, but it seems this is incompatible with FormatMultiImage."""

        return self._goniometer_factory.known_axis((0, 1, 0))

    def get_raw_data(self, index):
        if index is None:
            index = 0
        data = self._h5_handle["data"]
        im = data[index, :, :].astype("int32")  # convert from int16
        return flex.int(im)

    def get_static_mask(self):
        """Return the static mask that excludes the central cross of pixels."""

        mask = flex.bool(flex.grid((515, 515)), True)
        mask_untrusted_rectangle(mask, 0, 515, 255, 260)
        mask_untrusted_rectangle(mask, 255, 260, 0, 515)

        return (mask,)

    def get_num_images(self):
        return self._h5_handle["data"].shape[0]

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
