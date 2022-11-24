from __future__ import annotations

import math

import h5py

from libtbx import Auto

import dxtbx.nexus
from dxtbx.format.FormatNexus import FormatNexus
from dxtbx.masking import GoniometerMaskerFactory
from dxtbx.masking import mask_untrusted_circle, mask_untrusted_polygon

from scitbx.array_family import flex


def get_static_mask(nxdetector: nxmx.NXdetector) -> tuple[flex.bool, ...]:
    """Return the static mask for an NXdetector.

    This will be a tuple of flex.bool, of length equal to the number of modules. The
    result is intended to be compatible with the get_static_mask() method of dxtbx
    format classes.

    XXX Modified from the version in dxtbx.nexus to take the pixel mask from a
    XXX "detectorSpecific" group.
    """
    pixel_mask = nxdetector["detectorSpecific"].get("pixel_mask")
    assert pixel_mask and pixel_mask.ndim == 2
    all_slices = dxtbx.nexus.get_detector_module_slices(nxdetector)

    # Add mask for shadow of vacuum flange(?) on detector
    mask = flex.bool(flex.grid(pixel_mask.shape), True)
    for coord in [(201, 210), (202, 846), (836, 208), (838, 845)]:
        mask_untrusted_circle(mask, coord[0], coord[1], 194)
    vertices = [(7, 210), (201, 16), (836, 14), (1028, 174), (1028, 886), (838, 1039), (202, 1040), (8, 836)]
    polygon = flex.vec2_double(vertices)
    mask_untrusted_polygon(mask, polygon)
    result = []
    for slices in all_slices:
        result.append((dxtbx.format.nexus.dataset_as_flex(pixel_mask, slices) == 0) & ~mask)

    return tuple(result)


class FormatNexusSINGLA(FormatNexus):
    """Read HDF5 data from the SINGLA detector that partially adheres to NeXus format.
    Much of this is copypasta from elsewhere"""

    _cached_file_handle = None

    @staticmethod
    def understand(image_file):
        with h5py.File(image_file, swmr=True) as handle:
            name = dxtbx.nexus.nxmx.h5str(
                handle["/entry/instrument/detector/description"][()]
            )
        if name and ("SINGLA" in name):
            return True
        return False

    def __init__(self, image_file, **kwargs):
        """Initialise the image structure from the given file."""
        super().__init__(image_file, **kwargs)

    def _start(self):
        self._static_mask = None

        with h5py.File(self._image_file, swmr=True) as fh:
            nxmx = dxtbx.nexus.nxmx.NXmx(fh)
            nxsample = nxmx.entries[0].samples[0]
            nxinstrument = nxmx.entries[0].instruments[0]
            nxdetector = nxinstrument.detectors[0]
            nxbeam = nxinstrument.beams[0]

            # There is a bug that leads to data_size being reversed. Check this
            # file is correct
            if tuple(nxinstrument["detector/module/data_size"]) == (1028, 1062):
                raise ValueError(
                    f"The data_size values are reversed in "
                    f"{self._image_file}. Please correct this file."
                )

            # The following fail because of non-existent objects
            # self._goniometer_model = dxtbx.nexus.get_dxtbx_goniometer(nxsample)
            # self._beam_model = dxtbx.nexus.get_dxtbx_beam(nxbeam)
            # self._detector_model = dxtbx.nexus.get_dxtbx_detector(nxdetector, nxbeam)
            # self._scan_model = dxtbx.nexus.get_dxtbx_scan(nxsample, nxdetector)
            # self._static_mask = dxtbx.nexus.get_static_mask(nxdetector)

            # Salvage what we can
            self._static_mask = get_static_mask(nxdetector)
            self._bit_depth_readout = nxdetector.bit_depth_readout
            self._saturation_value = nxdetector.saturation_value
            module = nxdetector.modules[0]
            self._image_size = tuple(map(int, module.data_size[::-1]))
            self._pixel_size = (
                module.fast_pixel_direction[()].to("mm").magnitude.item(),
                module.slow_pixel_direction[()].to("mm").magnitude.item(),
            )

            nxdata = nxmx.entries[0].data[0]
            if nxdata.signal:
                data = nxdata[nxdata.signal]
            else:
                data = list(nxdata.values())[0]
            self._num_images, *_ = data.shape

    def _goniometer(self):
        """Dummy goniometer, 'vertical' as the images are viewed."""

        return self._goniometer_factory.known_axis((0, 1, 0))

    def _beam(self, index=None):
        """Dummy beam, energy 200 keV, unpolarised"""

        wavelength = 0.02508
        return self._beam_factory.make_polarized_beam(
            sample_to_source=(0.0, 0.0, 1.0),
            wavelength=wavelength,
            polarization=(0, 1, 0),
            polarization_fraction=0.5,
        )

    def get_num_images(self) -> int:
        return self._num_images

    def get_static_mask(self, index=None, goniometer=None):
        return self._static_mask

    def _detector(self):
        """Dummy detector"""

        pixel_size = self._pixel_size
        image_size = self._image_size
        dyn_range = self._bit_depth_readout
        trusted_range = (-1, self._saturation_value)
        beam_centre = [(p * i) / 2 for p, i in zip(pixel_size, image_size)]
        d = self._detector_factory.simple(
            "PAD", 2440, beam_centre, "+x", "-y", pixel_size, image_size, trusted_range
        )
        return d

    def get_raw_data(self, index):
        if self._cached_file_handle is None:
            self._cached_file_handle = h5py.File(self._image_file, swmr=True)

        nxmx = dxtbx.nexus.nxmx.NXmx(self._cached_file_handle)
        nxdata = nxmx.entries[0].data[0]
        nxdetector = nxmx.entries[0].instruments[0].detectors[0]

        if nxdata.signal:
            data = nxdata[nxdata.signal]
        else:
            data = list(nxdata.values())[0]
        all_data = []
        for module_slices in dxtbx.nexus.get_detector_module_slices(nxdetector):
            slices = [slice(index, index + 1, 1)]
            slices.extend(module_slices)

            data_as_flex = dxtbx.format.nexus.dataset_as_flex(data, tuple(slices))
            # Convert a slice of a 3- or 4-dimension array to a 2D array
            data_as_flex.reshape(flex.grid(data_as_flex.all()[-2:]))
            all_data.append(data_as_flex)
        raw_data = tuple(all_data)

        if self._bit_depth_readout:
            # if 32 bit then it is a signed int, I think if 8, 16 then it is
            # unsigned with the highest two values assigned as masking values
            if self._bit_depth_readout == 32:
                top = 2**31
            else:
                top = 2**self._bit_depth_readout
            for data in raw_data:
                d1d = data.as_1d()
                d1d.set_selected(d1d == top - 1, -1)
                d1d.set_selected(d1d == top - 2, -2)
        return raw_data

    def _scan(self):
        """Dummy scan model for this stack"""

        alpha = 0.0
        dalpha = 1.0
        exposure = 0.0

        oscillation = (alpha, dalpha)
        nframes = self.get_num_images()
        image_range = (1, nframes)
        epochs = [0] * nframes

        return self._scan_factory.make_scan(
            image_range, exposure, oscillation, epochs, deg=True
        )
