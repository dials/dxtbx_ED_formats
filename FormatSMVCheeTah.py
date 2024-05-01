"""A Format class to recognise images from a CheeTah T3 electron detector
with a 2x2 array of Timepix modules, converted to SMV."""


from __future__ import annotations

import calendar
import os
import time

from dxtbx.format.FormatSMVADSC import FormatSMVADSC
from dxtbx.model.beam import Probe
from dxtbx.model.detector import Detector
from iotbx.detectors import SMVImage
from scitbx import matrix


class FormatSMVCheeTahT3(FormatSMVADSC):
    """ASI CheeTah T3"""

    @staticmethod
    def understand(image_file):

        # Allow this class to override FormatSMVADSC with an environment variable
        if "FORCE_SMV_AS_CHEETAH_T3" in os.environ:
            return True

        return False

    def detectorbase_start(self):
        if not hasattr(self, "detectorbase") or self.detectorbase is None:
            self.detectorbase = SMVImage(self._image_file)
            self.detectorbase.open_file = self.open_file
            self.detectorbase.readHeader()

    def _goniometer(self):
        """Return a model for a simple single-axis goniometer."""

        return self._goniometer_factory.known_axis((0, 1, 0))

    def _beam(self):
        """Return an unpolarized beam model."""

        wavelength = float(self._header_dictionary["WAVELENGTH"])

        return self._beam_factory.make_polarized_beam(
            sample_to_source=(0.0, 0.0, 1.0),
            wavelength=wavelength,
            polarization=(0, 1, 0),
            polarization_fraction=0.5,
            probe=Probe.electron,
        )

    def _scan(self):
        """Return the scan information for this image."""
        exposure_time = float(self._header_dictionary["TIME"])
        epoch = None

        # PST, PDT timezones not recognised by default...

        epoch = 0
        try:
            date_str = self._header_dictionary["DATE"]
            date_str = date_str.replace("PST", "").replace("PDT", "")
        except KeyError:
            date_str = ""
        for format_string in ["%a %b %d %H:%M:%S %Y", "%a %b %d %H:%M:%S %Z %Y"]:
            try:
                epoch = calendar.timegm(time.strptime(date_str, format_string))
                break
            except ValueError:
                pass

        # assert(epoch)
        osc_start = float(self._header_dictionary["OSC_START"])
        osc_range = float(self._header_dictionary["OSC_RANGE"])

        return self._scan_factory.single_file(
            self._image_file, exposure_time, osc_start, osc_range, epoch
        )

    def _detector(self):
        """4 panel detector, 55 micron pixels except for pixels at the outer
        edge of each chip, which are 165 microns wide."""
        # The pixel size in the headers is wrong :-(
        # pixel_size = tuple([float(self._header_dictionary["PIXEL_SIZE"])] * 2)
        pixel_size = (0.055, 0.055)
        image_size = (
            int(self._header_dictionary["SIZE1"]),
            int(self._header_dictionary["SIZE2"]),
        )
        panel_size = tuple([int(e / 2) for e in image_size])

        # outer pixels have three times the width
        # panel_size_mm = (
        #    pixel_size[0] * 3 + (panel_size[0] - 2) * pixel_size[0],
        #    pixel_size[1] * 3 + (panel_size[1] - 2) * pixel_size[1],
        # )
        trusted_range = (0, 65535)
        thickness = 0.3  # assume 300 mu thick

        # Initialise detector frame
        fast = matrix.col((1.0, 0.0, 0.0))
        slow = matrix.col((0.0, -1.0, 0.0))
        beam_centre = (
            float(self._header_dictionary["BEAM_CENTER_X"]),
            float(self._header_dictionary["BEAM_CENTER_Y"]),
        )

        bx_px, by_px = beam_centre

        def px_to_mm(px, px_size_1d, panel_size_1d):
            # the beam centre is in pixels. We want to convert to mm, taking the
            # different size of outer pixels into account. Use this local function
            # to do that
            mm = 0
            if px > 1:  # add first outer pixel
                mm += px_size_1d * 3
            else:  # or fraction of first outer pixel
                mm += px * px_size_1d * 3
                return mm

            if px > panel_size_1d - 1:  # add full panel of inner pixels
                mm += (panel_size_1d - 2) * px_size_1d
            else:  # or fraction of inner pixels
                mm += (px - 1) * px_size_1d
                return mm

            if px > panel_size_1d:  # add second outer pixel
                mm += px_size_1d * 3
            else:  # or fraction of second outer pixel
                mm += (px - (panel_size_1d - 1)) * px_size_1d * 3
                return mm

            if px > panel_size_1d + 1:  # add first outer pixel of second panel
                mm += px_size_1d * 3
            else:  # or fraction of first outer pixel of second panel
                mm += (px - panel_size_1d) * px_size_1d * 3
                return mm

            if px > (2 * panel_size_1d - 1):  # add second full panel of inner pixels
                mm += (panel_size_1d - 2) * px_size_1d
                # plus remaining fraction of the second outer pixel
                mm += (px - (2 * panel_size_1d - 1)) * px_size_1d * 3
            else:  # or fraction of inner pixels of the second panel
                mm += (px - panel_size_1d - 1) * px_size_1d
            return mm

        bx_mm = px_to_mm(bx_px, pixel_size[0], panel_size[0])
        by_mm = px_to_mm(by_px, pixel_size[1], panel_size[1])

        # the beam centre is defined from the origin along fast, slow. To determine
        # the lab frame origin we place the beam centre down the -z axis
        dist = float(self._header_dictionary["DISTANCE"])
        cntr = matrix.col((0.0, 0.0, -1 * dist))
        orig = cntr - bx_mm * fast - by_mm * slow

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

    def get_raw_data(self):
        """Get the pixel intensities (i.e. read the image and return as a
        flex array of integers.)"""

        raw_data = self._get_endianic_raw_data(size=(512, 512))

        # split into separate panels
        self._raw_data = []
        d = self.get_detector()
        for panel in d:
            xmin, ymin, xmax, ymax = self.coords[panel.get_name()]
            self._raw_data.append(raw_data[ymin:ymax, xmin:xmax])

        return tuple(self._raw_data)
