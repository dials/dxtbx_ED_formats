#!/usr/bin/env python
# FormatTimepixRaw512x512.py
#  Copyright (C) (2016) STFC Rutherford Appleton Laboratory, UK.
#
#  Author: David Waterman.
#
#  This code is distributed under the BSD license. For dxtbx licensing see
#  https://github.com/cctbx/cctbx_project/blob/master/dxtbx/license.txt
#
"""Experimental implementation of a format class to recognise images
from a detector with a 2x2 array of Timepix modules, used in electron
diffraction experiments"""

from __future__ import division
import os

from dxtbx.format.Format import Format
from dxtbx.model import ScanFactory
from dxtbx.model.detector import Detector

class FormatTimepixRaw512x512(Format):
  '''An image reading class for Timepix raw format images. A description
  of the detector is given in this paper, but this contains an error, which
  is that the gap between tiles is actually designed to be 165 um, not 175 um:

    http://dx.doi.org/10.1107/S2053273315022500

  We have limited information about the raw data format at present. Tim Gruene
  notes:

    The Timepix 'bin' format can be displayed with adxv -swab -nx 512 -ny
    512 -skip 15 -ushort frame_value_1.bin i.e. they have a 15B header,
    encode 512x512 pixel by unsigned short in little endian.

    The inner pixels are 0.055mu x 0.055mu, but the outermost pixel frame is
    165mu x 165mu. I therefore blank pixels 256 and 257, the cross between
    the chips.

  The header does not contain useful information about the geometry, therefore
  we will construct dummy objects and expect to override on import using
  site.phil.'''

  @staticmethod
  def understand(image_file):
    '''Check to see if this looks like a Timepix raw format image. Not much
    to go on here. Let's use the file size and some bytes from the header that
    appear to be the same in the few datasets we've seen.'''

    with open(image_file, 'rb') as f:
      header = f.read(15)
      if os.fstat(f.fileno()).st_size != 524303: return False

    fingerprint = header[0] + header[2:8] + header[10:]
    if fingerprint != '\x00\x00\x0b\x03\x02\x02\x00\x00\x00\x00\x02\x00':
      return False

    return True

  def detectorbase_start(self): pass
  def _start(self):
    '''Open the image file and read the image header'''

    self._header_size = 15
    self._header_bytes = FormatTimepixRaw512x512.open_file(
        self._image_file, 'rb').read(self._header_size)

    return

  def get_raw_data(self):
    '''Get the pixel intensities'''

    from boost.python import streambuf
    from dxtbx import read_uint16_bs
    from scitbx.array_family import flex

    f = FormatTimepixRaw512x512.open_file(self._image_file, 'rb')
    f.read(self._header_size)

    raw_data = read_uint16_bs(streambuf(f), 512*512)
    image_size = (512,512)
    raw_data.reshape(flex.grid(image_size[1], image_size[0]))

    self._raw_data = []

    d = self.get_detector()

    for panel in d:
      xmin, ymin, xmax, ymax = self.coords[panel.get_name()]
      self._raw_data.append(raw_data[ymin:ymax,xmin:xmax])

    return tuple(self._raw_data)

  def _goniometer(self):
    '''Dummy goniometer'''

    return self._goniometer_factory.single_axis_reverse()

  def _detector(self):
    '''Dummy detector'''
    from scitbx import matrix

    # 55 mu pixels
    pixel_size = 0.055, 0.055
    image_size = 512, 512
    trusted_range = (-1, 65535)
    material = 'Si'
    thickness = 0.3 # assume 300 mu thick

    # Initialise detector frame - dummy origin to place detector at 100 mm
    # along canonical beam direction
    fast = matrix.col((1.0, 0.0, 0.0))
    slow = matrix.col((0.0, -1.0, 0.0))
    cntr = matrix.col((0.0, 0.0, -100.0))

    # shifts to go from the centre to the origin - outer pixels are 0.165 mm
    off_x = (image_size[0] / 2 - 2) * pixel_size[0]
    off_x += 2 * 0.165
    shift_x = -1. * fast * off_x
    off_y = (image_size[1] / 2 - 2) * pixel_size[1]
    off_y += 2 * 0.165
    shift_y = -1. * slow * off_y
    orig = cntr + shift_x + shift_y

    d = Detector()

    root = d.hierarchy()
    root.set_local_frame(
      fast.elems,
      slow.elems,
      orig.elems)

    self.coords = {}
    panel_idx = 0

    # set panel extent in pixel numbers and x, y mm shifts. Note that the
    # outer pixels are 0.165 mm in size. These are excluded from the panel
    # extents.
    pnl_data = []
    pnl_data.append({'xmin':1, 'ymin':1,
                     'xmax':255, 'ymax':255,
                     'xmin_mm': 1 * 0.165,
                     'ymin_mm': 1 * 0.165})
    pnl_data.append({'xmin':257, 'ymin':1,
                     'xmax':511, 'ymax':255,
                     'xmin_mm': 3 * 0.165 + (511 - 257) * pixel_size[0],
                     'ymin_mm': 1 * 0.165})
    pnl_data.append({'xmin':1, 'ymin':257,
                     'xmax':255, 'ymax':511,
                     'xmin_mm': 1 * 0.165,
                     'ymin_mm': 3 * 0.165 + (511 - 257) * pixel_size[1]})
    pnl_data.append({'xmin':257, 'ymin':257,
                     'xmax':511, 'ymax':511,
                     'xmin_mm': 3 * 0.165 + (511 - 257) * pixel_size[0],
                     'ymin_mm': 3 * 0.165 + (511 - 257) * pixel_size[1]})

    # redefine fast, slow for the local frame
    fast = matrix.col((1.0, 0.0, 0.0))
    slow = matrix.col((0.0, 1.0, 0.0))

    for ipanel, pd in enumerate(pnl_data):
      xmin = pd['xmin']
      xmax = pd['xmax']
      ymin = pd['ymin']
      ymax = pd['ymax']
      xmin_mm = pd['xmin_mm']
      ymin_mm = pd['ymin_mm']

      origin_panel = fast * xmin_mm + slow * ymin_mm

      panel_name = "Panel%d" % panel_idx
      panel_idx += 1

      p = d.add_panel()
      p.set_type('SENSOR_PAD')
      p.set_name(panel_name)
      p.set_raw_image_offset((xmin, ymin))
      p.set_image_size((xmax-xmin, ymax-ymin))
      p.set_trusted_range(trusted_range)
      p.set_pixel_size((pixel_size[0], pixel_size[1]))
      p.set_thickness(thickness)
      p.set_material('Si')
      #p.set_mu(mu)
      #p.set_px_mm_strategy(ParallaxCorrectedPxMmStrategy(mu, t0))
      p.set_local_frame(
        fast.elems,
        slow.elems,
        origin_panel.elems)
      p.set_raw_image_offset((xmin, ymin))
      self.coords[panel_name] = (xmin, ymin, xmax, ymax)

    return d

  def _beam(self):
    '''Dummy unpolarized beam, energy 200 keV'''

    wavelength = 0.02508
    return self._beam_factory.make_polarized_beam(
        sample_to_source=(0.0, 0.0, 1.0),
        wavelength=wavelength,
        polarization=(0, 1, 0),
        polarization_fraction=0.5)

  def _scan(self):
    '''Dummy scan for this image'''

    fname = os.path.split(self._image_file)[-1]
    index = int(fname.split('_')[-1].split('.')[0])
    return ScanFactory.make_scan(
                (index, index), 0.0, (0,1),
                {index:0})
