#!/usr/bin/env python
# FormatCBFMiniTimepix.py
#  Copyright (C) (2017) STFC Rutherford Appleton Laboratory, UK.
#
#  Author: David Waterman.
#
#  This code is distributed under the BSD license. For dxtbx licensing see
#  https://github.com/cctbx/cctbx_project/blob/master/dxtbx/license.txt
#
"""Experimental implementation of a format class to recognise images
from a detector with a 2x2 array of Timepix modules used in electron diffraction
experiments. The only known example of such reports S/N 24-0109-F, but
currently this format will recognise any image where the detector is described
as TIMEPIX"""

from __future__ import division
import os

from dxtbx.format.FormatCBFMini import FormatCBFMini
from dxtbx.model.detector import Detector
from dxtbx.model import ScanFactory
from scitbx.array_family import flex

class FormatCBFMiniTimepix(FormatCBFMini):
  '''A class for reading mini CBF format Timepix images, and correctly
  constructing a model for the experiment from this.'''

  @staticmethod
  def understand(image_file):
    '''Check to see if this looks like a Timepix mini CBF format image,
    i.e. we can make sense of it.'''

    header = FormatCBFMini.get_cbf_header(image_file)

    for record in header.split('\n'):
      if '# Detector' in record and \
             'TIMEPIX' in record:
        return True

    return False

  def __init__(self, image_file, **kwargs):
    '''Initialise the image structure from the given file, including a
    proper model of the experiment.'''

    from dxtbx import IncorrectFormatError
    if not self.understand(image_file):
      raise IncorrectFormatError(self, image_file)

    FormatCBFMini.__init__(self, image_file, **kwargs)

    return

  def _start(self):
    FormatCBFMini._start(self)

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
    thickness = 0.3 # assume 300 mu thick. This is actually in the header too
    # so could take it from there

    # Initialise detector frame - dummy origin to place detector at the header
    # distance along the canonical beam direction
    distance = float(self._cif_header_dictionary['Detector_distance'].split()[0]) * 1000
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

  def _scan(self):
    '''Dummy scan for this image'''

    format = self._scan_factory.format('CBF')

    exposure_time = float(
        self._cif_header_dictionary['Exposure_period'].split()[0])

    fname = os.path.split(self._image_file)[-1]
    index = int(fname.split('_')[-1].split('.')[0])

    return ScanFactory.make_scan(image_range=(index, index),
                                 exposure_times=exposure_time,
                                 oscillation=(0,1),
                                 epochs={index:0})

  def get_raw_data(self):

    raw_data = self.read_cbf_image(self._image_file)
    image_size = (512,512)
    raw_data.reshape(flex.grid(image_size[1], image_size[0]))

    self._raw_data = []

    d = self.get_detector()

    for panel in d:
      xmin, ymin, xmax, ymax = self.coords[panel.get_name()]
      self._raw_data.append(raw_data[ymin:ymax,xmin:xmax])

    return tuple(self._raw_data)


if __name__ == '__main__':

  import sys

  for arg in sys.argv[1:]:
    print FormatCBFMiniTimepix.understand(arg)
