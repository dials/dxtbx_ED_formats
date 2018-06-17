#!/usr/bin/env python
# FormatSEReBICpedestal.py

# Experimental format for TIA .ser files used by FEI microscope at eBIC. This
# version adds a pedestal level to the image data as determined by the
# environment variable ADD_PEDESTAL. A warning will be issued to counter
# against the use of this format by mistake

from __future__ import absolute_import, division, print_function
import os
from dxtbx.format.FormatSEReBIC import FormatSEReBIC
import logging
logger = logging.getLogger("dials")

class FormatSEReBICpedestal(FormatSEReBIC):

  def __init__(self, image_file, **kwargs):

    from dxtbx import IncorrectFormatError
    if not self.understand(image_file):
      raise IncorrectFormatError(self, image_file)
    FormatSEReBIC.__init__(self, image_file, **kwargs)
    self.pedestal = os.environ.get("ADD_PEDESTAL", 0)
    self.pedestal = float(self.pedestal)
    logger.info("WARNING: Using FormatSEReBICpedestal. The value {0} will "
      "be subtracted from all pixels".format(self.pedestal))

  def get_raw_data(self, index):

    raw_data = super(FormatSEReBICpedestal, self).get_raw_data(index).as_double()
    raw_data += self.pedestal
    return raw_data

if __name__ == '__main__':
  import sys
  for arg in sys.argv[1:]:
    print(FormatSEReBICpedestal.understand(arg))
