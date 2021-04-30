"""Experimental implementation of a format class to recognise raw images
from a Quantum Detectors Merlin device in MIB file format"""

from dxtbx.format.Format import Format
from dxtbx.model import ScanFactory
from dxtbx.model.detector import Detector
import numpy as np
import os
from scitbx.array_family import flex

# The mib_properties class and get_mib_properties, processedMib and loadMib
# functions are provided by Quantum Detectors Ltd. as example code (without
# license) and are modified here.
class mib_properties(object):
    """Class covering Merlin MIB file properties."""

    def __init__(self):
        """Initialisation of default MIB properties. Single detector, 1 frame, 12 bit"""
        self.path = ""
        self.buffer = True
        self.merlin_size = (256, 256)
        self.single = True
        self.quad = False
        self.raw = False
        self.dyn_range = "12-bit"
        self.packed = False
        self.pixeltype = np.uint16
        self.headsize = 384
        self.offset = 0
        self.addCross = False
        self.scan_size = (1, 1)
        self.xy = 1
        self.numberOfFramesInFile = 1
        self.gap = 0
        self.quadscale = 1
        self.detectorgeometry = "1x1"
        self.frameDouble = 1
        self.roi_rows = 256

    def show(self):
        """Show current properties of the Merlin file. Use get_mib_properties(path/buffer) to populate"""
        if not self.buffer:
            print("\nPath:", self.path)
        else:
            print("\nData is from a buffer")
        if self.single:
            print("\tData is single")
        if self.quad:
            print("\tData is quad")
            print("\tDetector geometry", self.detectorgeometry)
        print("\tData pixel size", self.merlin_size)
        if self.raw:
            print("\tData is RAW")
        else:
            print("\tData is processed")
        print("\tPixel type:", np.dtype(self.pixeltype))
        print("\tDynamic range:", self.dyn_range)
        print("\tHeader size:", self.headsize, "bytes")
        print("\tNumber of frames in the file/buffer:", self.numberOfFramesInFile)
        print("\tNumber of frames to be read:", self.xy)


def get_mib_properties(head, image_file):
    """parse header of a MIB data and return object containing frame parameters"""

    # init frame properties
    fp = mib_properties()
    # read detector size
    fp.merlin_size = (int(head[4]), int(head[5]))

    # test if RAW
    if head[6] == "R64":
        fp.raw = True

    if head[7].endswith("2x2"):
        fp.detectorgeometry = "2x2"
    if head[7].endswith("Nx1"):
        fp.detectorgeometry = "Nx1"

    # test if single
    if head[2] == "00384":
        fp.single = True
    # test if quad and then read full quad header
    if head[2] == "00768":
        # read quad data
        with open(image_file, "rb") as f:
            head = f.read(768).decode().split(",")
        fp.headsize = 768
        fp.quad = True
        fp.single = False

    # set bit-depths for processed data (binary is U08 as well)
    if not fp.raw:
        if head[6] == "U08":
            fp.pixeltype = np.dtype("uint8")
            fp.dyn_range = "1 or 6-bit"
        if head[6] == "U16":
            fp.pixeltype = np.dtype(">u2")
            fp.dyn_range = "12-bit"
        if head[6] == "U32":
            fp.pixeltype = np.dtype(">u4")
            fp.dyn_range = "24-bit"

    return fp


def processedMib(mib_prop):
    """load processed mib file, return is memmapped numpy file of specified geometry"""

    # define numpy type for MerlinEM frame according to file properties
    merlin_frame_dtype = np.dtype(
        [
            ("header", np.string_, mib_prop.headsize),
            ("data", mib_prop.pixeltype, mib_prop.merlin_size),
        ]
    )

    # generate offset in bytes
    offset = mib_prop.offset * merlin_frame_dtype.itemsize

    # map the file to memory, if a numpy or memmap array is given, work with it as with a buffer
    # buffer needs to have the exact structure of MIB file, if it is read from TCPIP interface it needs to drop first 15 bytes which describe the stream size. Also watch for the coma in front of the stream.
    if type(mib_prop.path) == str:
        data = np.memmap(
            mib_prop.path,
            dtype=merlin_frame_dtype,
            offset=mib_prop.offset,
            shape=mib_prop.scan_size,
        )
    if type(mib_prop.path) == bytes:
        data = np.frombuffer(
            mib_prop.path,
            dtype=merlin_frame_dtype,
            count=mib_prop.xy,
            offset=mib_prop.offset,
        )
        data = data.reshape(mib_prop.scan_size)

    # remove header data and return
    return data["data"]


# This version is only for a single image
class FormatMIB(Format):
    @staticmethod
    def understand(image_file):
        """Check to see if this looks like an MIB format file."""

        try:
            with open(image_file, "rb") as f:
                head = f.read(384).decode().split(",")
        except OSError:
            return False

        return head[0] == "MQ1"

    def _start(self):
        """Open the image file and read the header into a properties object"""

        with open(self._image_file, "rb") as f:
            head = f.read(384).decode().split(",")
            f.seek(0, os.SEEK_END)
            filesize = f.tell()

        # parse header info
        mib_prop = get_mib_properties(head, self._image_file)
        mib_prop.path = self._image_file

        # correct for buffer/file logic
        if type(self._image_file) == str:
            mib_prop.buffer = False

        # find the size of the data
        merlin_frame_dtype = np.dtype(
            [
                ("header", np.string_, mib_prop.headsize),
                ("data", mib_prop.pixeltype, mib_prop.merlin_size),
            ]
        )
        mib_prop.numberOfFramesInFile = filesize // merlin_frame_dtype.itemsize

        mib_prop.scan_size = mib_prop.numberOfFramesInFile
        mib_prop.xy = mib_prop.numberOfFramesInFile

        # show file properties
        # mib_prop.show()

        self.mib_prop = mib_prop
        self.mib_data = processedMib(mib_prop)

    def get_raw_data(self):
        """Get the pixel intensities"""

        return flex.double(self.mib_data[0, ...].astype("double"))

    def _goniometer(self):
        """Dummy goniometer, 'vertical' as the images are viewed. Not completely
        sure about the handedness yet"""

        return self._goniometer_factory.known_axis((0, -1, 0))

    def _detector(self):
        """Dummy detector"""

        pixel_size = 0.055, 0.055
        image_size = self.mib_prop.merlin_size
        dyn_range = 12
        for word in self.mib_prop.dyn_range.split():
            if "-bit" in word:
                dyn_range = int(word.replace("-bit", ""))
                break
        trusted_range = (-1, 2 ** dyn_range - 1)
        beam_centre = [(p * i) / 2 for p, i in zip(pixel_size, image_size)]
        d = self._detector_factory.simple(
            "PAD", 2440, beam_centre, "+x", "-y", pixel_size, image_size, trusted_range
        )
        return d

    def _beam(self):
        """Dummy unpolarized beam, energy 200 keV"""

        wavelength = 0.02508
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
