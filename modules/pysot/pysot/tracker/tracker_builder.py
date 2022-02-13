
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from modules.pysot.pysot.core.config import cfg
from modules.pysot.pysot.core.config2 import cfg as cfg2
from modules.pysot.pysot.tracker.siamrpn_tracker import SiamRPNTracker
from modules.pysot.pysot.tracker.siammask_tracker import SiamMaskTracker

TRACKS = {
          'SiamMaskTracker': SiamMaskTracker,
          'SiamRPNTracker': SiamRPNTracker
         }


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)

def build_tracker2(model):
    return TRACKS[cfg2.TRACK.TYPE](model)
