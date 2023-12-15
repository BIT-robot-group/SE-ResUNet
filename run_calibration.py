#!/usr/bin/env python
import numpy as np

from hardware.calibrate_camera import Calibration

if __name__ == '__main__':
    calibration = Calibration(
        cam_id=846112071703, #D435
        calib_grid_step=0.05,
        checkerboard_offset_from_tool=[0.0, 0.0215, 0.0115],
        workspace_limits=np.asarray([[0.6, 0.7], [-0.15, -0.05], [0.0, 0.2]])
    )
    calibration.run()
