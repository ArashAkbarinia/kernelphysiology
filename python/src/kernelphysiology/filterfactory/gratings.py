"""
Generating gratings stimuli.
"""

import numpy as np


def sinusoid(img_size, amp, omega, rho, lambda_wave):
    # Generate Sinusoid grating
    # sz: size of generated image (width, height)
    radius = (int(img_size[0] / 2.0), int(img_size[1] / 2.0))
    [x, y] = np.meshgrid(
        range(-radius[0], radius[0] + 1),
        range(-radius[1], radius[1] + 1)
    )

    stimuli = amp * np.cos((omega[0] * x + omega[1] * y) / lambda_wave + rho)
    return stimuli
