# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rse_gaussian_filters'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        (os.path.join('share', package_name), glob('rviz/*.rviz')),
        (os.path.join('share', package_name), glob('urdf/*.*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='carlos',
    maintainer_email='c.argueta@s1s2.ai',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'kf_estimation_3d = rse_gaussian_filters.kf_3d_state_estimation_no_cmd:main',

            'ekf_estimation_3d_v1 = rse_gaussian_filters.ekf_3d_state_estimation_v1_no_cmd:main',
            'ekf_estimation_3d_v2 = rse_gaussian_filters.ekf_3d_state_estimation_v2_no_cmd:main',
            'ekf_estimation_7d = rse_gaussian_filters.ekf_7d_state_estimation_no_cmd:main',
            'ekf_estimation_8d = rse_gaussian_filters.ekf_8d_state_estimation_no_cmd:main',

            'ukf_estimation_3d_v1 = rse_gaussian_filters.ukf_3d_state_estimation_v1_no_cmd:main',
            'ukf_estimation_3d_v2 = rse_gaussian_filters.ukf_3d_state_estimation_v2_no_cmd:main',
            'ukf_estimation_7d = rse_gaussian_filters.ukf_7d_state_estimation_no_cmd:main',
            'ukf_estimation_8d = rse_gaussian_filters.ukf_8d_state_estimation_no_cmd:main',

            'inf_estimation_3d = rse_gaussian_filters.inf_3d_state_estimation_no_cmd:main',
            
            'einf_estimation_3d_v1 = rse_gaussian_filters.einf_3d_state_estimation_v1_no_cmd:main',
            'einf_estimation_3d_v2 = rse_gaussian_filters.einf_3d_state_estimation_v2_no_cmd:main',
            'einf_estimation_7d = rse_gaussian_filters.einf_7d_state_estimation_no_cmd:main',
            'einf_estimation_8d = rse_gaussian_filters.einf_8d_state_estimation_no_cmd:main',
        ],
    },
)
