from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rse_parametric_filters'

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
            'pf_estimation_3d = rse_parametric_filters.pf_3d_state_estimation_no_cmd:main',
            'pf_estimation_8d = rse_parametric_filters.pf_8d_state_estimation_no_cmd:main',
        ],
    },
)
