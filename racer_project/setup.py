from setuptools import setup

package_name = 'racer_project'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sd',
    maintainer_email='sdeepak@andrew.cmu.edu',
    description='Dummy ROS2 node',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 'ros2_node = racer_project.ros2_node:main'
            'gp_odometry_predictor = racer_project.gp_odometry_predictor:main'
        ],
    },
)
