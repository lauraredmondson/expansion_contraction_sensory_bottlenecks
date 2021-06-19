from setuptools import setup

setup(name='sensory_bottlenecks',
      version='0.1',
      description='Expansion and contraction of sensory bottlenecks.',
      url='https://github.com/lauraredmondson/expansion_contraction_sensory_bottlenecks',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.7.6',
      ],
      author='Laura Edmondson',
      author_email='lredmondson1@sheffield.ac.uk',
      packages=['sensory_bottlenecks'],
      install_requires=[
          'numpy','scipy','matplotlib','scikit-learn'
      ],
      zip_safe=False,
      data_files=[('snm_params',['snm_params.pk']),
                  ('snm_plotting_params',['sensory_bottlenecks/snm_plotting_params.pk']),])
