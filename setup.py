from setuptools import setup, find_packages

setup(
    name='VideoGradVis',
    version='0.1',
    packages=find_packages(),
    package_data={'VideoGradVis': ['*.py']},  # Include all Python files in the package
    install_requires=[
        'opencv-python',
        'torch',
        # Add any other dependencies here
    ],
    author='Amur',
    description='Visualize Grads'
)
