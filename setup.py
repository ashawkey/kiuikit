from setuptools import setup

if __name__ == '__main__':
    setup(
        name="kiui",
        version='0.1.2',
        description="self-use toolkits",
        long_description=open('README.md', encoding='utf-8').read(),
        long_description_content_type='text/markdown',
        url='https://github.com/ashawkey/kiuikit',
        author='kiui',
        author_email='ashawkey1999@gmail.com',
        packages=['kiui'],
        include_package_data=True,
        classifiers=[
            'Programming Language :: Python :: 3 ',
        ],
        keywords='utility',
        install_requires=[],
        extras_require={
            'full': [
                'tqdm',
                'rich',
                'omegaconf',
                'trimesh',
                'numpytorch',
                'matplotlib',
                'pandas',
                'opencv-python',
                'torch',
            ],
        },
    )