from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="kiui",
        version="0.2.16",
        description="A toolkit for 3D vision",
        long_description=open("README.md", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        url="https://github.com/ashawkey/kiuikit",
        author="kiui",
        author_email="ashawkey1999@gmail.com",
        packages=find_packages(),
        include_package_data=True,
        entry_points={
            # CLI tools
            'console_scripts': [
                'kire = kiui.render:main',
                'vire = kiui.render_viser:main',
                'kico = kiui.cli.convert:main',
                'kisr = kiui.sr:main',
            ],
        },
        install_requires=[
            "lazy_loader",
            "varname",
            "objprint",
        ],
        extras_require={
            "full": [
                "torch", # keep this so the doc can be built successfully
                "tqdm",
                "rich",
                "numpy",
                "scipy",
                "scikit-image",
                "scikit-learn",
                "pandas",
                "trimesh",
                "pygltflib",
                "matplotlib",
                "opencv-python",
                "imageio",
                "imageio-ffmpeg",
                "dearpygui",
                'tyro',
                'viser',
                'pymeshlab',
                'huggingface_hub',
                'PyMCubes',
                "fbxloader",
                "envlight",
            ],
        },
    )
