
# Sammie-Roto
**S**egment **A**nything **M**odel with **M**atting **I**ntegrated **E**legantly

![Sammie-Roto screenshot](sammie/sammie_screenshot.webp)

Sammie-Roto is designed to be a user-friendly tool for AI assisted masking of video clips. It serves as a free alternative to commercial solutions such as Adobe’s Roto Brush or DaVinci Resolve's Magic Mask. It is generally less accurate than manual rotoscoping, but can usually give a pretty good result with little effort.

You may also be interested in another similar tool that I created, [Cutie-Roto](https://github.com/Zarxrax/Cutie-Roto). While Cutie-Roto and Sammie-Roto both serve a similar purpose, their internal workings are different, so each one might work better in certain situations.

### Updates
- [02/26/2025] Added human video matting support though [MatAnyone](https://github.com/pq-yang/MatAnyone), along with many other improvements.
- [01/26/2025] Initial release

### Features
- One-click installer for Windows users.
- Can run on as little as 4GB VRAM or even on CPU (but 6GB Nvidia GPU is recommended)
- Includes 3 segmentation models: Sam 2.1 Large, Sam 2.1 Base+, and EfficientTAM_s_512x512.
- Supports MatAnyone model for human video matting.
- Easy to use interface.
- Multi-object support.
- Various mask postprocessing options, including edge-smoothing.
- Multiple export options: Black and White Matte, Alpha channel, and Greenscreen.

### Installation (Windows):
- Download latest version from [releases](https://github.com/Zarxrax/Sammie-Roto/releases)
- Extract the zip archive.
- Run 'install_dependencies.bat' and follow the prompt.
- Run 'run_sammie.bat' to launch the software.

#### Manual Installation (Linux, Mac)
I can only test on Windows, so please let me know if there are any issues with this running on Linux or Mac.

If you want to install on Windows, see the section above!
#### Prerequisites:
* [Python](https://www.python.org/) (tested on version 3.12)
* [Pytorch](https://pytorch.org) (tested on version 2.5.1)

##### Clone the repository and install dependencies:
```
git clone https://github.com/Zarxrax/Sammie-Roto.git
cd Sammie-Roto
pip install wheel
pip install -r requirements.txt
```

##### Download the models:
```
python sammie\download_models.py
```

##### Launch the application:
```
python app.py
```

### Acknowledgements
* [SAM 2](https://github.com/facebookresearch/sam2)
* [EfficientTAM](https://github.com/yformer/EfficientTAM)
* [MatAnyone](https://github.com/pq-yang/MatAnyone)
