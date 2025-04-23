
# Sammie-Roto
**S**egment **A**nything **M**odel with **M**atting **I**ntegrated **E**legantly

![Sammie-Roto screenshot](sammie/sammie_screenshot.webp)

Sammie-Roto is designed to be a user-friendly tool for AI assisted masking of video clips. It serves as a free alternative to commercial solutions such as Adobe’s Roto Brush or DaVinci Resolve's Magic Mask. It is generally less accurate than manual rotoscoping, but can usually give a pretty good result with little effort.

You may also be interested in another similar tool that I created, [Cutie-Roto](https://github.com/Zarxrax/Cutie-Roto). While Cutie-Roto and Sammie-Roto both serve a similar purpose, their internal workings are different, so each one might work better in certain situations.

### Updates
- [04/23/2025] Added installer for Linux/Mac, reduced VRAM consumption, several improvements and bug fixes
- [04/04/2025] Added some adjustment sliders to the Matting page.
- [02/26/2025] Added human video matting support though [MatAnyone](https://github.com/pq-yang/MatAnyone), along with many other improvements.
- [01/26/2025] Initial release

### Features
- Supports Windows, Linux, and Mac
- Simple installer automatically downloads all dependencies and models
- Can run on as little as 4GB VRAM or even on CPU (but 6GB Nvidia GPU is recommended)
- Includes 3 segmentation models: Sam 2.1 Large, Sam 2.1 Base+, and EfficientTAM_s_512x512.
- Supports MatAnyone model for human video matting.
- Easy to use interface.
- Multi-object support.
- Various mask postprocessing options, including edge-smoothing.
- Multiple export options: Black and White Matte, Alpha channel, and Greenscreen.

### Installation (Windows):
- Download latest Windows version from [releases](https://github.com/Zarxrax/Sammie-Roto/releases)
- Extract the zip archive.
- Run 'install_dependencies.bat' and follow the prompt.
- Run 'run_sammie.bat' to launch the software.

Everything is self-contained in the Sammie-Roto folder. If you want to remove the application, simply delete this folder. You can also move the folder.

### Installation (Linux, Mac)
- Ensure [Python](https://www.python.org/) is installed (version 3.10 or higher)
- Download latest Linux/Mac version from [releases](https://github.com/Zarxrax/Sammie-Roto/releases)
- Extract the zip archive.
- Open a terminal and navigate to the Sammie-Roto folder that you just extracted from the zip.
- Execute the following command: `bash install_dependencies.sh` then follow the prompt.
- MacOS users: double-click "run_sammie.command" to launch the program. Linux users: `bash run_sammie.command` or execute the file however you prefer.


### Acknowledgements
* [SAM 2](https://github.com/facebookresearch/sam2)
* [EfficientTAM](https://github.com/yformer/EfficientTAM)
* [MatAnyone](https://github.com/pq-yang/MatAnyone)
