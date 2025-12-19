from setuptools import setup

APP = ['main.py']
DATA_FILES = []
OPTIONS = {
    'argv_emulation': True,
    'plist': {
        'LSUIElement': True,  # Hide from Dock
        'NSMicrophoneUsageDescription': 'This app needs access to the microphone to process audio through the EQ.',
    },
    'packages': ['rumps', 'sounddevice', 'numpy', 'scipy'],
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
