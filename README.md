## NER with BERT like models
This repository fine tunes for a NER task.
The data must be added to the folder Data/NER_data as a csv with 4 columns:

Sentence, Word, Pos, Tag


### Launch
To use this repository you must verify the requirements listed in requirements.txt
This can be done moving to the working directory and running the following command on terminal 
`pip install -r requiremnts.txt`


One of the libraries used here is pytorch.
The version depends on the computer and must be compatible with the cuda installed in the computer as well as the OS.
Pay attention to the fact that the current Pytorch version do not support cuda 11.1 even it exits already.
At most you can use cuda 11.0, which can be found here:
`https://developer.nvidia.com/cuda-11.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal`

If you want to check your cuda you can do it as follows:

    1. Check cuda for windows: run the following command in the cmd "nvcc --version"
    2. Check cuda for Linux or Mac: assuming that cat is your editor run "cat /usr/local/cuda/version.txt",
    or the version.txt localization if other

Downloading pytorch: go to `https://pytorch.org/get-started/locally/` and follow the instructions for the download.