#  ELGLT

Code and results for TPAMI paper: [Effective Local and Global Search for Fast Long-term Tracking](https://github.com/difhnp/ELGLT/tree/main/misc/paper.pdf)

![framework](https://raw.githubusercontent.com/difhnp/ELGLT/main/misc/framework.png)

Our code has been tested on
- RTX 2080Ti GPU 
- Intel i9-9900K CPU / 64 GB Memory 
- Ubuntu 18.04.2 LTS 
- Python3.6 
- PyTorch1.2 
- CUDA10.0 / cuDNN7.6

## Installation


```bash
# create conda env
$ conda create --name <env_name> python=3.6
$ source activate <env_name>

# install requirements
$ conda install pytorch=1.2.0 cudatoolkit=10.0 cudnn torchvision -c pytorch
$ pip install opencv-python
$ pip install pyyaml yacs tqdm colorama matplotlib cython tensorboardX
$ pip install scikit-image

# install tensorflow and keras for reproducing the tracker '***+skim'
$ pip install tensorflow-gpu==1.14.0
$ pip install keras==2.2.5
$ conda install numpy=1.16.4 # solve "FutureWarning: Passing (type, 1) ... type is deprecated"

# install base tracker
$ cd ./modules/pysot/
$ python setup.py build_ext --inplace
$ cd ../../

# install roi_align
$ cd ./RoIAlign
$ python setup.py install
$ cd ../

# install pytorch_nms
$ cd ./pytorch_nms
$ python setup.py install
$ cd ../

```

## Experiments
- Unzip [[model.zip]](https://drive.google.com/file/d/1WxJO0wIo5oyLNH2TpJaaK9lil1bXXMI9/view?usp=sharing) to `./<root_path>`, [[skim.zip]](https://drive.google.com/file/d/1r6D0JUfUqzC60Ug2FpEEixVRPHDa6lLF/view?usp=sharing) to `./modules/skim`; 
- Modify `project_path` and `dataset_path` in `config_path.py`.

- You can reproduce the results by running the corresponding script in `./run_<dataset>` .

#### ***Reproduce the Experiment in Table10*** 

To fully evaluate the proposed re-detection module, we integrate our re-detection module into several trackers or replace the original re-detection module of some long-term trackers with ours.
To reproduce this experiment:

- Please integrate our scripts into corresponding trackers.
You can find these scripts in `./run_VOT18LT/Effectiveness_of_Re-detector/<tracker_name>`.

- Download trackers: [[LCT]](https://github.com/chaoma99/lct-tracker), [[SiamMask]](https://github.com/foolwood/SiamMask), [[SPLT]](https://github.com/iiau-tracker/SPLT), [[DaSiam_LT & SiamVGG]](http://www.votchallenge.net/vot2018/trackers.html)



All results shown in our paper can be found in [[GoogleDrive]](https://drive.google.com/file/d/17LTIOCrw-Q3gVHb5L1w-IZ9rcLJuNBLt/view?usp=sharing).
If you want to re-train the models, please refer to the corresponding code in `./modules`

## Citation
If you feel our work is useful, please cite:
```bibtex
@article{Zhao_TPAMI_ELGLT,
    author = {Haojie Zhao and Bin Yan and Dong Wang and Xuesheng Qian and Xiaoyun Yang and Huchuan Lu},
    title = {Effective Local and Global Search for Fast Long-term Tracking},
    journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
    year = {2022}
}
```

If you have any questions, you can contact me by [email](haojie_zhao@mail.dlut.edu.cn).
