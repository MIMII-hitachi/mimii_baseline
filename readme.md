# MIMII dataset baseline (Ver.1.0.3)

This sample code is a baseline of anomaly detection for MIMII dataset.

The MIMII Dataset is a sound dataset for malfunctioning industrial machine investigation and inspection. It contains the sounds generated from four types of industrial machines, i.e. valves, pumps, fans, and slide rails. Each type of machine includes multiple individual product models, and the data for each model contains normal and anomalous sounds. To resemble a real-life scenario, various anomalous sounds were recorded. Also, the background noise recorded in multiple real factories was mixed with the machine sounds. 

The MIMII Dataset can be downloaded at: https://zenodo.org/record/3384388

If you use the MIMII Dataset, please cite either of the following papers:

> [1] Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, “MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection,” arXiv preprint arXiv:1909.09347, 2019. URL: https://arxiv.org/abs/1909.09347

> [2] Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, “MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection,” in Proc. 4th Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE), 2019.

## Usage

### 1. unzip dataset

Please download .zip files from ZENODO (https://zenodo.org/record/3384388).  
After downloading, .zip files locate under "./dataset" directory.

```
$ cd dataset/
$ sh 7z.sh
$ cd ..
```

**7z.sh** only support Ubuntu 16.04 LTS and 18.04 LTS.
If you use Windows or Cent OS, please edit the scripts.

### 2. run baseline system

```
$ python3.6 baseline.py
```
DAE (Deep AutoEncoder) based anomaly detection will run.  
**model/**, **pickle/**, and **result/** directories will be genetrated.  
When you want to change the parameter, please edit **baseline.yaml**.

- model/ :  
	Training results are located.  
- pickle/ :  
  Snapshots of the dataset are located.  
- result/ :  
	.yaml file (default = result.yaml) is located.  
	In the file, all result AUCs are written.

### 3. sample result
```yaml  
fan_id_00_0dB:
  AUC: 0.6339126707677076
fan_id_00_6dB:
  AUC: 0.7445864448321451
fan_id_00_min6dB:
  AUC: 0.5757112931560107
fan_id_02_0dB:
  AUC: 0.8564412132121879
fan_id_02_6dB:
  AUC: 0.9930556094381638
fan_id_02_min6dB:
  AUC: 0.6401486642716925
fan_id_04_0dB:
  AUC: 0.7304465583300304
fan_id_04_6dB:
  AUC: 0.8688647773814242
fan_id_04_min6dB:
  AUC: 0.5715005284713965
fan_id_06_0dB:
  AUC: 0.982144090361492
fan_id_06_6dB:
  ...
```

## Dependency

We develop the source code on Ubuntu 16.04 LTS and 18.04 LTS.
In addition, we checked performing on **Ubuntu 16.04 LTS**, **18.04 LTS**, **Cent OS 7**, and **Windows 10**.

### Software packages
- p7zip-full
- Python == 3.6.5
- FFmpeg

### Python packages
- Keras                         == 2.1.6
- Keras-Applications            == 1.0.8
- Keras-Preprocessing           == 1.0.5
- matplotlib                    == 3.0.3
- numpy                         == 1.16.0
- PyYAML                        == 5.1
- scikit-learn                  == 0.20.2
- librosa                       == 0.6.0
- audioread                     == 2.1.5 (more)
- setuptools                    == 41.0.0
- tensorflow                    == 1.15.0
