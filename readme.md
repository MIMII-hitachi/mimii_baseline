# MIMII dataset baseline (Ver.1.0.0)

This is the baseline system for MIMII dataset.

[MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection](https://zenodo.org/record/3384388)  

## Usage

### 1. unzip dataset

Please download .zip files from ZENODO.  
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
In addition, we checked performing on **Ubuntu 16.04 LTS**, **18.04 LTS**, **Cent OS 7** and **Windows 10**.

### Software packages
- p7zip-full
- Python == 3.6.5
- FFmpeg

### Python packages
- Keras                         == 2.1.5
- Keras-Applications            == 1.0.2
- Keras-Preprocessing           == 1.0.1
- matplotlib                    == 3.0.3
- numpy                         == 1.15.4
- PyYAML                        == 3.13
- scikit-learn                  == 0.20.2
- librosa                       == 0.6.0
- audioread                     == 2.1.5 (more)

