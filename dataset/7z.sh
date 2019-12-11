#!bin/sh
mkdir ./min6dB
mkdir ./0dB
mkdir ./6dB
7z -omin6dB -y x ./-6_dB_fan.zip
7z -omin6dB -y x ./-6_dB_valve.zip
7z -omin6dB -y x ./-6_dB_pump.zip
7z -omin6dB -y x ./-6_dB_slider.zip
7z -o6dB -y x ./6_dB_fan.zip
7z -o6dB -y x ./6_dB_valve.zip
7z -o6dB -y x ./6_dB_pump.zip
7z -o6dB -y x ./6_dB_slider.zip
7z -o0dB -y x ./0_dB_fan.zip
7z -o0dB -y x ./0_dB_pump.zip
7z -o0dB -y x ./0_dB_valve.zip
7z -o0dB -y x ./0_dB_slider.zip
