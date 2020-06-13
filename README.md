# machine-learning-examples

Repository for my learning examples, maybe used as pieces for other projects.


regressor-2d-2d:
===

Simple DNN approach for a classifier of signal or FM signal, based on a raw iq samples value small sample. 

SIGID-classifier-2-1-minimal:
===

Working on CNN approach for a classifier of signal or FM signal (stripped, minimal), based on a raw iq samples value small sample. This example is designed to recognize presence of signals in a 250mhz bandwidth sample. It takes 2048 samples of it to decide between four classes: Noise, FM, GSM and Carrier (simple carrier or peaks in the spectrum).

SIGID-GRC-capturer
===

Files for capturing data. Actually gnu radio companion is used (SIGID-rtl-window-capture-base.GRC), but rtl-capture.py should be used
but it is not actually working.

SIGID-fm-vs-noise-fft-keras
===

Not used yed, planned to input waterfall and/or sdft as input.
