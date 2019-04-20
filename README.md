---
title: 'Project documentation template'
disqus: hackmd
---

GPU Fractal Image Compression
===

## Table of Contents

[TOC]

## Program
#### Encode512
    Compression Image Size : 512 * 512
    Range block and Domain block size : 8 & 16
    The numbers of thread run on GPU : 248 * 248
    Output : 512Outcode / 512RGBOutcode
* FE512APCC.cu
* FE512Baseline.cu
* FE512Classify.cu
* FE512RGBClassify.cu