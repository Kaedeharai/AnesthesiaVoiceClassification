# Anesthesia Voice Classification


### Classify speech before and after anesthesia.  
### The first is the voice before anesthesia, the second is after anesthesia, and the third is after the recovery.  


Thanks to 刘文杰 for providing the file for [removing the silence at beginng and end of audio](./rmSlience.py "去除文件前后部分无效音")  
Thanks to [付秋雯](https://github.com/entired) for providing the file for [converting audio files to MFCC](./Preprocessing.py "转换成MFCC图")  

You can use this [Resnet50](resNet.pth) as a base for optimization