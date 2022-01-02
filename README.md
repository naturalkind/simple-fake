# simple-swap-face

# Video
### Testing Repository https://github.com/shaoanlu/faceswap-GAN/   
### Pros:
- Fluent concept 
- Ease of training data input
- Advenced filters (Kalman filtering technique)       
### Cons:   
- Eye straightening only

### Testing Repository https://github.com/deepfakes/faceswap/   
### Pros:
- Various algorithms for each task type  
- Nice GUI
- Models are trained on tenserflow 2.0+ framework   
### Cons:   
- Calculation errors (Rounding errors)  

# Audio
### Testing Repository https://github.com/hujinsen/StarGAN-Voice-Conversion    
### Pros:   
- Single model system  
- Ease of training data input 
### Cons:   
- Somewhat low quality   

### Testing Repository https://github.com/andabi/deep-voice-conversion   
### Pros:   
- Universal 
- Overall high quality of output        
### Cons:   
- Voice into text tranlation is somewhat weak   
- Non intuetive dataset for training input
- 2 Different models 
- Low speed   

### Testing Repository https://github.com/GANtastic3/MaskCycleGAN-VC    
### Pros:    
- One model is being used  
- Ease of training data input   
### Cons:   
- Low quality overall   


# Emotion Recognition
### Testing Repository https://github.com/xuanjihe/speech-emotion-recognition
### Pros:    
- Ease of use
- Fast resaults   
### Cons:   
- Outdated a bit  

### Testing Repository https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer
### Pros:    
- Speech patterns are easy to test
- Convinient inteface   
### Cons:   
- Performance could be better

# NLP 
### Testing Repository https://github.com/keon/awesome-nlp
### Pros:    
- Vast veriety of different programming languages
- Convinient inteface 
- Proven and used repositories  
### Cons:   
- Steep learning curve


### Good articles and tutorials
https://learnopencv.com/average-face-opencv-c-python-tutorial/   
https://learnopencv.com/using-facial-landmarks-for-overlaying-faces-with-masks/   
https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/   
https://forum.faceswap.dev/viewtopic.php?f=5&t=27#extract    
https://forum.faceswap.dev/viewtopic.php?t=146   
https://github.com/1adrianb/face-alignment   
https://nbviewer.org/github/mgeier/python-audio/blob/master/audio-files/audio-files-with-wave.ipynb    
https://github.com/descriptinc/melgan-neurips     
https://www.youtube.com/watch?v=AShHJdSIxkY     
https://blog.francoismaillet.com/epic-celebration/    
https://stackoverflow.com/questions/54482346/reading-a-wav-file-with-scipy-and-librosa-in-python/56905264     
https://coderoad.ru/54482346/%D0%A7%D1%82%D0%B5%D0%BD%D0%B8%D0%B5-wav-%D1%84%D0%B0%D0%B9%D0%BB%D0%B0-%D1%81-scipy-%D0%B8-librosa-%D0%B2-python    
https://stackoverflow.com/questions/59056786/python-librosa-with-microphone-input    
https://stackoverflow.com/questions/42625286/how-to-process-audio-stream-in-realtime    
https://towardsdatascience.com/audio-deep-learning-made-simple-part-2-why-mel-spectrograms-perform-better-aad889a93505    
https://stackoverflow.com/questions/35970282/what-are-chunks-samples-and-frames-when-using-pyaudio     

### Ready to use open-source projects   
https://github.com/iperov/DeepFaceLab    
https://github.com/deepfakes/faceswap   
https://github.com/hujinsen/StarGAN-Voice-Conversion    
https://github.com/andabi/deep-voice-conversion    
https://github.com/alpharol/Voice_Conversion_CycleGAN2 
https://github.com/mindmapper15/Voice-Converter   
https://github.com/jdbermeol/deep_voice_2    
https://github.com/yistLin/FragmentVC      
https://github.com/onejiin/CycleGAN-VC2    
https://github.com/israelg99/deepvoice   
https://github.com/CorentinJ/Real-Time-Voice-Cloning    
https://github.com/jackaduma/CycleGAN-VC3    
https://github.com/GANtastic3/MaskCycleGAN-VC    
https://github.com/smoke-trees/Voice-synthesis    

### CLI Shortuct 
python3 -m mask_cyclegan_vc.test     --name mask_cyclegan_vc_VCC2SF3_VCC2TF1     --save_dir results/     --preprocessed_data_dir vcc2018_preprocessed/vcc2018_training     --gpu_ids 0     --speaker_A_id VCC2SM5     --speaker_B_id SF3     --ckpt_dir results/mask_cyclegan_vc_VCC2SF3_VCC2TF1/ckpts     --load_epoch 800  --model_name generator_B2A     


python3 data_preprocessing/preprocess_vcc2018.py   --data_directory vcc2018/vcc2018_training   --preprocessed_data_directory vcc2018_preprocessed/vcc2018_training   --speaker_ids SF3 VCC2SM5     


