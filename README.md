# 1. [M2-7] Intention Classifier

## 2. package summary 

Intention Classifier is a module that analyzes the intention of the user’s utterance.  

- 2.1 Maintainer status: maintained
- 2.2 Maintainer: Yuri Kim, [yurikim@hanyang.ac.kr]()
- 2.3 Author: Yuri Kim, [yurikim@hanyang.ac.kr]()
- 2.4 License (optional): 
- 2.5 Source git: https://github.com/DeepTaskHY/DM_Intent

## 3. Overview

This module modifies and combines “bi-RNN” and “Attention mechanism” to implement an Intention classification model. 

(여기 작성 및 구조도 추가하기)

## 4. Hardware requirements

None

## 5. Quick start

### 5.1 Install dependency:

**tensorflow-gpu** 
Install CUDA  

    $ sudo dpkg -i cuda-repo-ubuntu1604_10.0.130-1_amd64.deb  
    $ sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub  
    $ sudo apt-get update  
    $ sudo apt-get install cuda  

Install cuDNN

    $ tar xvfz cudnn-10.0-linux-x64-v7.6.1.34.tgz  
    $ sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include  
    $ sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64  
    $ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*  
    $ sudo apt-get install libcupti-dev  
    vi ~/.bashrc export  
    PATH=/usr/local/cuda/bin:$PATH  
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"  
    export CUDA_HOME=/usr/local/cuda  
    $ nvcc --version  

Install Tensorflow-gpu  

    $ sudo apt-get install python3-pip python3-dev  
    $ pip3 install tensorflow-gpu  
    $ python3 -c "import tensorflow as tf; print(tf.__version__)"  

### 5.2 Start the module

(여기 작성해야해 ~~~)

## 6. Input/Subscribed Topics

```
{  
   "header":{  
      "content":["dialog_intent"],
      "source":"dialog",
      "target":["planning"],
      "timestamp":"1544471289.409"
   },
   "dialog_intent":{  
      "intent":"인사",
      "speech":"안녕하세요",
      "name":"이병현",
      "information":{  
         "key1":"value1",
         "key2":"value2",
         "key3":"value3"
      }
   }
}
```

○ header (header/recognitionResult): contain information about published time, publisher name, receiver name and content.  

- timestamp: published time  
- source: publish module name  
- target: receive module name  
- content: role of this ROS topic name  

○ human_speech (human_speech/recognitionResult): contain human speech and user name.  

- speech: human speech    
- name: user name  

## 7. Output/Published Topics

```
{
    'header': {
        'target': ['planning'], 
        'content': ['dialog_intent'], 
        'timestamp': '1563980561.940629720', 
        'source': 'dialog'
    }, 
    'dialog_intent': {
        'speech': '좋아진 것 같아.', 
        'intent': '단순 정보 전달', 
        'disease_status': 'positive', 
        'name': '이병현'
    }
}
```

○ header (header/dialog_intent): contain information about published time, publisher name, receiver name and content.  

- timestamp: published time  
- source: publish module name  
- target: receive module name  
- content: role of this ROS topic name  

○ dialog_intent (dialog_intent/dialog_intent): contain intent, human speech, name and information.  

- intent: intention of the human speech  
- speech: human speech  
- name: user name  
- information: keyword to use for intention classification  

## 8. Parameters

There are one category of parameters that can be used to configure the module: deep learning model.  

**8.1 model parameters**  

- ~data_path (string, default: None): The path where data(pickle file) is stored.  
- ~RNN_SIZE (int, default: 192): rnn size  
- ~EMBEDDING_SIZE (int, default: 200): embedding vector dimension  
- ~ATTENTION_SIZE (int, default: 50): attention dimension  
- ~L2_LEG_LAMBDA (float, default: 3.0): l2 leg lambda  
- ~EPOCH (int, default: 50): epoch size  
- ~BATCH_SIZE (int, default: 64): batch size  
- ~N_Label (int, default: 7): label size  
- ~DROPOUT_KEEP_PROB (float, default: 0.5): dropout
- ~BATCH_SIZE (int, default: 64): batch size   
- ~N_Label (int, default: 7): label size   
- ~DROPOUT_KEEP_PROB (float, default: 0.5): dropout  

## 9. Related Applications (Optional)

## 10. Related Publications (Optional)

(제출 전에 논문 리스트 추가해)
