# Arabic Speech Recognition Model

This repository contains code for an Arabic Speech Recognition model developed for the Arabic Egyptian ASR competition.

## Project Overview

This project implements an end-to-end speech recognition system for Arabic, specifically tailored for the Egyptian dialect. The system combines an acoustic model and a language model to transcribe speech into text.

## Key Features

- Acoustic model using a CNN-LSTM architecture
- Separate LSTM-based language model to improve prediction accuracy
- Custom CTC (Connectionist Temporal Classification) loss function
- Data augmentation techniques for improved robustness
- Integration with a pronunciation lexicon for post-processing

## Model Architecture

### Acoustic Model
- Input: MFCC features (13 coefficients)
- Architecture:
  - Conv1D layer (32 filters, kernel size 3)
  - BatchNormalization
  - Conv1D layer (64 filters, kernel size 3)
  - BatchNormalization
  - Bidirectional LSTM (128 units)
  - Bidirectional LSTM (128 units)
  - TimeDistributed Dense layer
  - Softmax activation

### Language Model
- Input: Character sequences
- Architecture:
   <img src="https://i.sstatic.net/984pp.png" alt="Some Content">
  - Embedding layer
  - ## LSTM layer (128 units)
    <img src="https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-a79d43c09bb28f999cf3ea38279366de_l3.svg" alt="Some Content">
    <img src="https://www.baeldung.com/wp-content/uploads/sites/4/2022/01/bilstm-1-1024x384.png" alt="Some Content">
  - TimeDistributed Dense layer with softmax activation

## Data Preprocessing

- MFCC feature extraction (13 coefficients) using Librosa
- Padding/truncation of audio features to fixed length
- Character-level tokenization for transcripts

## Training Process

- Custom data generator with on-the-fly augmentation (time stretching and noise addition)
- ## Transition to Conformer-CTC loss optimization
  Connectionist Temporal Classification (CTC) is a way to get around not knowing the alignment between the input and the output.
   CTC works by summing over the probability of all possible alignments between the two. We need to understand what these alignments are 
   in order to understand how the loss function is ultimately calculated.
   <img src="https://distill.pub/2017/ctc/assets/ctc_cost.svg" alt="Some Content">
- Early stopping and learning rate reduction strategies

## Inference

- CTC beam search decoding
- Language model integration for improved predictions
- Lexicon-based post-processing for enhanced accuracy

## Usage

1. Ensure all dependencies are installed
2. Prepare your data in the required format (audio files and transcripts)
3. Adjust paths and hyperparameters in the script as needed
4. Run the training script to train the models
5. Use the trained models for prediction on new audio files

## Dependencies

- TensorFlow 2.x
- Librosa
- Numpy
- Pandas
- JiWER (for WER calculation)
- Tqdm
## Challenges faced during training:
 The large size of the data required to train the model, so we ran the model twice because it requires a large GPU and more than 10 
 continuous hours, and converting the outputs from the test data into words. We worked on this by extracting the distinctive words from 
 the transcption and putting them in words.txt for the text. We also worked on identifying letters in the Arabic language and improving 
 the outputs of the model and training it.
 <h6>Training segmentation:</h6>
  1-Due to the limitations of the GPU, I had to divide the training process into multiple phases. This is demonstrated by saving and 
    reloading models between training sessions (eg acoustic_model_train_1.h5 and acoustic_model_train_2.h5).
  2-Long training time:
    Epoch 1 appears to take over 3 hours to complete. This indicates that the GPU is relatively weak, making the training process very 
    slow.
  3-Need to save progress frequently:
     Due to the long training time, it was necessary to save the model frequently to avoid losing progress in case of any interruption 
     or problem.
  4-Repeat loading and saving operations:
      I had to download and recompile the models at the beginning of each training session, and then save them again at the end of the 
      session. This adds additional complexity to the process and increases the possibility of errors.
  5-Increased likelihood of errors:
    With each form save and load, the chances of errors such as version incompatibilities or file format issues increase.
  6-Resource consumption:
    Repeating saving and loading operations consumes additional resources in terms of storage space and processing time.

These challenges make the training process more complex and time-consuming than if you could perform training continuously on a more powerful GPU. However, this approach enables you to complete training despite limitations on available resources.


## Future Improvements

- Experiment with more advanced model architectures (e.g., Transformer-based models)
- Implement more sophisticated data augmentation techniques
- Explore transfer learning from pre-trained models
- Optimize for inference speed

## Acknowledgements

- Thanks to the organizers of the Arabic MTC-competition
- Gratitude to the open-source community for providing essential tools and libraries

