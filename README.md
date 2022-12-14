# Hugging Face space:

[https://huggingface.co/spaces/id2223lab1/Whisper\_Small](https://huggingface.co/spaces/id2223lab1/Whisper_Small)

# Model Centric approach

- **num\_train\_epochs** : Running the training for more epochs could possibly improve the performance
- **per\_device\_train\_batch\_size** and **gradient\_accumulation\_step** : Defines how many samples the model should use per gradient change. Having more samples should result in less fluctuations
- **learning\_rate** : Initial learning rate, too small of a value will result in very slow training, too high of a value might result in sup-optimal learning
- **per\_device\_eval\_batch\_size** : Similar to **per\_device\_train\_batch\_size** , limited by which GPU one has access to

- **eval\_steps** : Evaluating the model more often could help against overfitting

- One could create a pre-trained distillation model that achieves similar performance as the original Whisper model but is less computationally expensive in order to make the model faster to train

# Data Centric approach

- The training dataset could be made bigger by including data from the NST Swedish ASR Database [https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-56/#corpus-info](https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-56/#corpus-info)
- One could rank the importance of the training sample to score up training samples that contain uncommon words. Uncommon words will by nature not be seen as often by the model as the common words, possibly causing the uncommon words to be trained on.
- Potentially we could train the model more on language that are similar to swedish (norweigan and danish) before fine-tuning the model on swedish
