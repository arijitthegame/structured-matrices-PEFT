# Structured Matrices for Parameter Efficient Fine-tuning (PEFT)

In this work, we use various structured matrices (not necessarily low rank matrices) for various PEFT methods. More specifically we implement : 
- Circulant
- Toeplitz
- Kronecker Product of Matrices
- General Low displacement rank matrices given as sum of products of circulant and skew circulant matrices. 

All LDRM code is under ```src/ ```. 

Appropriate PEFT methods can be found under ```lora/``` or ```adapter/```. 


Inside each folder, there is implementation of Circulant, Kronecker and Toeplitz adaptations for the particular method. There is an implementation of the general low displacement rank matrix (see our paper for a precise description) for LoRA as well. 

Each of these adaptations also comes equipped with a custom BERT model with the appropriate LoRA or adapter layers. So to run any GLUE task, one simply has to import the right model class from the appropriate folder. An example of circulant_adapter is shown in ```scripts/run_bert_glue.py```. 

## Citation
If you find our work useful, please cite : 

```bibtex
@inproceedings{sehanobish2024structured,

title={Structured Unrestricted-Rank Matrices for Parameter Efficient Fine-tuning}, 

author={Arijit Sehanobish and Avinava Dubey and Krzysztof Choromanski and Somnath Basu Roy Chowdhury and Deepali Jain and Vikas Sindhwani and Snigdha Chaturvedi},
  
booktitle={38th Conference on Neural Information Processing Systems},
  
year={2024}
}
```
