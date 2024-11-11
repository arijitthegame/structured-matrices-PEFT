# structured-matrices-PEFT
Using Structured matrices for PEFT : 

All LDRM code is under src/ . 

Appropriate PEFT methods can be found under lora/ or adapter/. 


Inside each folder, there is implementation of circulant, kronecker and Toeplitz adaptations for the particular method. 
Each of these adaptations also comes equipped with a custom BERT model with the appropriate LoRA or adapter layers. So to run any GLUE task, one simply has to import the right model class from the appropriate folder. An example of circulant_adapter is shown in scripts/run_bert_glue.py
