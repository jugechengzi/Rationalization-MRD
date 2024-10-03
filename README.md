## Environments
torch 1.13.1+cu11.6.  
python 3.7.16.   
RTX4090  

## Datasets 

Beer: you can get it [here](http://people.csail.mit.edu/taolei/beer/). Then place it in the ./data/beer directory.  
Hotel: you can get it [here](https://people.csail.mit.edu/yujia/files/r2a/data.zip). 
Then  find hotel_Location.train, hotel_Location.dev, hotel_Service.train, hotel_Service.dev, hotel_Cleanliness.train, hotel_Cleanliness.dev from data/oracle and put them in the ./data/hotel directory. 
Find hotel_Location.train, hotel_Service.train, hotel_Cleanliness.train from data/target and put them in the ./data/hotel/annotations directory.  
Word embedding: [glove.6B.100d.txt](https://nlp.stanford.edu/projects/glove/). Then put it in the ./data/hotel/embeddings directory.

## An example about how to run the code

For Beer-Appearance with sparsity being about 10\%, use the following code to get the results of MRD in Table 1 (with random seed=1):  
python -u adv_causal.py --model_type sp --seed 1 --gen_sparse 1 --div kl --correlated 1 --data_type beer --lr 0.0001 --batch_size 128 --gpu 1 --sparsity_percentage 0.075 --sparsity_lambda 6 --continuity_lambda 6 --epochs 150 --aspect 0
