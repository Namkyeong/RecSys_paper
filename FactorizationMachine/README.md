# Wide & Deep Learning for Recommender Systems
Pytorch Implementation of 'Factorization Machine' on Movielens Dataset   
> Rendle, Steffen. "Factorization machines." 2010 IEEE International Conference on Data Mining. IEEE, 2010.


* [paper](https://ieeexplore.ieee.org/document/5694074)

## Experimental Results
Experiments on Movielens dataset  
![exp_1](./img/exp_1.PNG)
![exp_2](./img/exp_2.PNG)

## Future works
처음에는 embedding layer를 사용해서 feature를 임베딩하려고 했지만, user와 item을 제외한 다른 변수에 대한 embedding을 고려했을때 문제가 발생하여 matrix를 정의하고 pytorch로 학습을 했음.  
Matrix Factorization using Embedding Layer