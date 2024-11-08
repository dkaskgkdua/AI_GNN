# GNN(Graph Neural Network)

## What is a Graph?
- 객체간의 관계를 표현하는 것이다.
- 각 객체는 node or vertex로 표현된다.
- 관계는 edge로 표현된다.
- Graph는 엔티티를 관계와 상호작용과 함께 설명하고 분석하기 위한 일반적인 언어이다.

## 현실 세계에서 Graph Data 예시
- Social Networks
- Traffic, Subway Maps
- Computer Networks
- Protein-Protein Interaction
- Chemical Compounds
- Knowledge Graphs
- Citation Networks
- Networks of Neurons

## Graph 개념이 머신러닝에서 발전된 과정
|-|-|
|**Category**|**Examples**|
|Traditional Methods|Graphlets, Graph Kernels|
|Node embedding methods|DeepWalk,node2vec|
|Graph Neural Networks(GNNs)|GCN, GraphSage, GAT..|

### Node Embedding(Node Representation Learning)
그래포 G = (V, E) 가 주어졌을 때 아래와 같이 노드 임베딩을 만든다.
![node_embedding.jpg](./images/node_embedding/node_embedding.jpg)
그리고 Graph에서  가깝거나 같은 그룹인 경우 N차원 공간에서도 유사하게 위치하는게 목표다.
![node_embedding2.jpg](./images/node_embedding/node_embedding2.jpg)

그렇다면 유사하게 위치하기 위해서는 유사도를 구해야 하는데 내적, 코사인유사도 같은 방법으로 계산한다.

### Transductive vs Inductive
임베딩을 학습하는 방법은 2가지이다.
1. Transductive : 학습결과가 노드 임베딩 그 자체인 것(= 노드임베딩이 파라미터)
   - 학습을 하는 동안 보지 못한 노드는 임베딩을 못 만든다.
   - 필요한 파라미터 수가 노드 수에 비례해서 커지기 때문에 확장성이 없다.
   - 테스트 데이터가 학습 과정에서 그래프에 포함되어 있음.
   - node feature를 node embedding에 포함하기 힘들다.
2. Inductive : 학습결과가 encoder인 것(= 노드임베딩을 만드는 함수를 구한다.)
   - 못 본 노드도 임베딩을 생성 가능하다. (Transductive의 대안으로 나옴)
   - 학습에서 훈련 중에는 테스트데이터에 접근하지 않음.(2020,2021년 데이터 훈련, 2022년 데이터 테스트)
   - 이러한 기술이 바로 GNN이다.

|-|-|-|
|특징|Transductive|Inductive|
|테스트노드|학습 중에 그래프에 포함됨|학습 중에 포함되지 않음|
|일반화|새로운 노드에 대한 예측 불가|새로운 노드에 대한 예측 가능|
|그래프 구조 활용|테스트 노드의 구조적 정보 사용 가능|훈련 시 주어진 그래프 구조만 사용|
|적용 예시|노드 분류 문제|새로운 그래프나 노드 예측|
|응용 가능성|고정된 환경에서 성능 우수|유연성이 요구되는 상황에 적합|

## GNN
### GNN Intro
그패르 구조 데이터를 처리하고 분석하는데 사용되는 인공지능 모델이다. 그래프는 node edge로 구성되며, 각각의 node는 entity를, edge는 relationship을 나타낸다.
복잡한 구조적 데이터를 처리하는데 적합하다.
- grid or sequence 같이 표현이 정형화되지 않을 때
- 관계 구조를 어떻게 활용해서 더 나은 예측이 필요할 때

### GNN Task Type
- Graph-level
    <img src="./images/gnn/loss_function1.jpg" width="450" />
  - Graph generation(바이오,화학에서 새로운 화합물,유기물 조합 시)
  - Graph classification(각 그래프들의 카테고리 분류)
  - prediction heads: 모든 node embedding을 사용해서 예측을 수행하는 모듈
    - 그래프 대표 벡터를 만드는데 mean pooling, max pooling, sum pooling과 같은 방법이 있는데 그래프 크기가 커질 경우 정보 손실의 우려가 있다.(합계가 같다는지..)
    - hierarchical pooling을 사용하면 이러한 케이스를 방지 가능하다.
- Node-level
  - Node embedding/Node classification(단백질 접힘 문제, e커머스 유저/상품 관계)
  - prediction heads: node embedding을 사용해서 예측을 수행하는 모듈
- Edge-level
  - Link prediction(entity간에 relationship이 있는지?)
    - self-supervised, unsupervised task
    - 기존 그래프에서 엣지 제거 후 제거된 엣지를 label로 주고 학습이 가능하다.
    - ![loss_function2.jpg](./images/gnn/loss_function2.jpg)
  - prediction heads: node embedding pair를 사용해서 예측을 수행하는 모듈(결합 방식 2개)
    - concatenation + linear : 노드 임베딩을 연결 후 선형 변환하여 상호작용을 학습
    - dot product : 두 임베딩이 얼마나 비슷한지를 측정

### 학습 loss
- classification loss
  - cross entropy 사용
- regression loss
  - mean squared error(MSE) 사용

### 학습 평가(evaluation)
- classification
  - multi-class classification
    - accuracy
  - binary classification
    - accuracy, precision/recall/f1-score
    - roc curve(TPR, FPR 관계 시각화- y축 상단일수록 좋음), roc auc(roc 곡선 아래 면적 평가)
- regression
  - RMSE(Root mean square error)
  - MAE(Mean absolute error)

### Dataset split
- transductive setting
  - training / validation / test set가 같은 그래프 내에 있다.
  - 모든 dataset split에서 전체 그래프가 관찰 가능하며 오직 라벨만 나눠져있다.
  - node/edge 예측 수행이 가능
- inductive setting
  - 각 dataset은 각각의 그래프가 있다.
  - node/edge/graph 예측 수행이 가능
- Link Prediction 케이스
  - Message edge, supervision edge로 나눈다.
  - inductive split
    - message, supervision edge가 각각 다른 3가지의 그래프를 만든다(training, validation, test)
    - ![link_prediction.jpg](./images/link_prediction.jpg)
  - transductive split
    - 각 edge를 training message edge, training supervision edge, validation edge, test edge로 나눈다.
    - ![link_prediction2.jpg](./images/link_prediction2.jpg)
    - 그래프가 너무 커지면 아예 메모리에 못올라가는 케이스 발생, sub graph 활용
    - 각 엣지별 비율은 7:1:1:1 정도 됨

### Example
![protein_folding_problem.png](./images/gnn/protein_folding_problem.png)
![recommender_system.png](./images/gnn/recommender_system.png)

#### 지도학습 및 비지도학습 개념이 들어간다면?(저자 네트워크 예시)
![semi-supervised.jpg](./images/gnn/node_classification.jpg)
semi는 정답이 일부주어지고 안주어진 노드의 특성을 이용하여 label을 예측
![unsupervised.jpg](./images/gnn/unsupervised.jpg)
![supervised.jpg](./images/gnn/supervised.jpg)
- Node classification은 semi-supervised, supervised가 있다.

### Graph에서 Convolution을 적용가능한가?
- CNN은 유클리드 구조화된 데이터에서 작동하도록 구축되었음.
- GCN은 불규칙하고 비유클리드 데이터에 대한 CNN의 일반화된 버전이다.
  - 이웃의 수는 다양하다.
  - 노드의 이웃은 순서가 없다.
- 즉, CNN의 아키텍처가 GNN 쪽에서도 적용이 가능하며 이를 GCN이라 함.
  ![gcn.png](./images/gnn/gcn.jpg)

### 작동 방식
GNN은 그래프의 각 노드가 인접한 노드의 정보와 관계를 바탕으로 자신의 상태를 업데이트하는 과정을 반복한다.
이 과정을 통해 노드와 그래프 전반에 대한 표현을 학습하게 된다.
![architecture.jpg](./images/gnn/architecture.jpg)
![architecture2.jpg](./images/gnn/architecture2.jpg)
순차적으로 layer를 쌓고 input(node feature)를 넣으면 output(node embedding)이 출력되는 구조.

- Message Passing: 각 노드는 이웃 노드의 정보를 받아서 자신의 상태를 갱신한다.(Message Passing Function 필요)
![message_func.png](./images/gnn/message_func.png)
![message_for.png](./images/gnn/message_for.png)
- Aggregation: 이웃 노드에서  받은 정보를 요약하거나 집계하여 자신의 상태에 반영한다.(sum, mean, max 등)
![aggregate.png](./images/gnn/aggregate.png)
![aggregation_for.png](./images/gnn/aggregation_for.png)
- Update: Aggregation 과정을 거친 정보를 바탕으로 노드의 상태를 갱신한다.
- Combine: 이웃으로부터 정보를 받으며 자신의 정보가 손실되는 경우가 있어서 추가해준다.(aggregation + target node)
![combine.png](./images/gnn/combine.png)
- activation func(nonlinearity): message or aggregation이 있을 때마다 추가될 수 있음

> Inductive Capability
> 앞서 Transductive는 파라미터 수가 노드 수에 비례해서 커지기 때문에 확장성이 없다고 했는데
> Inductive의 경우 아래와 같은 구조이며 같은 통합 파라미터를 공유한다.
> ![inductive_capability.jpg](./images/gnn/inductive_capability.jpg)

## GCN
GCN(Graph Convolutional Network)은 그래프 데이터를 처리하기 위한 신경망 모델로, 그래프의 구조를 반영하여 노드 간의 관계를 학습하는 방식입니다. 전통적인 CNN(Convolutional Neural Network)이 2D 이미지 데이터를 처리하는 데 강점을 보이듯, GCN은 그래프와 같은 비유클리드 데이터에서 성능을 발휘합니다. GCN은 그래프의 노드, 엣지, 인접 노드들의 정보를 학습하여, 노드 또는 그래프의 임베딩을 생성하거나 예측 문제를 해결하는 데 사용됩니다.
![gcn_for.png](./images/gnn/gcn_for.jpg)
위 예시는 노드가 1개 홉일 경우이며 k-hop일 경우는 아래와 같다.
![k-hop.jpg](./images/gnn/k-hop.jpg)

GCN의 layer가 너무 깊어질 경우 정보 수집 노드가 많아지고 평균을 내다보니 결국 임베딩이 비슷해져서 정확도가 떨어진다.

### gcn-layer asymmetric normalization
![gcn_layer.jpg](./images/gnn/gcn_layer.jpg)
Asymmetric Normalization은 인접 행렬 A를 정규화할 때 노드의 출발점 또는 도착점의 정도(degree)를 한쪽만 사용하는 방식입니다.

- 수식
  - <img src="./images/gnn/asymmetric_normalization.jpg" width="150" />
  - D^-1^: 차수의 역수, A: 연결여부, H^k-1^: 직전 hidden state 
  - self loop가 더해지면 이웃에 자기자신도 있다고 본다. D^-1^에서 차수 역수가 1 늘어남.


### gcn symmetric normalization
노드의 degree를 양쪽 모두 고려하여 정규화하는 방식입니다. 이는 그래프의 양쪽 degree에 대한 정보를 포함하여 대칭적 정규화를 적용합니다.

- 수식
  ![gcn_final.jpg](./images/gnn/gcn_final.jpg)
  

> 수식들을 보면 이웃 노드의 수로 나눠준다.
> 이유는 메세지를 받는데 이웃이 많은 쪽으로 영향을 많이 받을 수 있음.
> 즉 SNS에서 인플루언서가 있고 팔로워 적은 친한 친구가 있는데 어떤 이웃이 더 중요하냐? 인플루언서가 더 중요하거나 나의 특성을 대변해주지는 않음.

### 도전과제
1. Over-Smoothing: GCN layer가 쌓이다보면 노드 임베딩이 서로 비슷해져 분간하기가 어려워짐
   - receptive field(수용 영역)은 서로 다른 두 노드에 얼마나 많은 이웃을 공유하는가이다.
   - receptive field는 선형적이 아닌 기하급수적으로 커지는데 유사한 노드 임베딩을 만들게 되어 over-smoothing의 원인이 된다.
   - 해결방안
     - residual connection으로 해결 가능(적극적으로 모든 단계에 연결 가능)
       - ![residual_connection.jpg](./images/residual_connection.jpg)
     - layer 쌓는 것을 조심스렇게 한다.(도메인별로 다르며 적당한 지점을 찾는다.)
     - 각 layer의 표현력을 증가시킨다.
     - message를 전달하지 않는 레이어를 추가한다.
       - MLP(multi-layer perceptron) layer를 추가해서 표현력을 향상시킨다(+비선형성)
2. 멀리 떨어진 노드에서 전달된 정보가 특정 노드에 도달하면서 과도하게 압축되어 정보가 왜곡되거나 손실되는 현상이다.(정보 병목)
   - residual connection으로 해결 가능


## GAT(Graph Attention Network)
그래프 데이터에서 노드 간의 중요도를 동적으로 학습하여 정보를 전달하는 그래프 신경망(GNN) 모델입니다.
GAT는 특정 노드에 대해 더 중요한 이웃 노드의 정보를 더욱 강조하면서, 그 중요도가 낮은 이웃의 정보는 덜 강조합니다.

- 수식
  - $$a_vu$$: attention weight
  - <img src="./images/gnn/gat.jpg" width="400" />

  
## GraphSAGE
그래프 데이터에서 대규모 그래프의 효율적인 학습을 위해 고안된 대표적인 그래프 신경망(GNN) 모델입니다.
일반적인 GCN과 달리, 이웃 노드를 샘플링하여 그래프의 부분 구조에서 정보를 추출하고, 이를 통해 노드의 임베딩을 학습합니다.

- 핵심 아이디어
  - GCN에서 노드 임베딩을 훈련하는 대신 집계 함수를 학습합니다.(속성을 사용해 보이지 않은 데이터에 대한 표현을 생성-inductive)
  - 노드가 모든 이웃을 탐색하는 대신, 정해진 수의 이웃 노드만 무작위로 샘플링
  - 집계함수(Aggregation)
    - Mean Aggregator: 샘플링된 이웃 노드 특성의 평균을 구함.
    - LSTM Aggregator: LSTM을 사용해 이웃 노드의 특성을 순차적으로 처리하여 정보를 집계.
    - Pooling Aggregator: 각 이웃의 특성에 비선형 함수를 적용하고, 최대값 또는 평균값을 취함.

> Graph 동형사상
> 두 그래프가 있고, 1대1 매핑, 매핑 시 label이 같고, 간선이 같다면 이는 동형체라고 할 수 있다.
> 난이도가 높은 문제이다.
> <img src="./images/gnn/동형사상.jpg" width="200" />

## GIN
동형사상 문제에서 정답에 가깝게 근사라도 하자는 목표를 가졌다. 다만 이웃 노드의 정보를 단순히 평균, 최대값 등의 방식으로 집계해서 구조가 다른 그래프도 동일하게 처리할 수 있다는 한계가 있다.
- 집계함수에 합계 사용함
- ![gin.jpg](./images/gin.jpg)


## 교수님 TIP
- 데이터 전처리가 중요하며, input 범위를 제한하고 정규화가 필요하다.
- 최적화는 ADAM이 괜찮다
- activation func은 ReLU(LeakyReLU), SWISH가 많이 쓰인다
- 모든 레이어에 bias term을 포함하다
- embedding dimension은 32, 64, 128이 적당하다.